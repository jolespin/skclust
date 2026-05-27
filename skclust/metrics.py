# -*- coding: utf-8 -*-
# skclust/metrics.py
"""
Cluster validation metrics for continuous and binary features.

These functions evaluate cluster quality by measuring within-cluster compactness
and between-cluster separation.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import (
    median_abs_deviation,
    entropy, 
    variation,
)
from tqdm.auto import tqdm
from loguru import logger
# from scipy.stats.contingency import association


def clustered_cosine_distances(X_l2, labels, check=True, chunk_size=None, verbose=0):
    """
    Compute intra-cluster and inter-cluster cosine distance summaries
    for L2-normalized embeddings.
    
    Parameters
    ----------
    X_l2 : array-like of shape (n, m)
        L2-normalized embeddings (DataFrame or ndarray)
    labels : array-like of shape (n,)
        Cluster labels (Series or ndarray)
    check : bool, default True
        Verify that X_l2 is L2-normalized
    chunk_size : int or None, default None
        Row chunk size for inter-cluster matrix multiplications to limit peak memory.
    verbose : int, default 0
        If > 0, log warnings for singleton or empty clusters.
    
    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (intra-cluster|inter-cluster) × (n_pairs|mean|median|std|mad)
        Plus a top-level 'size' column.
        Index: cluster labels
    """
    # Type concordance checks
    _X_is_df = isinstance(X_l2, pd.DataFrame)
    _labels_is_series = isinstance(labels, pd.Series)
    if _X_is_df and not _labels_is_series:
        raise TypeError("X_l2 is a DataFrame but labels is not a Series")
    if _labels_is_series and not _X_is_df:
        raise TypeError("labels is a Series but X_l2 is not a DataFrame")

    # Index alignment check for pandas inputs
    if _X_is_df and _labels_is_series:
        if not X_l2.index.equals(labels.index):
            raise ValueError(
                f"X_l2.index and labels.index do not match. "
                f"X_l2 has {len(X_l2.index)} entries, labels has {len(labels.index)} entries, "
                f"with {len(X_l2.index.difference(labels.index))} in X_l2 only and "
                f"{len(labels.index.difference(X_l2.index))} in labels only"
            )

    # Coerce to numpy
    if _X_is_df:
        X_l2 = X_l2.values
    if _labels_is_series:
        labels = labels.values
    X_l2 = np.asarray(X_l2, dtype=np.float32)
    labels = np.asarray(labels)

    assert X_l2.shape[0] == labels.shape[0], \
        f"X_l2 rows ({X_l2.shape[0]}) != labels length ({labels.shape[0]})"

    if check:
        norms = np.linalg.norm(X_l2, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(
                f"X_l2 is not L2-normalized. "
                f"Norm range: [{norms.min():.6f}, {norms.max():.6f}]"
            )

    metrics = ["n_pairs", "mean", "median", "std", "mad"]
    unique_labels = np.unique(labels)

    def _summarize(distances):
        return {
            "n_pairs": distances.size,
            "mean": np.mean(distances),
            "median": np.median(distances),
            "std": np.std(distances, ddof=1),
            "mad": median_abs_deviation(distances),
        }

    results = dict()

    for id_cluster in tqdm(unique_labels, desc="Clusters", unit="cluster"):
        mask = labels == id_cluster
        A = X_l2[mask]
        B = X_l2[~mask]
        n_cluster = A.shape[0]

        row = {("size", "n"): n_cluster}

        # --- Intracluster ---
        if n_cluster < 2:
            if verbose > 0:
                logger.warning(f"Cluster '{id_cluster}' has {n_cluster} member(s); intra-cluster stats will be NaN")
            for m in metrics:
                row[("intra-cluster", m)] = np.nan
        else:
            D = 1 - A @ A.T
            np.fill_diagonal(D, 0)
            intra_distances = squareform(D, checks=False)
            for m, v in _summarize(intra_distances).items():
                row[("intra-cluster", m)] = v
            del D, intra_distances

        # --- Intercluster ---
        if B.shape[0] == 0:
            if verbose > 0:
                logger.warning(f"Cluster '{id_cluster}' has no out-group points; inter-cluster stats will be NaN")
            for m in metrics:
                row[("inter-cluster", m)] = np.nan
        elif chunk_size is not None:
            chunks = []
            for i in range(0, n_cluster, chunk_size):
                chunk = 1 - A[i:i + chunk_size] @ B.T
                chunks.append(chunk.ravel())
            inter_distances = np.concatenate(chunks)
            for m, v in _summarize(inter_distances).items():
                row[("inter-cluster", m)] = v
            del chunks, inter_distances
        else:
            inter_distances = (1 - A @ B.T).ravel()
            for m, v in _summarize(inter_distances).items():
                row[("inter-cluster", m)] = v
            del inter_distances

        results[id_cluster] = row

    df_results = pd.DataFrame.from_dict(results, orient="index")
    df_results.columns = pd.MultiIndex.from_tuples(df_results.columns)
    df_results.index.name = "id_cluster"

    return df_results

# ============================================================================
# CONTINUOUS FEATURES
# ============================================================================
def cv_score(X: pd.DataFrame, labels: pd.Series, checks: bool = True, atol: float = 1e-3) -> pd.Series:
    """
    Compute coefficient of variation (CV) within each cluster for continuous features.
    
    Lower CV = more compact clusters (members are similar).
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features, continuous values).
        Values should be strictly positive for CV to be meaningful.
    labels : pd.Series
        Cluster assignments (index = sample IDs)
    checks : bool, default=True
        If True, raise ValueError when features violate CV assumptions
        (non-positive values or near-zero means). If False, emit a warning
        and proceed.
    atol : float, default=1e-3
        Absolute tolerance for near-zero mean detection. Features with
        |mean| < atol are flagged as unreliable for CV computation.
        
    Returns
    -------
    pd.Series
        Mean CV for each cluster (cluster_id -> mean CV across features)
        
    Warnings
    --------
    CV is undefined when feature means approach zero. This metric is only
    appropriate for strictly positive features (e.g., abundances, proportions,
    completion ratios). For zero-centered or mixed-sign features, use
    eta_squared_score instead.
        
    Examples
    --------
    >>> cv = cv_score(X_continuous, labels)
    >>> print(f"Mean CV: {cv.mean():.3f}")  # Lower = better
    """
    def _mean_cv(group):
        """Mean coefficient of variation across features for a single cluster."""
        return np.nanmean(variation(group.values, axis=0, nan_policy='omit'))

    # Subset to overlapping indices
    index_overlap = labels.index.intersection(X.index)
    X = X.loc[index_overlap]
    labels = labels.loc[index_overlap]
    
    # Check for non-positive values
    has_non_positive = (X.values <= 0).any()
    
    # Check for near-zero feature means
    feature_means = X.mean(axis=0)
    near_zero_features = feature_means.index[np.abs(feature_means) < atol].tolist()
    
    if has_non_positive or near_zero_features:
        msg = (
            "CV is unreliable when features contain non-positive values or "
            "have near-zero means. Consider using eta_squared_score instead."
        )
        if near_zero_features:
            msg += f" Near-zero mean features (atol={atol}): {near_zero_features[:5]}"
            if len(near_zero_features) > 5:
                msg += f" (and {len(near_zero_features) - 5} more)"
        if checks:
            raise ValueError(msg)
        else:
            warnings.warn(msg, stacklevel=2)
    
    # Group by cluster and compute CV for each feature
    cv_per_cluster = X.groupby(labels).apply(_mean_cv)
    
    return cv_per_cluster

def _eta_squared_trace(X: pd.DataFrame, labels: pd.Series) -> float:
    """
    Trace-based η² (Calinski-Harabasz / MANOVA style).
    
    Variance-weighted: features with higher total variance contribute more.
    η² = Σⱼ between_SSⱼ / Σⱼ total_SSⱼ
    """
    grand_means = X.mean(axis=0).values
    total_ss = ((X.values - grand_means) ** 2).sum(axis=0)
    
    within_ss = np.zeros(X.shape[1])
    for _, group in X.groupby(labels):
        group_means = group.mean(axis=0).values
        within_ss += ((group.values - group_means) ** 2).sum(axis=0)
    
    between_ss = total_ss - within_ss
    
    total_ss_sum = total_ss.sum()
    if total_ss_sum == 0:
        return 0.0
    
    return between_ss.sum() / total_ss_sum


def _eta_squared_per_feature(X: pd.DataFrame, labels: pd.Series) -> float:
    """
    Per-feature average η².
    
    Equal-weighted: each feature contributes equally regardless of scale.
    η² = meanⱼ(between_SSⱼ / total_SSⱼ)
    """
    grand_means = X.mean(axis=0).values
    total_ss = ((X.values - grand_means) ** 2).sum(axis=0)
    
    within_ss = np.zeros(X.shape[1])
    for _, group in X.groupby(labels):
        group_means = group.mean(axis=0).values
        within_ss += ((group.values - group_means) ** 2).sum(axis=0)
    
    between_ss = total_ss - within_ss
    
    valid = total_ss > 0
    if not valid.any():
        return 0.0
    
    eta2_per_feature = np.where(valid, between_ss / total_ss, 0.0)
    return eta2_per_feature[valid].mean()


def eta_squared_score(X: pd.DataFrame, labels: pd.Series, method: str = "trace") -> float:
    """
    Compute eta-squared (η²) for continuous features.
    
    Eta-squared is the ratio of between-cluster variance to total variance,
    equivalent to R² in one-way ANOVA.
    
    Higher η² = clusters are more separated.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features, continuous values)
    labels : pd.Series
        Cluster assignments (index = sample IDs)
    method : str, default="trace"
        - "trace": Trace-based η² (Calinski-Harabasz / MANOVA style).
          Variance-weighted average across features. Standard formulation
          for multivariate cluster validation.
        - "per_feature": Equal-weighted mean of per-feature η² values.
          Each feature contributes equally regardless of scale.
        
    Returns
    -------
    float
        Eta-squared (0-1, higher = more separated clusters)
        
    Notes
    -----
    Interpretation:
    - η² = 0.0: Clusters have same feature profiles (no separation)
    - η² = 0.3: 30% of variance explained by cluster membership
    - η² = 1.0: Clusters have completely different profiles (perfect separation)
    
    The "trace" method upweights high-variance features, which for binary
    data means features near 50% prevalence contribute more. The "per_feature"
    method treats all non-constant features equally.
    """
    # Subset to overlapping indices
    index_overlap = labels.index.intersection(X.index)
    X = X.loc[index_overlap]
    labels = labels.loc[index_overlap]
    
    if method == "trace":
        return _eta_squared_trace(X, labels)
    elif method == "per_feature":
        return _eta_squared_per_feature(X, labels)
    else:
        raise ValueError(f"Unknown method: {method!r}. Expected 'trace' or 'per_feature'.")



# ============================================================================
# BINARY FEATURES
# ============================================================================

def entropy_score(X: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """
    Compute Shannon entropy within each cluster for binary features.
    
    Lower entropy = more consistent features within cluster.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features, binary 0/1 values)
    labels : pd.Series
        Cluster assignments (index = sample IDs)
        
    Returns
    -------
    pd.Series
        Mean Shannon entropy per cluster (cluster_id -> mean entropy across features)
        
    Notes
    -----
    Interpretation:
    - Entropy = 0: All samples in cluster have same feature value (perfect consistency)
    - Entropy = 0.5: Mostly consistent (e.g., 90% positive, 10% negative)
    - Entropy = 1.0: Maximum uncertainty (50% positive, 50% negative)
    
    Examples
    --------
    >>> entropy_vals = entropy_score(X_binary, labels)
    >>> print(f"Mean entropy: {entropy_vals.mean():.3f}")  # Lower = better
    >>> # Target: mean entropy < 0.50 for consistent clusters
    """
    # Subset to overlapping indices
    index_overlap = labels.index.intersection(X.index)
    X = X.loc[index_overlap]
    labels = labels.loc[index_overlap]
    
    # For binary data, we can compute entropy directly from proportions
    # Binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
    def binary_entropy(p):
        """Compute binary entropy, handling edge cases."""
        # Avoid log(0) by clipping
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    
    # Group by cluster and compute mean proportion for each feature (vectorized)
    # This gives us p for each (cluster, feature) combination
    proportions = X.groupby(labels).mean()  # Fast: single groupby operation
    
    # Compute entropy for all (cluster, feature) pairs at once (vectorized)
    entropies = binary_entropy(proportions.values)
    
    # Take mean across features for each cluster
    entropy_per_cluster = pd.Series(
        entropies.mean(axis=1),
        index=proportions.index
    )
    
    return entropy_per_cluster

def cramers_v_score(X: pd.DataFrame, labels: pd.Series) -> float:
    """
    Compute Cramér's V for binary features (optimized vectorized version).
    
    Cramér's V measures association between cluster membership and feature values.
    
    Higher V = clusters are more separated.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features, binary 0/1 values)
    labels : pd.Series
        Cluster assignments (index = sample IDs)
        
    Returns
    -------
    float
        Mean Cramér's V (0-1, higher = more separated clusters)
        
    Notes
    -----
    Interpretation:
    - V = 0.0: Feature is completely independent of cluster (clusters not separated)
    - V = 0.3: Knowing cluster improves feature prediction by ~30% (moderate)
    - V = 1.0: Feature perfectly predicts cluster (perfect separation)
    """
    # Subset to overlapping indices
    index_overlap = labels.index.intersection(X.index)
    X = X.loc[index_overlap]
    labels = labels.loc[index_overlap]
    
    n_samples = len(labels)
    X_array = X.values.astype(int)
    
    # Create cluster index mapping
    unique_labels, cluster_indices = np.unique(labels.values, return_inverse=True)
    n_clusters = len(unique_labels)
    
    # One-hot encode cluster assignments: (n_samples, n_clusters)
    one_hot = np.zeros((n_samples, n_clusters), dtype=int)
    one_hot[np.arange(n_samples), cluster_indices] = 1
    
    # Counts of 1s per (cluster, feature) via matrix multiply: (n_clusters, n_features)
    counts_ones = one_hot.T @ X_array
    cluster_sizes = one_hot.sum(axis=0)  # (n_clusters,)
    counts_zeros = cluster_sizes[:, None] - counts_ones
    
    # Contingency tables: (n_clusters, 2, n_features)
    contingency = np.stack([counts_zeros, counts_ones], axis=1)
    
    # Filter constant features (must have both 0s and 1s)
    total_ones = counts_ones.sum(axis=0)
    valid_features = (total_ones > 0) & (total_ones < n_samples)
    
    if not valid_features.any():
        return 0.0
    
    contingency = contingency[:, :, valid_features]
    
    # Vectorized chi-square computation
    row_totals = contingency.sum(axis=1, keepdims=True)
    col_totals = contingency.sum(axis=0, keepdims=True)
    expected = (row_totals * col_totals) / n_samples
    expected = np.maximum(expected, 1e-10)
    
    chi2 = ((contingency - expected) ** 2 / expected).sum(axis=(0, 1))
    
    # Cramér's V for binary features: V = sqrt(chi2 / n)
    cramers_v_values = np.sqrt(chi2 / n_samples)
    
    return cramers_v_values.mean()



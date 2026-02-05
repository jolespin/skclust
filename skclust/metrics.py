# -*- coding: utf-8 -*-
# skclust/metrics.py
"""
Cluster validation metrics for continuous and binary features.

These functions evaluate cluster quality by measuring within-cluster compactness
and between-cluster separation.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy, variation
# from scipy.stats.contingency import association


# ============================================================================
# CONTINUOUS FEATURES
# ============================================================================

def cv_score(X: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """
    Compute coefficient of variation (CV) within each cluster for continuous features.
    
    Lower CV = more compact clusters (members are similar).
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features, continuous values)
    labels : pd.Series
        Cluster assignments (index = sample IDs)
        
    Returns
    -------
    pd.Series
        Mean CV for each cluster (cluster_id -> mean CV across features)
        
    Examples
    --------
    >>> cv = cv_score(X_continuous, labels)
    >>> print(f"Mean CV: {cv.mean():.3f}")  # Lower = better
    >>> # Target: mean CV < 0.35 for compact clusters
    """
    # Subset to overlapping indices
    index_overlap = labels.index.intersection(X.index)
    X = X.loc[index_overlap]
    labels = labels.loc[index_overlap]
    
    # Group by cluster and compute CV for each feature
    def compute_cv(group):
        """Compute mean CV across all features using scipy.stats.variation."""
        # variation computes CV for each column (axis=0)
        cv_per_feature = variation(group.values, axis=0, nan_policy='omit')
        return np.nanmean(cv_per_feature)  # Average across features
    
    cv_per_cluster = X.groupby(labels).apply(compute_cv)
    
    return cv_per_cluster


# def _eta_squared_continuous(X: pd.DataFrame, labels: pd.Series) -> float:
#     """
#     Compute eta-squared (η²) for continuous features.
    
#     Eta-squared is the ratio of between-cluster variance to total variance,
#     equivalent to R² in one-way ANOVA.
    
#     Higher η² = clusters are more separated.
    
#     Parameters
#     ----------
#     X : pd.DataFrame
#         Feature matrix (samples x features, continuous values)
#     labels : pd.Series
#         Cluster assignments (index = sample IDs)
        
#     Returns
#     -------
#     float
#         Eta-squared (0-1, higher = more separated clusters)
        
#     Notes
#     -----
#     Interpretation:
#     - η² = 0.0: Clusters have same feature profiles (no separation)
#     - η² = 0.3: 30% of variance explained by cluster membership
#     - η² = 1.0: Clusters have completely different profiles (perfect separation)
#     """
#     # Subset to overlapping indices
#     index_overlap = labels.index.intersection(X.index)
#     X = X.loc[index_overlap]
#     labels = labels.loc[index_overlap]
    
#     # Total sum of squares (variance around grand mean)
#     grand_mean = X.values.flatten().mean()
#     total_sum_of_squares = ((X - grand_mean) ** 2).values.sum()
    
#     # Within-cluster sum of squares (variance around each cluster's mean)
#     def within_sum_of_squares(group):
#         group_mean = group.values.flatten().mean()
#         return ((group - group_mean) ** 2).values.sum()
    
#     within_cluster_sum_of_squares = X.groupby(labels).apply(within_sum_of_squares).sum()
    
#     # Between-cluster sum of squares
#     between_cluster_sum_of_squares = total_sum_of_squares - within_cluster_sum_of_squares
    
#     # Eta-squared (proportion of variance explained by cluster membership)
#     eta_squared = between_cluster_sum_of_squares / total_sum_of_squares if total_sum_of_squares > 0 else 0
    
#     return eta_squared


# def eta_squared_score(X: pd.DataFrame, labels: pd.Series) -> float:
#     """
#     Compute eta-squared (η²) for continuous features.
    
#     Eta-squared measures the proportion of variance explained by cluster membership.
    
#     Higher η² = clusters are more separated.
    
#     Parameters
#     ----------
#     X : pd.DataFrame
#         Feature matrix (samples x features, continuous values)
#     labels : pd.Series
#         Cluster assignments (index = sample IDs)
        
#     Returns
#     -------
#     float
#         Eta-squared (0-1, higher = more separated clusters)
        
#     Notes
#     -----
#     Interpretation:
#     - η² = 0.0: Clusters have same feature profiles (no separation)
#     - η² = 0.3: 30% of variance explained by cluster membership
#     - η² = 1.0: Clusters have completely different profiles (perfect separation)
    
#     Examples
#     --------
#     >>> eta_sq = eta_squared_score(X_continuous, labels)
#     >>> print(f"Eta-squared: {eta_sq:.3f}")  # Higher = better
#     >>> # Target: η² > 0.30 for separated clusters
#     """
#     return _eta_squared_continuous(X, labels)
def _eta_squared_continuous(X: pd.DataFrame, labels: pd.Series) -> float:
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
    """
    # Subset to overlapping indices
    index_overlap = labels.index.intersection(X.index)
    X = X.loc[index_overlap]
    labels = labels.loc[index_overlap]
    
    # Total sum of squares
    grand_mean = X.values.mean()
    total_sum_of_squares = ((X.values - grand_mean) ** 2).sum()
    
    if total_sum_of_squares == 0:
        return 0.0
    
    # Within-cluster sum of squares
    def within_ss(group):
        group_mean = group.values.mean()
        return ((group.values - group_mean) ** 2).sum()
    
    within_cluster_sum_of_squares = X.groupby(labels).apply(within_ss).sum()
    
    # Between-cluster sum of squares
    between_cluster_sum_of_squares = total_sum_of_squares - within_cluster_sum_of_squares
    
    # Eta-squared
    eta_squared = between_cluster_sum_of_squares / total_sum_of_squares
    
    return eta_squared


def eta_squared_score(X: pd.DataFrame, labels: pd.Series) -> float:
    """
    Compute eta-squared (η²) for continuous features.
    
    Eta-squared measures the proportion of variance explained by cluster membership.
    
    Higher η² = clusters are more separated.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features, continuous values)
    labels : pd.Series
        Cluster assignments (index = sample IDs)
        
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
    
    Examples
    --------
    >>> eta_sq = eta_squared_score(X_continuous, labels)
    >>> print(f"Eta-squared: {eta_sq:.3f}")  # Higher = better
    >>> # Target: η² > 0.30 for separated clusters
    """
    return _eta_squared_continuous(X, labels)


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


# def _cramers_v_binary(X: pd.DataFrame, labels: pd.Series) -> float:
#     """
#     Compute Cramér's V for binary features (optimized vectorized version).
    
#     Cramér's V measures association between cluster membership and feature values.
    
#     Higher V = clusters are more separated.
    
#     Parameters
#     ----------
#     X : pd.DataFrame
#         Feature matrix (samples x features, binary 0/1 values)
#     labels : pd.Series
#         Cluster assignments (index = sample IDs)
        
#     Returns
#     -------
#     float
#         Mean Cramér's V (0-1, higher = more separated clusters)
        
#     Notes
#     -----
#     Interpretation:
#     - V = 0.0: Feature is completely independent of cluster (clusters not separated)
#     - V = 0.3: Knowing cluster improves feature prediction by ~30% (moderate)
#     - V = 1.0: Feature perfectly predicts cluster (perfect separation)
#     """
#     # Subset to overlapping indices
#     index_overlap = labels.index.intersection(X.index)
#     X = X.loc[index_overlap]
#     labels = labels.loc[index_overlap]
    
#     n_samples = len(labels)
    
#     # Convert to numpy for speed
#     X_array = X.values.astype(int)
    
#     # Create cluster index mapping (faster than repeated lookups)
#     unique_labels, cluster_indices = np.unique(labels.values, return_inverse=True)
#     n_clusters = len(unique_labels)
#     n_features = X.shape[1]
    
#     # Build all contingency tables at once using vectorized operations
#     # Shape: (n_clusters, 2, n_features)
#     # This is the key optimization - one pass instead of n_features passes
#     contingency = np.zeros((n_clusters, 2, n_features), dtype=int)
    
#     # Use advanced indexing to populate all contingency tables simultaneously
#     for i in range(n_samples):
#         cluster_idx = cluster_indices[i]
#         contingency[cluster_idx, X_array[i], np.arange(n_features)] += 1
    
#     # Filter out constant features (all 0 or all 1)
#     # A feature is valid if it has both 0s and 1s across all clusters
#     has_zeros = (contingency[:, 0, :].sum(axis=0) > 0)
#     has_ones = (contingency[:, 1, :].sum(axis=0) > 0)
#     valid_features = has_zeros & has_ones
    
#     if not valid_features.any():
#         return 0.0
    
#     # Keep only valid features
#     contingency = contingency[:, :, valid_features]
    
#     # Vectorized chi-square computation for all features at once
#     # Row totals: sum across feature values (axis=1)
#     row_totals = contingency.sum(axis=1, keepdims=True)  # (n_clusters, 1, n_valid_features)
    
#     # Column totals: sum across clusters (axis=0)
#     col_totals = contingency.sum(axis=0, keepdims=True)  # (1, 2, n_valid_features)
    
#     # Expected frequencies using broadcasting
#     expected = (row_totals * col_totals) / n_samples
#     expected = np.maximum(expected, 1e-10)  # Avoid division by zero
    
#     # Chi-square statistic for each feature
#     chi2 = ((contingency - expected) ** 2 / expected).sum(axis=(0, 1))
    
#     # Cramér's V for binary features: V = sqrt(chi2 / n)
#     # (min_dim = min(n_clusters-1, 2-1) = 1 for binary features)
#     cramers_v_values = np.sqrt(chi2 / n_samples)
    
#     # Return mean across all features
#     return cramers_v_values.mean()


# def cramers_v_score(X: pd.DataFrame, labels: pd.Series) -> float:
#     """
#     Compute Cramér's V for binary features.
    
#     Cramér's V measures the association between cluster membership and feature values.
    
#     Higher V = clusters are more separated.
    
#     Parameters
#     ----------
#     X : pd.DataFrame
#         Feature matrix (samples x features, binary 0/1 values)
#     labels : pd.Series
#         Cluster assignments (index = sample IDs)
        
#     Returns
#     -------
#     float
#         Mean Cramér's V (0-1, higher = more separated clusters)
        
#     Notes
#     -----
#     Interpretation:
#     - V = 0.0: Feature is completely independent of cluster (clusters not separated)
#     - V = 0.3: Knowing cluster improves feature prediction by ~30% (moderate)
#     - V = 1.0: Feature perfectly predicts cluster (perfect separation)
    
#     Examples
#     --------
#     >>> v = cramers_v_score(X_binary, labels)
#     >>> print(f"Cramér's V: {v:.3f}")  # Higher = better
#     >>> # Target: V > 0.25 for separated clusters
#     """
#     return _cramers_v_binary(X, labels)
def _cramers_v_binary(X: pd.DataFrame, labels: pd.Series) -> float:
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
    
    # Convert to numpy for speed
    X_array = X.values.astype(int)
    
    # Create cluster index mapping
    unique_labels, cluster_indices = np.unique(labels.values, return_inverse=True)
    n_clusters = len(unique_labels)
    n_features = X.shape[1]
    
    # Build all contingency tables at once: (n_clusters, 2, n_features)
    contingency = np.zeros((n_clusters, 2, n_features), dtype=int)
    
    for i in range(n_samples):
        cluster_idx = cluster_indices[i]
        contingency[cluster_idx, X_array[i], np.arange(n_features)] += 1
    
    # Filter constant features (must have both 0s and 1s)
    has_zeros = (contingency[:, 0, :].sum(axis=0) > 0)
    has_ones = (contingency[:, 1, :].sum(axis=0) > 0)
    valid_features = has_zeros & has_ones
    
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


def cramers_v_score(X: pd.DataFrame, labels: pd.Series) -> float:
    """
    Compute Cramér's V for binary features.
    
    Cramér's V measures the association between cluster membership and feature values.
    
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
    - V = 0.0: Features are independent of cluster membership
    - V = 0.3: Knowing cluster improves feature prediction by ~30% (moderate)
    - V = 1.0: Features perfectly predict cluster membership (perfect separation)
    
    Examples
    --------
    >>> v = cramers_v_score(X_binary, labels)
    >>> print(f"Cramér's V: {v:.3f}")  # Higher = better
    >>> # Target: V > 0.25 for separated clusters
    """
    return _cramers_v_binary(X, labels)


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def distinctiveness_score(X: pd.DataFrame, labels: pd.Series, data_type: str = "auto") -> float:
    """
    Measure cluster separation for continuous or binary features.
    
    Uses eta-squared (η²) for continuous features or Cramér's V for binary features.
    
    Higher value = clusters are more separated.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features)
    labels : pd.Series
        Cluster assignments (index = sample IDs)
    data_type : str, default="auto"
        Type of features: "continuous", "binary", or "auto" to detect automatically
        
    Returns
    -------
    float
        Eta-squared (continuous) or Cramér's V (binary), both in range 0-1
        
    Notes
    -----
    Auto-detection considers data binary if:
    - All values are in {0, 1}
    - No more than 2 unique values per column
    
    Interpretation:
    - 0.0: Clusters not separated
    - 0.3: Moderate separation (30% of variance/association explained)
    - 0.6+: Strong separation
    
    Examples
    --------
    >>> # Continuous features (auto-detected)
    >>> eta_sq = distinctiveness_score(X_continuous, labels)
    >>> print(f"Eta-squared: {eta_sq:.3f}")
    
    >>> # Binary features (auto-detected)
    >>> v = distinctiveness_score(X_binary, labels)
    >>> print(f"Cramér's V: {v:.3f}")
    
    >>> # Explicit specification
    >>> eta_sq = distinctiveness_score(X, labels, data_type="continuous")
    >>> v = distinctiveness_score(X, labels, data_type="binary")
    """
    # Auto-detect data type
    if data_type == "auto":
        # Check if data is binary (all values in {0, 1})
        unique_values = set(X.values.flatten())
        unique_values.discard(np.nan)  # Ignore NaN
        
        if unique_values.issubset({0, 1, 0.0, 1.0}):
            data_type = "binary"
        else:
            data_type = "continuous"
    
    # Validate data_type
    if data_type not in ["continuous", "binary"]:
        raise ValueError(f"data_type must be 'continuous', 'binary', or 'auto', got '{data_type}'")
    
    # Call appropriate function
    if data_type == "continuous":
        return _eta_squared_continuous(X, labels)
    else:  # binary
        return _cramers_v_binary(X, labels)

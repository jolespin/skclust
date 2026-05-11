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

def eta_squared_score(X: pd.DataFrame, labels: pd.Series) -> float:
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



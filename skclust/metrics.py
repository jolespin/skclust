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
from scipy.stats.contingency import association


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
    
    # Total sum of squares (variance around grand mean)
    grand_mean = X.values.flatten().mean()
    total_sum_of_squares = ((X - grand_mean) ** 2).values.sum()
    
    # Within-cluster sum of squares (variance around each cluster's mean)
    def within_sum_of_squares(group):
        group_mean = group.values.flatten().mean()
        return ((group - group_mean) ** 2).values.sum()
    
    within_cluster_sum_of_squares = X.groupby(labels).apply(within_sum_of_squares).sum()
    
    # Between-cluster sum of squares
    between_cluster_sum_of_squares = total_sum_of_squares - within_cluster_sum_of_squares
    
    # Eta-squared (proportion of variance explained by cluster membership)
    eta_squared = between_cluster_sum_of_squares / total_sum_of_squares if total_sum_of_squares > 0 else 0
    
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
    
    # Group by cluster and compute entropy for each feature
    def compute_entropy(group):
        """Compute mean entropy across all features using scipy.stats.entropy."""
        entropies = []
        for col in group.columns:
            value_counts = group[col].value_counts()
            feature_entropy = entropy(value_counts, base=2)
            entropies.append(feature_entropy)
        return np.mean(entropies)
    
    entropy_per_cluster = X.groupby(labels).apply(compute_entropy)
    
    return entropy_per_cluster


def _cramers_v_binary(X: pd.DataFrame, labels: pd.Series) -> float:
    """
    Compute Cramér's V for binary features.
    
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
    
    cramers_v_values = []
    
    # Compute Cramér's V for each feature using scipy.stats.contingency.association
    for feature in X.columns:
        # Create contingency table: clusters × feature (0/1)
        contingency = pd.crosstab(labels, X[feature])
        
        # Skip if feature is constant (all 0 or all 1)
        if contingency.shape[1] < 2:
            continue
        
        # Compute Cramér's V using scipy
        v = association(contingency.values, method='cramer')
        cramers_v_values.append(v)
    
    # Return mean across all features
    return np.mean(cramers_v_values) if cramers_v_values else 0.0


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
    - V = 0.0: Feature is completely independent of cluster (clusters not separated)
    - V = 0.3: Knowing cluster improves feature prediction by ~30% (moderate)
    - V = 1.0: Feature perfectly predicts cluster (perfect separation)
    
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

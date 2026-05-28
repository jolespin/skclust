# -*- coding: utf-8 -*-
# skclust/distance.py

import numpy as np
import pandas as pd

def cosine_distances_to_representatives(
    X_l2,
    labels,
    X_representatives_l2,
    representative_labels,
    representative_types,
    check=True,
    index_name=None,
):
    """
    Compute cosine distances from each observation to its group's representative embeddings.
    
    Parameters
    ----------
    X_l2 : np.ndarray or pd.DataFrame of shape (n_observations, d)
        L2-normalized embeddings for observations.
    labels : np.ndarray or pd.Series of shape (n_observations,)
        Group assignment for each observation.
    X_representatives_l2 : np.ndarray or pd.DataFrame of shape (n_representatives, d)
        L2-normalized embeddings for group representatives.
    representative_labels : np.ndarray or pd.Series of shape (n_representatives,)
        Group ID for each representative row.
    representative_types : np.ndarray or pd.Series of shape (n_representatives,)
        Type label for each representative (e.g., "core", "union").
    check : bool, default True
        Verify L2 normalization.
    index_name : str, default None
        Name of output index. If None, inferred from labels if pandas.
        
    Returns
    -------
    pd.DataFrame
        Columns: one per representative type, values are cosine distances.
        Index: observation IDs from labels.
    """
    # --- Type concordance: observations ---
    _X_is_df = isinstance(X_l2, pd.DataFrame)
    _labels_is_series = isinstance(labels, pd.Series)
    if _X_is_df and not _labels_is_series:
        raise TypeError("X_l2 is a DataFrame but labels is not a Series")
    if _labels_is_series and not _X_is_df:
        raise TypeError("labels is a Series but X_l2 is not a DataFrame")
    
    # --- Type concordance: representatives ---
    _Xrep_is_df = isinstance(X_representatives_l2, pd.DataFrame)
    _replabels_is_series = isinstance(representative_labels, pd.Series)
    _reptypes_is_series = isinstance(representative_types, pd.Series)
    _rep_pandas = [_Xrep_is_df, _replabels_is_series, _reptypes_is_series]
    if any(_rep_pandas) and not all(_rep_pandas):
        raise TypeError(
            "X_representatives_l2, representative_labels, and representative_types "
            "must all be pandas objects or all be array-like, not a mix"
        )

    # --- Index alignment: observations ---
    obs_index = None
    if _X_is_df and _labels_is_series:
        if not X_l2.index.equals(labels.index):
            raise ValueError(
                f"X_l2.index and labels.index do not match. "
                f"X_l2 has {len(X_l2.index)} entries, labels has {len(labels.index)} entries, "
                f"with {len(X_l2.index.difference(labels.index))} in X_l2 only and "
                f"{len(labels.index.difference(X_l2.index))} in labels only"
            )
        obs_index = labels.index
        if index_name is None:
            index_name = labels.index.name

    # --- Index alignment: representatives ---
    if _Xrep_is_df and _replabels_is_series:
        if not X_representatives_l2.index.equals(representative_labels.index):
            raise ValueError(
                "X_representatives_l2.index and representative_labels.index do not match"
            )
        if not X_representatives_l2.index.equals(representative_types.index):
            raise ValueError(
                "X_representatives_l2.index and representative_types.index do not match"
            )

    # --- Coerce to numpy ---
    if _X_is_df:
        X_l2 = X_l2.values
    if _labels_is_series:
        labels_values = labels.values
    else:
        labels_values = np.asarray(labels)
    if _Xrep_is_df:
        X_representatives_l2 = X_representatives_l2.values
    if _replabels_is_series:
        representative_labels = representative_labels.values
    else:
        representative_labels = np.asarray(representative_labels)
    if _reptypes_is_series:
        representative_types = representative_types.values
    else:
        representative_types = np.asarray(representative_types)

    X_l2 = np.asarray(X_l2, dtype=np.float32)
    X_representatives_l2 = np.asarray(X_representatives_l2, dtype=np.float32)

    # --- Shape checks ---
    assert X_l2.shape[0] == len(labels_values), \
        f"X_l2 rows ({X_l2.shape[0]}) != labels length ({len(labels_values)})"
    assert X_l2.shape[1] == X_representatives_l2.shape[1], \
        f"Embedding dimensions differ: {X_l2.shape[1]} vs {X_representatives_l2.shape[1]}"
    assert X_representatives_l2.shape[0] == len(representative_labels) == len(representative_types), \
        "X_representatives_l2, representative_labels, and representative_types must have equal length"
    
    # --- L2 normalization check ---
    if check:
        for name, arr in [("X_l2", X_l2), ("X_representatives_l2", X_representatives_l2)]:
            norms = np.linalg.norm(arr, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-5):
                raise ValueError(
                    f"{name} is not L2-normalized. "
                    f"Norm range: [{norms.min():.6f}, {norms.max():.6f}]"
                )

    # --- Compute distances per representative type ---
    unique_types = np.unique(representative_types)
    results = {}

    for rep_type in unique_types:
        mask = representative_types == rep_type
        X_rep = X_representatives_l2[mask]
        rep_labels = representative_labels[mask]

        group_to_idx = pd.Series(np.arange(len(rep_labels)), index=rep_labels)
        
        missing = set(labels_values) - set(rep_labels)
        if missing:
            raise ValueError(
                f"Representative type '{rep_type}' is missing for {len(missing)} group(s): "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
        
        aligned_idx = group_to_idx.loc[labels_values].values
        X_rep_aligned = X_rep[aligned_idx]
        
        distances = 1.0 - np.sum(X_l2 * X_rep_aligned, axis=1)
        results[rep_type] = distances

    df_distances = pd.DataFrame(results, index=obs_index if obs_index is not None else np.arange(len(labels_values)))
    df_distances.index.name = index_name
    
    return df_distances
# -*- coding: utf-8 -*-
# skclust/distance.py

import numpy as np
import pandas as pd
from itertools import combinations
from typing import (
    Optional,
    Union,
)
from loguru import logger
from scipy.spatial.distance import squareform
from scipy.stats import (
    mannwhitneyu,
    median_abs_deviation,
)
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
from tqdm import tqdm

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
        y_values = labels.values
    else:
        y_values = np.asarray(labels)
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
    assert X_l2.shape[0] == len(y_values), \
        f"X_l2 rows ({X_l2.shape[0]}) != labels length ({len(y_values)})"
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
        
        missing = set(y_values) - set(rep_labels)
        if missing:
            raise ValueError(
                f"Representative type '{rep_type}' is missing for {len(missing)} group(s): "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
        
        aligned_idx = group_to_idx.loc[y_values].values
        X_rep_aligned = X_rep[aligned_idx]
        
        distances = 1.0 - np.sum(X_l2 * X_rep_aligned, axis=1)
        results[rep_type] = distances

    df_distances = pd.DataFrame(results, index=obs_index if obs_index is not None else np.arange(len(y_values)))
    df_distances.index.name = index_name
    
    return df_distances

def pairwise_cosine_distances(X, check=True, redundant_form=True):
    """
    Cosine distances via matrix multiplication (assumes L2-normalized rows).

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape (n, d)
        L2-normalized embeddings.
    check : bool, default True
        Verify each row has unit L2 norm.
    redundant_form : bool, default True
        True -> (n, n) square matrix; False -> condensed upper triangle.

    Returns
    -------
    np.ndarray or pd.DataFrame/pd.Series if X is a pd.DataFrame
        - pd.DataFrame input, redundant_form=True  -> pd.DataFrame(index, columns=index)
        - pd.DataFrame input, redundant_form=False -> pd.Series(index=frozenset pairs)
        - np.ndarray input                         -> np.ndarray
    """
    index = None
    if isinstance(X, pd.DataFrame):
        index = X.index
        X = X.values
    X = np.asarray(X, dtype=np.float32)

    if check:
        norms = np.linalg.norm(X, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(
                f"X is not L2-normalized (norm range "
                f"[{norms.min():.6f}, {norms.max():.6f}]). Pass check=False to skip."
            )

    distances = np.clip(1.0 - X @ X.T, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)

    if redundant_form:
        if index is not None:
            return pd.DataFrame(distances, index=index, columns=index)
        return distances
    else:
        condensed = squareform(distances, checks=False)
        if index is not None:
            pair_index = pd.Index(list(map(frozenset, combinations(index, 2))), name=index.name)
            if pair_index.name is None:
                pair_index.name = "cosine_distance"
            return pd.Series(condensed, index=pair_index)
        return condensed


def pairwise_jaccard_distances(X, check=True, redundant_form=True):
    """
    Jaccard distances via matrix multiplication (assumes binary rows).

    Jaccard distance = 1 - |A intersect B| / |A union B|, with the intersection
    obtained as X @ X.T and the union from row sums. Two all-zero rows are
    treated as identical (distance 0).

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape (n, d)
        Binary (0/1) presence-absence matrix.
    check : bool, default True
        Verify all values are 0 or 1.
    redundant_form : bool, default True
        True -> (n, n) square matrix; False -> condensed upper triangle.

    Returns
    -------
    np.ndarray or pd.DataFrame/pd.Series if X is a pd.DataFrame
        - pd.DataFrame input, redundant_form=True  -> pd.DataFrame(index, columns=index)
        - pd.DataFrame input, redundant_form=False -> pd.Series(index=frozenset pairs)
        - np.ndarray input                         -> np.ndarray
    """
    index = None
    if isinstance(X, pd.DataFrame):
        index = X.index
        X = X.values
    X = np.asarray(X)

    if check and not np.isin(np.unique(X), (0, 1)).all():
        raise ValueError("X is not binary (values must be 0 or 1). Pass check=False to skip.")

    X = X.astype(np.float32)
    intersection = X @ X.T
    row_sums = X.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    distances = np.where(union > 0, 1.0 - intersection / union, 0.0)
    np.fill_diagonal(distances, 0.0)

    if redundant_form:
        if index is not None:
            return pd.DataFrame(distances, index=index, columns=index)
        return distances
    else:
        condensed = squareform(distances, checks=False)
        if index is not None:
            pair_index = pd.Index(list(map(frozenset, combinations(index, 2))), name=index.name)
            if pair_index.name is None:
                pair_index.name = "jaccard_distance"
            return pd.Series(condensed, index=pair_index)
        return condensed

class ClusteredDistances:
    """
    Compute intra- and inter-cluster distance summaries with effect sizes
    and optional PERMANOVA testing.

    For each cluster, intra-cluster distances are the n*(n-1)/2 pairwise
    distances among cluster members, and inter-cluster distances are the
    n_cluster * n_other pairwise distances between members and non-members.
    A rank-biserial correlation quantifies separation; PERMANOVA provides
    a valid significance test.

    Parameters
    ----------
    metric : str, default "cosine"
        Distance metric: "cosine" (requires L2-normalized rows),
        "jaccard" (requires binary 0/1 rows), or "precomputed"
        (X is a square distance matrix, pd.DataFrame, or skbio
        DistanceMatrix). Passing a square pd.DataFrame with matching
        index/columns or a skbio DistanceMatrix requires
        ``metric="precomputed"`` explicitly.
    check : bool, default True
        Validate input (L2-norm for cosine, binary for jaccard,
        symmetry for precomputed).
    chunk_size : int or None, default None
        Row chunk size for inter-cluster matrix multiplications to
        limit peak memory. Only applies to cosine/jaccard metrics.
    verbose : int, default 0
        If > 0, log warnings for singleton or empty clusters.
    n_permutations : int or None, default None
        Number of PERMANOVA permutations. None skips PERMANOVA
        entirely (and avoids computing the full N×N distance matrix
        for cosine/jaccard). Setting this on large datasets will
        force an O(N²) distance computation.
    random_state : int or None, default None
        Random state of PERMANOVA


    Attributes
    ----------
    results_ : pd.DataFrame
        MultiIndex columns with cluster size and intra/inter-cluster
        summary statistics (n_pairs, mean, median, std, mad).
        Index named ``id_cluster``.
    effect_sizes_ : pd.Series
        Rank-biserial correlation per cluster (index: id_cluster).
    u_statistics_ : pd.Series
        Mann-Whitney U statistic per cluster (index: id_cluster).
    p_values_naive_ : pd.Series
        Mann-Whitney U p-values per cluster. **These are invalid —
        see Notes.** Stored for transparency; do not cite.
    effect_size_method_ : str
        Fixed: ``"rank_biserial"``.
    permanova_ : pd.Series or None
        Output of ``skbio.stats.distance.permanova``, or None if
        ``n_permutations`` was not set.
    p_value_ : float
        PERMANOVA p-value (shortcut for ``permanova_["p-value"]``).
        Only set when ``n_permutations`` is not None.
    r_squared_ : float
        Proportion of variance explained by the grouping, derived
        from the PERMANOVA pseudo-F statistic:
        R² = 1 / (1 + (n − g) / ((g − 1) × F)).
        Only set when ``n_permutations`` is not None.
    labels_ : np.ndarray or pd.Series
        Cluster labels as passed to ``fit``.
    n_samples_ : int
        Number of samples.
    n_features_ : int or None
        Number of features (None for precomputed).

    Notes
    -----
    **Interpreting effect sizes vs. p-values**

    The rank-biserial correlation (``.effect_sizes_``) is a valid
    descriptive statistic. It is the probability that a randomly chosen
    within-cluster pair is closer than a randomly chosen between-cluster
    pair, computed deterministically from the observed distances. It
    assumes nothing about independence.

    The Mann-Whitney U p-value (``.p_values_naive_``), however, is
    **not valid** and should not be used for inference. The test is run
    on pairwise distances, which are not independent observations: each
    sample appears in (n − 1) pairs, so the effective sample size is the
    number of points, not the number of pairs. Mann-Whitney's variance
    formula assumes independent observations, treats the inflated pair
    count as real information, understates the standard error, and
    produces anti-conservative p-values that typically underflow to ≈ 0
    regardless of effect magnitude. Pair count, not effect strength,
    drives the result.

    These are two distinct failure modes with a single root cause.
    Non-independence breaks the p-value; it does not affect the effect
    size. That asymmetry is the key point.

    **PERMANOVA**

    For a valid significance test of group structure in the embedding
    space, use the PERMANOVA result (``.permanova_``). PERMANOVA
    permutes group labels across points (the correct observational
    unit) and is unaffected by the non-independence of pairwise
    distances. Note that PERMANOVA tests centroid separation, which
    is a related but distinct contrast from the rank-biserial, and
    is sensitive to dispersion differences in unbalanced designs.
    It should not be interpreted as a drop-in p-value for the
    rank-biserial correlation.

    When PERMANOVA is run, R² (proportion of total variance explained
    by the grouping) is derived from the pseudo-F statistic and stored
    in ``.r_squared_`` and in the ``permanova_`` Series under key
    ``"R-squared"``.

    **Confounding**

    Neither the effect size nor PERMANOVA separates ecological or
    functional distinctness from phylogenetic clustering, since group
    labels often track clades. Both statistics quantify embedding-space
    separation, not its cause.
    """

    _VALID_METRICS = {"cosine", "jaccard", "precomputed"}

    def __init__(
        self,
        metric: str = "cosine",
        check: bool = True,
        chunk_size: Optional[int] = None,
        verbose: int = 0,
        n_permutations: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        if metric not in self._VALID_METRICS:
            raise ValueError(
                f"metric must be one of {sorted(self._VALID_METRICS)}, "
                f"got '{metric}'"
            )
        self.metric = metric
        self.check = check
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.n_permutations = n_permutations
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _summarize(distances: np.ndarray) -> dict:
        return {
            "n_pairs": distances.size,
            "mean": np.mean(distances),
            "median": np.median(distances),
            "std": np.std(distances, ddof=1),
            "mad": median_abs_deviation(distances),
        }

    def _compute_intra(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        metric: str,
        id_cluster,
    ) -> Optional[np.ndarray]:
        """Return condensed intra-cluster distances, or None for singletons."""
        n = mask.sum()
        if n < 2:
            if self.verbose > 0:
                logger.warning(
                    f"Cluster '{id_cluster}' has {n} member(s); "
                    f"intra-cluster stats will be NaN"
                )
            return None

        if metric == "precomputed":
            D = X[np.ix_(mask, mask)].copy()
            np.fill_diagonal(D, 0.0)
            return squareform(D, checks=False)

        A = X[mask]

        if metric == "cosine":
            D = np.clip(1.0 - A @ A.T, 0.0, 2.0)
            np.fill_diagonal(D, 0.0)
            return squareform(D, checks=False)

        # jaccard
        row_sums = A.sum(axis=1)
        intersection = A @ A.T
        union = row_sums[:, None] + row_sums[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.where(union > 0, 1.0 - intersection / union, 0.0)
        np.fill_diagonal(D, 0.0)
        return squareform(D, checks=False)

    def _compute_inter(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        metric: str,
        id_cluster,
    ) -> Optional[np.ndarray]:
        """Return flattened inter-cluster distances, or None if no out-group."""
        if metric == "precomputed":
            sub = X[np.ix_(mask, ~mask)]
            if sub.shape[1] == 0:
                if self.verbose > 0:
                    logger.warning(
                        f"Cluster '{id_cluster}' has no out-group points; "
                        f"inter-cluster stats will be NaN"
                    )
                return None
            return sub.ravel()

        A = X[mask]
        B = X[~mask]

        if B.shape[0] == 0:
            if self.verbose > 0:
                logger.warning(
                    f"Cluster '{id_cluster}' has no out-group points; "
                    f"inter-cluster stats will be NaN"
                )
            return None

        if metric == "cosine":
            if self.chunk_size is not None:
                chunks = [
                    np.clip(1.0 - A[i : i + self.chunk_size] @ B.T, 0.0, 2.0).ravel()
                    for i in range(0, A.shape[0], self.chunk_size)
                ]
                return np.concatenate(chunks)
            return np.clip(1.0 - A @ B.T, 0.0, 2.0).ravel()

        # jaccard
        row_sums_A = A.sum(axis=1)
        row_sums_B = B.sum(axis=1)

        if self.chunk_size is not None:
            chunks = []
            for i in range(0, A.shape[0], self.chunk_size):
                A_chunk = A[i : i + self.chunk_size]
                intersection = A_chunk @ B.T
                union = (
                    row_sums_A[i : i + self.chunk_size, None]
                    + row_sums_B[None, :]
                    - intersection
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    d = np.where(union > 0, 1.0 - intersection / union, 0.0)
                chunks.append(d.ravel())
            return np.concatenate(chunks)

        intersection = A @ B.T
        union = row_sums_A[:, None] + row_sums_B[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, 1.0 - intersection / union, 0.0).ravel()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_cosine(X: np.ndarray):
        norms = np.linalg.norm(X, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise ValueError(
                f"X is not L2-normalized (norm range "
                f"[{norms.min():.6f}, {norms.max():.6f}]). "
                f"Pass check=False to skip."
            )

    @staticmethod
    def _validate_jaccard(X: np.ndarray):
        if X.dtype.kind == "b":
            return
        xmin, xmax = float(X.min()), float(X.max())
        if xmin < 0 or xmax > 1:
            raise ValueError(
                f"X must be binary (0/1 or bool). "
                f"Value range: [{xmin}, {xmax}]"
            )
        if not np.array_equal(X, X.astype(bool)):
            raise ValueError(
                "X must be binary (0/1 or bool). Found non-integer values."
            )

    @staticmethod
    def _validate_precomputed(X: np.ndarray):
        if X.shape[0] != X.shape[1]:
            raise ValueError(
                f"metric='precomputed' but X is not square: "
                f"shape {X.shape}"
            )
        if not np.allclose(X, X.T, atol=1e-5):
            raise ValueError(
                "metric='precomputed' but X is not symmetric."
            )

    # ------------------------------------------------------------------
    # fit / fit_transform
    # ------------------------------------------------------------------
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, DistanceMatrix],
        y: Union[np.ndarray, pd.Series],
    ) -> "ClusteredDistances":
        """
        Compute cluster distance summaries and effect sizes.

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or skbio.DistanceMatrix
            Data matrix, precomputed distance matrix, or skbio
            DistanceMatrix. A DistanceMatrix is always treated as
            precomputed regardless of the ``metric`` parameter.
        y : np.ndarray or pd.Series, shape (n,)
            Cluster labels aligned with rows of X. When X is a
            DistanceMatrix with non-default IDs and y is a Series,
            ``y.index`` must match the DistanceMatrix IDs.

        Returns
        -------
        self
        """
        # -- Handle skbio DistanceMatrix --
        _X_is_dm = isinstance(X, DistanceMatrix)
        if _X_is_dm:
            dm_ids = list(X.ids)
            dm_default_ids = [str(i) for i in range(X.shape[0])]
            _dm_is_labeled = dm_ids != dm_default_ids

            if isinstance(y, pd.Series) and _dm_is_labeled:
                dm_index = pd.Index(dm_ids)
                if not dm_index.equals(y.index):
                    if set(dm_ids) == set(y.index):
                        raise ValueError(
                            "DistanceMatrix IDs and y.index contain the "
                            "same values but in different order. Reindex "
                            "y to match the DistanceMatrix ID order."
                        )
                    raise ValueError(
                        f"DistanceMatrix IDs and y.index do not match. "
                        f"DistanceMatrix has {len(dm_ids)} IDs, y has "
                        f"{len(y.index)} entries, with "
                        f"{len(dm_index.difference(y.index))} in "
                        f"DistanceMatrix only and "
                        f"{len(y.index.difference(dm_index))} in y only."
                    )

            self._dm_input_ = X
            X = pd.DataFrame(
                X.data,
                index=dm_ids,
                columns=dm_ids,
            ) if _dm_is_labeled else X.data

            if _dm_is_labeled and not isinstance(y, pd.Series):
                y = pd.Series(np.asarray(y), index=dm_ids)

        # -- Type concordance --
        _X_is_df = isinstance(X, pd.DataFrame)
        _y_is_series = isinstance(y, pd.Series)
        if _X_is_df and not _y_is_series:
            raise TypeError("X is a DataFrame but y is not a Series")
        if _y_is_series and not _X_is_df:
            raise TypeError("y is a Series but X is not a DataFrame")

        # -- Resolve metric (auto-detect precomputed) --
        metric = self.metric
        if _X_is_dm:
            if metric != "precomputed":
                raise ValueError(
                    f"X is a skbio DistanceMatrix but metric='{metric}'. "
                    f"Set metric='precomputed' when passing a DistanceMatrix."
                )
        elif (
            _X_is_df
            and X.shape[0] == X.shape[1]
            and X.index.equals(X.columns)
        ):
            if metric != "precomputed":
                raise ValueError(
                    f"X is a square DataFrame with matching index/columns "
                    f"but metric='{metric}'. Set metric='precomputed' when "
                    f"passing a precomputed distance matrix."
                )

        # -- Index alignment --
        index = None
        if _X_is_df and _y_is_series:
            if not X.index.equals(y.index):
                raise ValueError(
                    f"X.index and y.index do not match. "
                    f"X has {len(X.index)} entries, y has "
                    f"{len(y.index)} entries, with "
                    f"{len(X.index.difference(y.index))} in X only and "
                    f"{len(y.index.difference(X.index))} in y only."
                )
            index = X.index

        # -- Store metadata --
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1] if metric != "precomputed" else None
        self.labels_ = y

        # -- Coerce to numpy --
        X_values = X.values if _X_is_df else np.asarray(X)
        y_values = y.values if _y_is_series else np.asarray(y)

        assert X_values.shape[0] == y_values.shape[0], (
            f"X rows ({X_values.shape[0]}) != "
            f"y length ({y_values.shape[0]})"
        )

        # -- Validate --
        if self.check:
            if metric == "cosine":
                self._validate_cosine(X_values)
            elif metric == "jaccard":
                self._validate_jaccard(X_values)
            elif metric == "precomputed":
                self._validate_precomputed(X_values)

        # -- Cast for matmul --
        X_values = X_values.astype(np.float32)

        # -- Per-cluster loop --
        _summary_metrics = ["n_pairs", "mean", "median", "std", "mad"]
        unique_labels = np.unique(y_values)

        results = {}
        effect_sizes = {}
        u_statistics = {}
        p_values_naive = {}

        n_clusters = len(unique_labels)

        cluster_dists = {}

        for id_cluster in tqdm(unique_labels, desc=f"Pairwise distances ({n_clusters} clusters)", unit="cluster"):
            mask = y_values == id_cluster
            n_cluster = int(mask.sum())

            row = {("size", "n"): n_cluster}

            intra_dists = self._compute_intra(X_values, mask, metric, id_cluster)
            if intra_dists is None:
                for m in _summary_metrics:
                    row[("intra-cluster", m)] = np.nan
            else:
                for m, v in self._summarize(intra_dists).items():
                    row[("intra-cluster", m)] = v

            inter_dists = self._compute_inter(X_values, mask, metric, id_cluster)
            if inter_dists is None:
                for m in _summary_metrics:
                    row[("inter-cluster", m)] = np.nan
            else:
                for m, v in self._summarize(inter_dists).items():
                    row[("inter-cluster", m)] = v

            cluster_dists[id_cluster] = (intra_dists, inter_dists)
            results[id_cluster] = row

        for id_cluster in tqdm(unique_labels, desc=f"Rank-biserial correlation effect sizes ({n_clusters} clusters)", unit="cluster"):
            intra_dists, inter_dists = cluster_dists[id_cluster]

            if intra_dists is not None and inter_dists is not None:
                stat, pval = mannwhitneyu(
                    intra_dists, inter_dists, alternative="two-sided",
                )
                es = 1.0 - (2.0 * stat) / (intra_dists.size * inter_dists.size)
                effect_sizes[id_cluster] = es
                u_statistics[id_cluster] = stat
                p_values_naive[id_cluster] = pval
            else:
                effect_sizes[id_cluster] = np.nan
                u_statistics[id_cluster] = np.nan
                p_values_naive[id_cluster] = np.nan

        del cluster_dists

        # -- Assemble results DataFrame --
        df_results = pd.DataFrame.from_dict(results, orient="index")
        df_results.columns = pd.MultiIndex.from_tuples(df_results.columns)
        df_results.index.name = "id_cluster"

        col_order = (
            [("size", "n")]
            + [("intra-cluster", m) for m in _summary_metrics]
            + [("inter-cluster", m) for m in _summary_metrics]
        )
        self.results_ = df_results[col_order]

        # -- Store test attributes --
        self.effect_sizes_ = pd.Series(
            effect_sizes, name="effect_size",
        )
        self.effect_sizes_.index.name = "id_cluster"

        self.u_statistics_ = pd.Series(
            u_statistics, name="u_statistic",
        )
        self.u_statistics_.index.name = "id_cluster"

        self.p_values_naive_ = pd.Series(
            p_values_naive, name="p_value_naive",
        )
        self.p_values_naive_.index.name = "id_cluster"

        self.effect_size_method_ = "rank_biserial"

        # -- PERMANOVA --
        if self.n_permutations is not None:
            logger.info(
                f"Running PERMANOVA with {self.n_permutations} permutations "
                f"({self.n_samples_} samples, {n_clusters} groups)"
            )
            ids = list(index) if index is not None else list(range(self.n_samples_))

            if _X_is_dm:
                dm = self._dm_input_
                grouping_ids = list(dm.ids)
            elif metric == "precomputed":
                dm = DistanceMatrix(X_values, ids=ids)
                grouping_ids = ids
            elif metric == "cosine":
                full_dm = pairwise_cosine_distances(
                    X_values, check=False, redundant_form=True,
                )
                dm = DistanceMatrix(full_dm, ids=ids)
                del full_dm
                grouping_ids = ids
            elif metric == "jaccard":
                full_dm = pairwise_jaccard_distances(
                    X_values, check=False, redundant_form=True,
                )
                dm = DistanceMatrix(full_dm, ids=ids)
                del full_dm
                grouping_ids = ids

            grouping = pd.Series(y_values, index=grouping_ids, name="group")
            self.permanova_ = permanova(
                dm, grouping, permutations=self.n_permutations,seed=self.random_state,
            )
            self.p_value_ = self.permanova_["p-value"]

            F = self.permanova_["test statistic"]
            n = self.permanova_["sample size"]
            g = self.permanova_["number of groups"]
            if F == 0:
                self.r_squared_ = 0.0
            else:
                self.r_squared_ = 1.0 / (1.0 + (n - g) / ((g - 1) * F))

            self.permanova_["R-squared"] = self.r_squared_
            self.permanova_ = self.permanova_[
                ["method name", "test statistic name", "sample size",
                 "number of groups", "test statistic", "p-value",
                 "R-squared", "number of permutations"]
            ]

            del dm
        else:
            self.permanova_ = None

        if hasattr(self, "_dm_input_"):
            del self._dm_input_

        # -- Runtime caveat --
        logger.warning(
            "Effect sizes (rank-biserial correlation) are valid descriptive "
            "statistics. Mann-Whitney U p-values (.p_values_naive_) are "
            "anti-conservative due to non-independence of pairwise distances "
            "— pair count, not effect strength, drives the result. "
            "For valid significance testing of group structure, use "
            "PERMANOVA (.permanova_). See class docstring for details."
        )

        return self

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, DistanceMatrix],
        y: Union[np.ndarray, pd.Series],
    ) -> pd.DataFrame:
        """
        Fit and return the cluster summary table.

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or skbio.DistanceMatrix
            Data matrix or precomputed distance matrix.
        y : np.ndarray or pd.Series
            Cluster labels.

        Returns
        -------
        pd.DataFrame
            Same as ``self.results_``.
        """
        return self.fit(X, y).results_
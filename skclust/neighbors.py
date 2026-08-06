# -*- coding: utf-8 -*-
# skclust/neighbors.py

from __future__ import annotations
import warnings
from typing import Union
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.base import clone, BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.neighbors import KNeighborsTransformer
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from tqdm import tqdm
from loguru import logger
from .utils import adjacency_to_igraph

try:
    from deslib.util.faiss_knn_wrapper import FaissKNNClassifier as _FaissKNNClassifier
    DESLIB_AVAILABLE = True
except ImportError:
    DESLIB_AVAILABLE = False
    _FaissKNNClassifier = None


def _check_deslib():
    if not DESLIB_AVAILABLE:
        raise ImportError(
            "FaissKNN wrappers require 'deslib' and 'faiss'. "
            "Install with: pip install deslib faiss-cpu"
        )

def kneighbors_graph_from_transformer(
    X, 
    knn_transformer: Union[KNeighborsTransformer, type] = KNeighborsTransformer, 
    mode: str = "connectivity", 
    include_self: Union[bool, str] = True, 
    **transformer_kwargs
) -> sps.csr_matrix:
    """
    Calculate distance or connectivity graph using any KNN transformer.
    
    This function provides a generalized interface for creating k-nearest neighbors
    graphs from various KNN transformer implementations, with flexible handling of
    self-connections.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data matrix.
        
    knn_transformer : KNeighborsTransformer instance or class, default=KNeighborsTransformer
        Either:
        - A fitted KNN transformer instance (will be cloned if include_self=True)
        - An uninstantiated KNN transformer class (will be instantiated with **transformer_kwargs)
        
    mode : {'connectivity', 'distance'}, default='connectivity'
        Type of returned matrix:
        - 'connectivity': binary matrix with 1s for neighbors, 0s otherwise
        - 'distance': actual distances between neighbors
        
    include_self : bool or 'auto', default=True
        Whether to mark each sample as its own nearest neighbor.
        - If 'auto': True for mode='connectivity', False for mode='distance'
        - If True: adjusts n_neighbors internally (uses n_neighbors-1 in transformer)
        - If False: uses n_neighbors as-is
        
    **transformer_kwargs : dict
        Keyword arguments passed to knn_transformer constructor if not already instantiated.
        Must include 'n_neighbors' if transformer is a class.
        
    Returns
    -------
    knn_graph : scipy.sparse.csr_matrix of shape (n_samples, n_samples)
        Sparse matrix representing the k-nearest neighbors graph.
        
    Raises
    ------
    AssertionError
        If mode is not 'distance' or 'connectivity'.
        If transformer_kwargs provided with already-instantiated transformer.
    Exception
        If n_neighbors not provided when transformer is a class.
        
    Notes
    -----
    When include_self=True and n_neighbors=k, this is equivalent to
    include_self=False and n_neighbors=(k-1). The function handles this
    internally by adjusting n_neighbors.
    
    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsTransformer
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> 
    >>> # Using class with kwargs
    >>> graph = kneighbors_graph_from_transformer(
    ...     X, 
    ...     knn_transformer=KNeighborsTransformer,
    ...     n_neighbors=2,
    ...     mode='connectivity'
    ... )
    >>> 
    >>> # Using fitted instance
    >>> knn = KNeighborsTransformer(n_neighbors=2).fit(X)
    >>> graph = kneighbors_graph_from_transformer(X, knn_transformer=knn)
    """
    # Validate mode
    assert mode in {"distance", "connectivity"}, \
        f"mode must be either 'distance' or 'connectivity', got '{mode}'"

    # Handle auto include_self
    if include_self == "auto":
        include_self = mode == "connectivity"

    # Handle transformer instantiation
    if isinstance(knn_transformer, type):
        # knn_transformer is a class, need to instantiate
        if "n_neighbors" not in transformer_kwargs:
            raise Exception(
                "Please provide `n_neighbors` in transformer_kwargs when passing "
                "an uninstantiated transformer class"
            )
        
        n_neighbors = transformer_kwargs["n_neighbors"]
        if include_self:
            transformer_kwargs["n_neighbors"] = n_neighbors - 1
            
        knn_transformer = knn_transformer(**transformer_kwargs)
    else:
        # knn_transformer is already instantiated
        if transformer_kwargs:
            raise AssertionError(
                "Please provide uninstantiated `knn_transformer` class OR "
                "do not provide `transformer_kwargs`"
            )
        
        if include_self:
            warnings.warn(
                "`include_self=True and n_neighbors=k` is equivalent to "
                "`include_self=False and n_neighbors=(k-1)`. Backend is creating "
                "a clone with n_neighbors=(k-1)"
            )
            knn_transformer = clone(knn_transformer)
            params = knn_transformer.get_params()
            n_neighbors = params["n_neighbors"]
            knn_transformer.set_params(n_neighbors=n_neighbors - 1)
        
    # Compute KNN graph
    knn_graph = knn_transformer.fit_transform(X)
    
    # Convert to connectivity if requested
    if mode == "connectivity":
        knn_graph = (knn_graph > 0).astype(float)
        
        # Set diagonal to 1.0 for self-connections
        if include_self:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                knn_graph.setdiag(1.0)
                
    return knn_graph

def kneighbors_to_snn_graph(indices, k=None, skip_self=True):
    """
    Build a Jaccard-weighted Shared Nearest Neighbor (SNN) graph from KNN indices.
    
    For each pair of connected nodes (i, j), computes the Jaccard similarity
    of their K-nearest neighbor sets:
    
        J(i, j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
    
    where N(i) and N(j) are the K-nearest neighbor sets of nodes i and j.
    Since each node has exactly K neighbors, |N(i) ∪ N(j)| = 2K - |N(i) ∩ N(j)|.
    
    SNN reweighting amplifies edges between nodes that share many neighbors
    (within-cluster) and suppresses edges between nodes with little neighborhood
    overlap (between-cluster), producing a more informative graph for community
    detection than raw distance or similarity weights.
    
    Parameters
    ----------
    indices : np.ndarray, shape (n_samples, K_max)
        Neighbor indices, e.g. from KNeighborsCosineGraph.indices_
    k : int or None, default=None
        Number of neighbors to use. If None, uses all available neighbors.
    skip_self : bool, default=True
        If True, assumes column 0 contains self-indices and slicing starts at 1.
        
    Returns
    -------
    A_snn : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        Symmetric Jaccard-weighted adjacency matrix. Edges exist for all pairs
        where at least one node has the other in its K-nearest neighbors.
        
    Raises
    ------
    ValueError
        If k exceeds the number of available neighbor columns.
        
    Notes
    -----
    The output graph is symmetrized via union: an edge (i, j) exists if i is 
    in N(j) OR j is in N(i). This follows the convention used in PhenoGraph [2]_
    and Seurat.
    
    References
    ----------
    .. [1] Jarvis, R.A. & Patrick, E.A. (1973). Clustering Using a Similarity 
       Measure Based on Shared Near Neighbors. IEEE Transactions on Computers, 
       C-22(11), 1025-1034. doi:10.1109/T-C.1973.223640
       
    .. [2] Levine, J.H. et al. (2015). Data-Driven Phenotypic Dissection of AML 
       Reveals Progenitor-like Cells that Correlate with Prognosis. Cell, 162(1), 
       184-197. doi:10.1016/j.cell.2015.05.047
       
    Examples
    --------
    >>> from skclust.neighbors import KNeighborsCosineGraph, kneighbors_to_snn_graph
    >>> knn = KNeighborsCosineGraph(n_neighbors=100, backend="faiss")
    >>> knn.fit(X)
    >>> A_snn = kneighbors_to_snn_graph(knn.indices_, k=75)
    """
    offset = 1 if skip_self else 0
    k_max = indices.shape[1] - offset

    if k is None:
        k = k_max
    if k > k_max:
        raise ValueError(f"k={k} exceeds available neighbors ({k_max})")

    I = indices[:, offset:k + offset]
    n = indices.shape[0]

    rows = np.repeat(np.arange(n), k)
    A = sps.csr_matrix((np.ones(len(rows)), (rows, I.ravel())), shape=(n, n))

    intersection = A.dot(A.T)
    A_sym = A.maximum(A.T)
    r, c = A_sym.nonzero()
    inter_vals = np.array(intersection[r, c]).ravel()
    jaccard_vals = inter_vals / (k + k - inter_vals)

    return sps.csr_matrix((jaccard_vals, (r, c)), shape=(n, n))


def brute_force_kneighbors_graph_from_rectangular_distance(
    distance_matrix: np.ndarray, 
    n_neighbors: int, 
    mode: str = "connectivity", 
    include_self: bool = True
) -> sps.csr_matrix:
    """
    Build k-nearest neighbors graph from a pre-computed rectangular distance matrix.
    
    This function efficiently constructs a sparse kNN graph from a distance matrix
    without requiring square/symmetric input. Uses numpy's partitioning for O(n)
    complexity per row instead of full sorting.
    
    Parameters
    ----------
    distance_matrix : array-like of shape (n_samples, n_candidates)
        Pre-computed distance matrix. Can be rectangular (e.g., distances from
        n_samples to a different set of n_candidates points).
        
    n_neighbors : int
        Number of nearest neighbors to retain for each sample.
        
    mode : {'connectivity', 'distance'}, default='connectivity'
        Type of returned matrix:
        - 'connectivity': binary matrix with 1s for neighbors
        - 'distance': actual distance values for neighbors
        
    include_self : bool, default=True
        If True, adjusts n_neighbors to n_neighbors-1 to account for self-connections.
        
    Returns
    -------
    graph : scipy.sparse.csr_matrix of shape (n_samples, n_candidates)
        Sparse k-nearest neighbors graph.
        
    Notes
    -----
    Uses np.argpartition for O(n) complexity per row, which is faster than
    full sorting when n_neighbors << n_candidates.
    
    Examples
    --------
    >>> distances = np.array([[0.0, 1.0, 3.0], 
    ...                       [1.0, 0.0, 2.0]])
    >>> graph = brute_force_kneighbors_graph_from_rectangular_distance(
    ...     distances, n_neighbors=2, mode='distance'
    ... )
    """
    assert mode in {"distance", "connectivity"}, \
        f"mode must be either 'distance' or 'connectivity', got '{mode}'"

    if include_self:
        n_neighbors = n_neighbors - 1
        
    # Get indices of k nearest neighbors using partial sort
    indices = np.argpartition(distance_matrix, n_neighbors, axis=1)[:, :n_neighbors]
    
    # Prepare data values
    if mode == "connectivity":
        data = np.ones(distance_matrix.shape[0] * n_neighbors, dtype=float)
    else:  # mode == "distance"
        data = np.partition(distance_matrix, n_neighbors, axis=1)[:, :n_neighbors].ravel()
    
    # Build sparse matrix in COO format
    row = np.repeat(np.arange(distance_matrix.shape[0]), n_neighbors)
    col = indices.ravel()
    
    graph = sps.coo_matrix((data, (row, col)), shape=distance_matrix.shape)
    
    return graph.tocsr()


def pairwise_distances_kneighbors(
    X, 
    metric: str, 
    n_neighbors: int = None, 
    n_jobs: int = 1, 
    redundant_form: bool = True, 
    include_self: bool = False,
    symmetric: bool = True,
    **kws,
):
    """
    Calculate pairwise distances or k-nearest neighbors distances between samples.
    
    Provides a unified interface for computing either full pairwise distances or
    sparse k-nearest neighbor distances, with options for symmetrization and
    output format.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or DataFrame
        Input data matrix. If DataFrame, index will be preserved in output.
        
    metric : str or callable
        Distance metric to use (e.g., 'euclidean', 'cosine', 'correlation').
        Passed to sklearn.metrics.pairwise_distances.
        
    n_neighbors : int, optional
        Number of nearest neighbors. If None, computes full pairwise distances.
        If provided, computes sparse kNN distance matrix.
        
    n_jobs : int, default=1
        Number of parallel jobs for distance computation.
        
    redundant_form : bool, default=True
        If True, returns full (n_samples, n_samples) matrix.
        If False, returns condensed 1D array of unique pairwise distances.
        
    include_self : bool, default=False
        Whether to include each sample as its own neighbor in kNN computation.
        Only applies when n_neighbors is not None.
        
    symmetric : bool, default=True
        Whether to symmetrize the kNN distance matrix by taking element-wise maximum.
        Only applies when n_neighbors is not None.
        
    **kws : dict
        Additional keyword arguments passed to the distance metric function.
        
    Returns
    -------
    distances : ndarray or DataFrame or Series
        Distance matrix in requested format:
        - If redundant_form=True and X is array: ndarray of shape (n_samples, n_samples)
        - If redundant_form=True and X is DataFrame: DataFrame with sample indices
        - If redundant_form=False and X is array: 1D condensed distance array
        - If redundant_form=False and X is DataFrame: Series with frozenset indices
        
    Notes
    -----
    When n_neighbors is provided and redundant_form=False, the condensed form may
    contain zeros for non-neighbor pairs, which differs from standard condensed
    distance matrices.
        
    Examples
    --------
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> 
    >>> # Full pairwise distances
    >>> dists = pairwise_distances_kneighbors(X, metric='euclidean')
    >>> 
    >>> # Sparse kNN distances
    >>> knn_dists = pairwise_distances_kneighbors(
    ...     X, metric='euclidean', n_neighbors=2, symmetric=True
    ... )
    >>> 
    >>> # Condensed form
    >>> condensed = pairwise_distances_kneighbors(
    ...     X, metric='euclidean', redundant_form=False
    ... )
    """
    # Handle DataFrame input
    if isinstance(X, pd.DataFrame):
        samples = X.index
        X = X.to_numpy()
    else:
        samples = None

    if n_neighbors is None:
        # Calculate full pairwise distance matrix
        distances = pairwise_distances(X, metric=metric, n_jobs=n_jobs, **kws)
    else:
        # Calculate sparse kNN distances using transformer
        n_neighbors_adj = n_neighbors - 1 if include_self else n_neighbors
        knn_transformer = KNeighborsTransformer(
            n_neighbors=n_neighbors_adj,
            mode='distance',
            metric=metric,
            n_jobs=n_jobs,
            **kws
        )
        distances = knn_transformer.fit_transform(X)
        
        # Convert sparse to dense
        distances = np.array(distances.todense())
        
        # Add self-connections if requested
        if include_self:
            np.fill_diagonal(distances, 0.0)
        
        # Symmetrize if requested
        if symmetric:
            distances = np.maximum(distances, distances.T)
    
    # Return in requested format
    if redundant_form:
        if samples is not None:
            return pd.DataFrame(distances, index=samples, columns=samples)
        else:
            return distances
    else:
        # Convert to condensed form
        distances = squareform(distances, checks=False)
        if samples is not None:
            combinations_samples = pd.Index(map(frozenset, combinations(samples, 2)))
            return pd.Series(distances, index=combinations_samples)
        else:
            return distances


def convert_distance_matrix_to_kneighbors_matrix(
    distance_matrix, 
    n_neighbors: int, 
    redundant_form: bool = True,
    include_self: bool = False, 
    symmetric: bool = True,
):
    """
    Convert a fully-connected distance matrix to a sparse k-nearest neighbors matrix.
    
    Takes a dense pairwise distance matrix and creates a sparse version containing
    only the k-nearest neighbors for each sample, with optional symmetrization.
    
    Parameters
    ----------
    distance_matrix : array-like of shape (n_samples, n_samples) or DataFrame
        Full pairwise distance matrix. If DataFrame, index/columns are preserved.
        
    n_neighbors : int
        Number of nearest neighbors to retain for each sample.
        
    redundant_form : bool, default=True
        If True, returns full (n_samples, n_samples) matrix with zeros for non-neighbors.
        If False, returns condensed 1D array.
        
    include_self : bool, default=False
        Whether to include each sample as one of its own k-nearest neighbors.
        If False, diagonal is excluded from neighbor selection.
        
    symmetric : bool, default=True
        Whether to symmetrize the result by taking element-wise maximum.
        Ensures if A is a neighbor of B, then B is also marked as neighbor of A.
        
    Returns
    -------
    knn_matrix : ndarray or DataFrame or Series
        Sparse k-nearest neighbors distance matrix:
        - If redundant_form=True: same shape as input, with non-neighbor distances = 0
        - If redundant_form=False: condensed 1D array
        - DataFrame/Series if input was DataFrame, ndarray/array otherwise
        
    Notes
    -----
    When redundant_form=False, the condensed form will contain zeros for non-neighbor
    pairs, which differs from standard condensed distance matrices.
        
    Examples
    --------
    >>> distances = np.array([[0., 1., 3.],
    ...                       [1., 0., 2.],
    ...                       [3., 2., 0.]])
    >>> 
    >>> # Keep only 2 nearest neighbors per sample
    >>> knn = convert_distance_matrix_to_kneighbors_matrix(
    ...     distances, n_neighbors=2, include_self=False
    ... )
    >>> # Result will have only 2 non-zero values per row
    >>> 
    >>> # Symmetric version ensures mutual neighbors
    >>> knn_sym = convert_distance_matrix_to_kneighbors_matrix(
    ...     distances, n_neighbors=2, symmetric=True
    ... )
    """
    # Handle DataFrame input
    if isinstance(distance_matrix, pd.DataFrame):
        samples = distance_matrix.index
        distance_matrix = distance_matrix.to_numpy()
    else:
        samples = None
        
    n = distance_matrix.shape[0]
    knn_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # For each sample, find k nearest neighbors
    for i in range(n):
        if not include_self:
            # Exclude diagonal element
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            sorted_indices = np.argsort(distance_matrix[i][mask])
            # Map back to original indices
            orig_indices = np.arange(n)[mask][sorted_indices]
            knn_indices = orig_indices[:n_neighbors]
        else:
            # Include self as potential neighbor
            sorted_indices = np.argsort(distance_matrix[i])
            knn_indices = sorted_indices[:n_neighbors]
        
        # Assign distances to k nearest neighbors
        knn_matrix[i, knn_indices] = distance_matrix[i, knn_indices]
    
    # Symmetrize by taking maximum
    if symmetric:
        knn_matrix = np.maximum(knn_matrix, knn_matrix.T)
    
    # Return in requested format
    if redundant_form:
        if samples is not None:
            return pd.DataFrame(knn_matrix, index=samples, columns=samples)
        else:
            return knn_matrix
    else:
        # Convert to condensed form
        knn_matrix = squareform(knn_matrix, checks=False)
        if samples is not None:
            combinations_samples = pd.Index(map(frozenset, combinations(samples, 2)))
            return pd.Series(knn_matrix, index=combinations_samples)
        else:
            return knn_matrix
        
def kneighbors_to_igraph(D, I, index=None, include_self=False):
    """
    Convert k-nearest neighbors results to igraph.
    
    Parameters
    ----------
    D : np.ndarray, shape (n, k)
        Cosine similarities to k nearest neighbors (higher = more similar)
    I : np.ndarray, shape (n, k)
        Indices of k nearest neighbors
    index : array-like or None, default=None
        Node labels/IDs. If None, uses integer indices 0, 1, ..., n-1
    include_self : bool, default=False
        Whether to include self-loops
    
    Returns
    -------
    ig.Graph
        Directed graph with edges weighted by cosine similarity
    """
    import igraph as ig
    n, k = I.shape
    
    if not include_self:
        I = I[:, 1:]
        D = D[:, 1:]
        k = k - 1
    
    # Vectorized edge construction
    sources = np.repeat(np.arange(n), k)
    targets = I.flatten()
    weights = D.flatten()
    
    # Map to node labels only if index is provided and non-scalar
    if index is not None:
        index_array = np.asarray(index)
        if index_array.ndim > 0:  # Check it's actually an array
            sources = index_array[sources]
            targets = index_array[targets]
    
    # Create edge list
    edges = list(zip(sources, targets, weights))
    
    # If you compute ALL pairwise similarities
    # sim(A, B) == sim(B, A)  # Cosine similarity is symmetric
    # With kNN, you only keep top-k neighbors
    # A's neighbors: [B, C, D]  # B is one of A's 3 nearest neighbors
    # B's neighbors: [X, Y, Z]  # A might NOT be one of B's 3 nearest neighbors!
    graph = ig.Graph.TupleList(edges, weights=True, directed=True)
    return graph

def kneighbors_classification_assignment_score(df_proba, tol_probability=0.1, tol_assignment_score=2.0):
    """
    Compute assignment quality metrics for KNN classification.
    
    Parameters
    ----------
    df_proba : pd.DataFrame
        Probability matrix (rows=queries, columns=classes) from KNeighborsClassifier.predict_proba
    tol_probability : float, default=0.1
        Minimum top class probability for unambiguous assignment
    tol_assignment_score : float, default=2.0
        Minimum ratio of top / runner-up probability for unambiguous assignment
    
    Returns
    -------
    pd.DataFrame
        Per-query metrics with columns:
        - max_probability: top class probability
        - runner_up_probability: second class probability
        - assignment_score: max_probability / runner_up_probability ratio
        - prediction: class with highest probability
        - not_ambiguous: bool, meets both thresholds
    """
    eps = np.finfo(df_proba.values.dtype).eps
    top2 = np.sort(df_proba.values, axis=1)[:, -2:]
    max_probability = top2[:, 1]
    runner_up_probability = top2[:, 0]
    assignment_score = max_probability / (runner_up_probability + eps)

    df_assignment = pd.DataFrame({
        "max_probability": max_probability,
        "runner_up_probability": runner_up_probability,
        "assignment_score": assignment_score,
        "prediction": df_proba.idxmax(axis=1),
        "not_ambiguous": (max_probability >= tol_probability) & (assignment_score >= tol_assignment_score),
    }, index=df_proba.index)

    return df_assignment

# ══════════════════════════════════════════════════════════════════════════════
# Private utilities
# ══════════════════════════════════════════════════════════════════════════════

def _to_numpy_with_index(X):
    """
    Convert X to float32 numpy array and extract row index if available.

    Returns
    -------
    X_arr : np.ndarray, dtype float32
    row_index : pd.Index or None
    """
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32), X.index
    if isinstance(X, pd.Series):
        return X.values.astype(np.float32).reshape(1, -1), pd.Index([X.name])
    return np.asarray(X, dtype=np.float32), None


def _determine_backend(backend):
    """
    Resolve backend preference to 'faiss' or 'sklearn'.

    Parameters
    ----------
    backend : {'auto', 'faiss', 'sklearn'}

    Returns
    -------
    str : 'faiss' or 'sklearn'
    """
    if backend == "sklearn":
        return "sklearn"
    if backend == "faiss":
        try:
            import faiss  # noqa: F401
            return "faiss"
        except ImportError:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
    # auto
    try:
        import faiss  # noqa: F401
        return "faiss"
    except ImportError:
        warnings.warn("FAISS not available, falling back to sklearn.", UserWarning)
        return "sklearn"


def _build_faiss_index(X, mode="exact", n_voronoi_cells="auto", n_probes=1,
                       n_subvectors=None, n_bits=8):
    """
    Build and return a fitted FAISS index for inner product (cosine on L2-normed data).

    Parameters
    ----------
    X : np.ndarray, float32, L2-normalized
    mode : {'exact', 'ivf', 'pq'}
    n_voronoi_cells : int or 'auto'
    n_probes : int
    n_subvectors : int or None
    n_bits : int

    Returns
    -------
    faiss.Index
    """
    import faiss

    n_samples, d = X.shape

    if mode == "exact":
        index = faiss.IndexFlatIP(d)
        index.add(X)

    elif mode == "ivf":
        nlist = int(np.sqrt(n_samples)) if n_voronoi_cells == "auto" else n_voronoi_cells
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(X)
        index.add(X)
        index.nprobe = n_probes

    elif mode == "pq":
        if n_subvectors is None:
            m = d // 16
            while d % m != 0 and m > 1:
                m -= 1
            if m == 1:
                raise ValueError(
                    f"Cannot determine n_subvectors for dimension {d}. "
                    f"Specify n_subvectors that divides {d} evenly."
                )
        else:
            m = n_subvectors
            if d % m != 0:
                raise ValueError(f"n_subvectors ({m}) must divide dimension ({d}) evenly.")
        index = faiss.IndexPQ(d, m, n_bits)
        index.train(X)
        index.add(X)

    else:
        raise ValueError(f"mode must be 'exact', 'ivf', or 'pq', got '{mode}'")

    return index


def _search_index(index, X_query, k, backend, X_fit=None):
    """
    Run nearest-neighbor search against a fitted index.

    Parameters
    ----------
    index : faiss.Index or None
        FAISS index (None if sklearn backend).
    X_query : np.ndarray, float32
    k : int
    backend : str, 'faiss' or 'sklearn'
    X_fit : np.ndarray or None
        Required when backend='sklearn'.

    Returns
    -------
    similarities : np.ndarray, shape (n_query, k)
    indices : np.ndarray, shape (n_query, k)
    """
    if backend == "faiss":
        return index.search(X_query, k)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_fit)
    distances, indices = nn.kneighbors(X_query)
    return 1 - distances, indices  # convert distance → similarity




# ══════════════════════════════════════════════════════════════════════════════
# KNeighborsCosineGraph (v1)
# ══════════════════════════════════════════════════════════════════════════════
# class KNeighborsCosineGraph(BaseEstimator, TransformerMixin):
#     """
#     K-Nearest Neighbors using cosine similarity.

#     Parameters
#     ----------
#     n_neighbors : int
#         Number of neighbors to find.
#     mode : {'exact', 'ivf', 'pq'}, default='exact'
#         Search strategy.
#     backend : {'auto', 'faiss', 'sklearn'}, default='auto'
#         Which library to use. 'auto' prefers FAISS, falls back to sklearn.
#     n_voronoi_cells : int or 'auto', default='auto'
#         Number of IVF cells. If 'auto', uses sqrt(n_samples).
#     n_probes : int, default=1
#         Number of cells to search in IVF.
#     n_subvectors : int or None, default=None
#         Number of sub-vectors for PQ. If None, uses d//16.
#     n_bits : int, default=8
#         Bits per sub-vector for PQ.

#     Attributes
#     ----------
#     backend_ : str
#     index_ : faiss.Index or None
#     similarities_ : np.ndarray, shape (n_samples_fit, n_neighbors)
#     indices_ : np.ndarray, shape (n_samples_fit, n_neighbors)
#     """

#     def __init__(
#         self,
#         n_neighbors,
#         mode="exact",
#         backend="auto",
#         n_voronoi_cells="auto",
#         n_probes=1,
#         n_subvectors=None,
#         n_bits=8,
#     ):
#         self.n_neighbors = n_neighbors
#         self.mode = mode
#         self.backend = backend
#         self.n_voronoi_cells = n_voronoi_cells
#         self.n_probes = n_probes
#         self.n_subvectors = n_subvectors
#         self.n_bits = n_bits

#     def fit(self, X, y=None):
#         """
#         Fit the k-NN model.

#         Parameters
#         ----------
#         X : array-like, shape (n_samples, n_features)
#             L2-normalized training data.
#         y : Ignored
#         """
#         self.index_labels_ = None
#         if isinstance(X, pd.DataFrame):
#             self.index_labels_ = X.index
#         X_arr, row_index = _to_numpy_with_index(X)
#         X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)



#         self.n_samples_fit_ = X_arr.shape[0]
#         self.n_features_in_ = X_arr.shape[1]
#         self.backend_ = _determine_backend(self.backend)

#         if self.backend_ == "faiss":
#             self.index_ = _build_faiss_index(
#                 X_arr,
#                 mode=self.mode,
#                 n_voronoi_cells=self.n_voronoi_cells,
#                 n_probes=self.n_probes,
#                 n_subvectors=self.n_subvectors,
#                 n_bits=self.n_bits,
#             )
#             self.X_fit_ = None
#         else:
#             if self.mode != "exact":
#                 warnings.warn(
#                     f"sklearn backend only supports exact search, ignoring mode='{self.mode}'.",
#                     UserWarning,
#                 )
#             self.X_fit_ = X_arr
#             self.index_ = None

#         self.similarities_, self.indices_ = self.transform(X_arr)
#         return self

#     def transform(self, X):
#         """
#         Find k-nearest neighbors.

#         Parameters
#         ----------
#         X : array-like, shape (n_samples, n_features)
#             L2-normalized query vectors.

#         Returns
#         -------
#         similarities : np.ndarray, shape (n_samples, n_neighbors)
#         indices : np.ndarray, shape (n_samples, n_neighbors)
#         """
#         check_is_fitted(self)
#         X_arr, _ = _to_numpy_with_index(X)
#         X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)

#         if X_arr.shape[1] != self.n_features_in_:
#             raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}.")
#         if self.n_neighbors > self.n_samples_fit_:
#             raise ValueError(
#                 f"n_neighbors ({self.n_neighbors}) > n_samples ({self.n_samples_fit_})."
#             )

#         return _search_index(
#             self.index_, X_arr, self.n_neighbors, self.backend_,
#             X_fit=self.X_fit_,
#         )

#     def fit_transform(self, X, y=None):
#         """Fit and return neighbors for training data."""
#         self.fit(X, y)
#         return self.similarities_, self.indices_

#     def to_igraph(self, index="auto", include_self=False):
#         """
#         Convert fitted k-NN results to igraph.

#         Parameters
#         ----------
#         index : array-like or 'auto'
#             Node labels. 'auto' uses DataFrame index if available, else integers.
#         include_self : bool, default=False

#         Returns
#         -------
#         ig.Graph
#         """
#         check_is_fitted(self, ["similarities_", "indices_"])

#         if index == "auto":
#             index = getattr(self, "index_labels_", None)
#         elif isinstance(index, pd.Index):
#             index = list(index)

#         return kneighbors_to_igraph(
#             self.similarities_,
#             self.indices_,
#             index=index,
#             include_self=include_self,
#         )

# # ============================================================================
# # KNeighborsCosineGraph (v2)
# # ============================================================================
# class KNeighborsCosineGraph(BaseEstimator, TransformerMixin):
#     """
#     K-Nearest Neighbors graph using cosine similarity.
 
#     Uses FAISS (preferred) or sklearn backend. Stores similarities
#     as the native output format for cosine.
 
#     Parameters
#     ----------
#     n_neighbors : int
#         Number of neighbors (includes self as neighbor 0).
#     mode : str
#         "exact", "ivf", or "pq".
#     backend : str
#         "auto", "faiss", or "sklearn".
#     n_voronoi_cells : int or "auto"
#     n_probes : int
#     n_subvectors : int or None
#     n_bits : int
 
#     Attributes
#     ----------
#     similarities_ : np.ndarray, shape (n_samples, n_neighbors)
#         Cosine similarities (column 0 = self).
#     indices_ : np.ndarray, shape (n_samples, n_neighbors)
#         Neighbor indices (column 0 = self).
#     labels_ : pd.Index or None
#         Node labels from DataFrame index.
#     """
 
#     def __init__(
#         self,
#         n_neighbors,
#         mode="exact",
#         backend="auto",
#         n_voronoi_cells="auto",
#         n_probes=1,
#         n_subvectors=None,
#         n_bits=8,
#     ):
#         self.n_neighbors = n_neighbors
#         self.mode = mode
#         self.backend = backend
#         self.n_voronoi_cells = n_voronoi_cells
#         self.n_probes = n_probes
#         self.n_subvectors = n_subvectors
#         self.n_bits = n_bits
 
#     def fit(self, X, y=None):
#         """
#         Fit KNN and compute neighbors for training data.
 
#         Parameters
#         ----------
#         X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
#             L2-normalized data.
#         """
#         self.labels_ = X.index if isinstance(X, pd.DataFrame) else None
 
#         X_arr, _ = _to_numpy_with_index(X)
#         X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)
 
#         self.n_samples_fit_ = X_arr.shape[0]
#         self.n_features_in_ = X_arr.shape[1]
#         self.backend_ = _determine_backend(self.backend)
 
#         if self.backend_ == "faiss":
#             self.index_ = _build_faiss_index(
#                 X_arr, mode=self.mode,
#                 n_voronoi_cells=self.n_voronoi_cells,
#                 n_probes=self.n_probes,
#                 n_subvectors=self.n_subvectors,
#                 n_bits=self.n_bits,
#             )
#             self.X_fit_ = None
#         else:
#             if self.mode != "exact":
#                 warnings.warn(f"sklearn backend only supports exact search, ignoring mode='{self.mode}'.")
#             self.X_fit_ = X_arr
#             self.index_ = None
 
#         self.similarities_, self.indices_ = _search_index(
#             self.index_, X_arr, self.n_neighbors, self.backend_,
#             X_fit=self.X_fit_,
#         )
#         return self
 
#     def transform(self, X):
#         """Find neighbors for new data. Returns (similarities, indices)."""
#         check_is_fitted(self)
#         X_arr, _ = _to_numpy_with_index(X)
#         X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)
 
#         if X_arr.shape[1] != self.n_features_in_:
#             raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}.")
 
#         return _search_index(
#             self.index_, X_arr, self.n_neighbors, self.backend_,
#             X_fit=self.X_fit_,
#         )
 
#     def fit_transform(self, X, y=None):
#         self.fit(X, y)
#         return self.similarities_, self.indices_
 
#     def __repr__(self):
#         return f"KNeighborsCosineGraph(n_neighbors={self.n_neighbors})"

# ============================================================================
# KNeighborsCosineGraph
# ============================================================================
class KNeighborsCosineGraph(BaseEstimator, TransformerMixin):
    """
    K-Nearest Neighbors graph using cosine similarity with automatic
    k selection via knee detection.

    Uses FAISS (preferred) or sklearn backend. Stores similarities
    as the native output format for cosine.

    Parameters
    ----------
    n_neighbors : int
        Maximum number of neighbors to compute (includes self as neighbor 0).
        Acts as k_max for knee detection and parameter sweeps.
    detect_optimal_k : bool, default=True
        If True, run knee detection during fit() to select optimal k.
        If False, skip knee detection; to_igraph() uses full n_neighbors.
    sensitivity : float
        Kneedle sensitivity for knee detection (higher = less sensitive).
    aggregation : {'median', 'mean'}, default='median'
        Aggregation function for the k-similarity curve used in knee detection.
    direction : {'increasing', 'decreasing', 'auto'}, default='decreasing'
        Direction of the k-similarity curve for KneeLocator.
        'auto' uses kneed.find_shape to infer the direction.
    curve : {'concave', 'convex', 'auto'}, default='convex'
        Curvature of the k-similarity curve for KneeLocator.
        'auto' uses kneed.find_shape to infer the curvature.
    mode : str
        "exact", "ivf", or "pq".
    backend : str
        "auto", "faiss", or "sklearn".
    n_voronoi_cells : int or "auto"
    n_probes : int
    n_subvectors : int or None
    n_bits : int

    Attributes
    ----------
    similarities_ : np.ndarray, shape (n_samples, n_neighbors)
        Cosine similarities (column 0 = self).
    indices_ : np.ndarray, shape (n_samples, n_neighbors)
        Neighbor indices (column 0 = self).
    labels_ : pd.Index or None
        Node labels from DataFrame index.
    k_ : int or None
        Selected k from knee detection, or None if not detected.
    max_k_ : int
        Maximum usable k (n_neighbors - 1, excluding self).
    k_similarity_curve_ : pd.Series
        Aggregated similarity to k-th neighbor for each k.
    k_similarity_q25_ : np.ndarray or None
        25th percentile of k-th neighbor similarity.
    k_similarity_q75_ : np.ndarray or None
        75th percentile of k-th neighbor similarity.
    kneedle_ : KneeLocator or None
        Kneedle object (if knee detection was run).

    Examples
    --------
    >>> # Auto k detection (default)
    >>> knn = KNeighborsCosineGraph(n_neighbors=150)
    >>> knn.fit(X_l2)
    >>> graph = knn.to_igraph()       # uses detected k_
    >>>
    >>> # Skip knee detection; to_igraph uses full neighborhood
    >>> knn = KNeighborsCosineGraph(n_neighbors=150, detect_optimal_k=False)
    >>> knn.fit(X_l2)
    >>> graph = knn.to_igraph()       # uses max_k_ (149)
    >>> graph = knn.to_igraph(k=30)   # explicit k
    >>>
    >>> # Run knee detection later
    >>> knn.detect_knee()
    >>> graph = knn.to_igraph()       # now uses detected k_
    >>>
    >>> # Re-run with different sensitivity
    >>> knn.detect_knee(sensitivity=3.0)
    >>>
    >>> # Parameter sweep (stateless)
    >>> for k in [20, 30, 50]:
    ...     graph = knn.to_igraph(k=k)
    """

    _VALID_AGGREGATIONS = {"median", "mean"}
    _VALID_DIRECTIONS = {"increasing", "decreasing", "auto"}
    _VALID_CURVES = {"concave", "convex", "auto"}

    def __init__(
        self,
        n_neighbors,
        detect_optimal_k=True,
        sensitivity=1.0,
        aggregation="median",
        direction="decreasing",
        curve="convex",
        mode="exact",
        backend="auto",
        n_voronoi_cells="auto",
        n_probes=1,
        n_subvectors=None,
        n_bits=8,
    ):
        if aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {self._VALID_AGGREGATIONS}, got '{aggregation}'"
            )
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {self._VALID_DIRECTIONS}, got '{direction}'"
            )
        if curve not in self._VALID_CURVES:
            raise ValueError(
                f"curve must be one of {self._VALID_CURVES}, got '{curve}'"
            )

        self.n_neighbors = n_neighbors
        self.detect_optimal_k = detect_optimal_k
        self.sensitivity = sensitivity
        self.aggregation = aggregation
        self.direction = direction
        self.curve = curve
        self.mode = mode
        self.backend = backend
        self.n_voronoi_cells = n_voronoi_cells
        self.n_probes = n_probes
        self.n_subvectors = n_subvectors
        self.n_bits = n_bits

    def fit(self, X, y=None):
        """
        Fit KNN, compute neighbors, build k-similarity curve, and detect knee.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            L2-normalized data.
        """
        self.labels_ = X.index if isinstance(X, pd.DataFrame) else None

        X_arr, _ = _to_numpy_with_index(X)
        X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)

        self.n_samples_fit_ = X_arr.shape[0]
        self.n_features_in_ = X_arr.shape[1]
        self.max_k_ = self.n_neighbors - 1  # exclude self (column 0)
        self.backend_ = _determine_backend(self.backend)

        if self.backend_ == "faiss":
            self.index_ = _build_faiss_index(
                X_arr, mode=self.mode,
                n_voronoi_cells=self.n_voronoi_cells,
                n_probes=self.n_probes,
                n_subvectors=self.n_subvectors,
                n_bits=self.n_bits,
            )
            self.X_fit_ = None
        else:
            if self.mode != "exact":
                warnings.warn(f"sklearn backend only supports exact search, ignoring mode='{self.mode}'.")
            self.X_fit_ = X_arr
            self.index_ = None

        self.similarities_, self.indices_ = _search_index(
            self.index_, X_arr, self.n_neighbors, self.backend_,
            X_fit=self.X_fit_,
        )

        # Build k-similarity curve (always, for detect_knee if called later)
        self._build_k_similarity_curve()

        # Resolve k
        self.k_ = None
        self.kneedle_ = None
        if self.detect_optimal_k:
            self.detect_knee()

        return self

    def _build_k_similarity_curve(self):
        """Compute aggregated similarity to k-th neighbor (with IQR) for each k."""
        k_values = np.arange(1, self.max_k_ + 1)
        data = self.similarities_[:, 1:]  # skip self (column 0)

        if self.aggregation == "median":
            agg_values = np.median(data, axis=0)
        else:
            agg_values = np.mean(data, axis=0)

        self.k_similarity_curve_ = pd.Series(
            agg_values,
            index=k_values,
            name=f"{self.aggregation}_kth_neighbor_similarity",
        )
        self.k_similarity_curve_.index.name = "k"

        # IQR
        self.k_similarity_q25_ = np.percentile(data, 25, axis=0)
        self.k_similarity_q75_ = np.percentile(data, 75, axis=0)

    def detect_knee(self, sensitivity=None, aggregation=None, direction=None, curve=None):
        """
        (Re)run knee detection, optionally updating parameters.

        Call after fit() to try different sensitivity or curve settings
        without recomputing the KNN.

        Parameters
        ----------
        sensitivity : float or None
            New Kneedle sensitivity. None keeps current.
        aggregation : {'median', 'mean'} or None
            New aggregation. Rebuilds the k-similarity curve if changed.
        direction : {'increasing', 'decreasing', 'auto'} or None
            New direction. None keeps current.
        curve : {'concave', 'convex', 'auto'} or None
            New curvature. None keeps current.

        Returns
        -------
        self
        """
        check_is_fitted(self, ["similarities_"])

        if aggregation is not None and aggregation != self.aggregation:
            if aggregation not in self._VALID_AGGREGATIONS:
                raise ValueError(
                    f"aggregation must be one of {self._VALID_AGGREGATIONS}, got '{aggregation}'"
                )
            self.aggregation = aggregation
            self._build_k_similarity_curve()
        if sensitivity is not None:
            self.sensitivity = sensitivity
        if direction is not None:
            if direction not in self._VALID_DIRECTIONS:
                raise ValueError(
                    f"direction must be one of {self._VALID_DIRECTIONS}, got '{direction}'"
                )
            self.direction = direction
        if curve is not None:
            if curve not in self._VALID_CURVES:
                raise ValueError(
                    f"curve must be one of {self._VALID_CURVES}, got '{curve}'"
                )
            self.curve = curve

        from kneed import KneeLocator

        _direction = self.direction
        _curve = self.curve

        if _direction == "auto" or _curve == "auto":
            from kneed import find_shape
            auto_direction, auto_curve = find_shape(
                self.k_similarity_curve_.index.values,
                self.k_similarity_curve_.values,
            )
            if _direction == "auto":
                _direction = auto_direction
            if _curve == "auto":
                _curve = auto_curve

        logger.info(f"Curve shape: direction={_direction}, curve={_curve}")

        self.kneedle_ = KneeLocator(
            self.k_similarity_curve_.index.values,
            self.k_similarity_curve_.values,
            curve=_curve,
            direction=_direction,
            S=self.sensitivity,
        )

        if self.kneedle_.knee is None:
            logger.warning("No knee detected, defaulting to midpoint")
            self.k_ = int(self.k_similarity_curve_.index[len(self.k_similarity_curve_) // 2])
        else:
            self.k_ = int(self.kneedle_.knee)

        logger.info(f"Auto k={self.k_} (sensitivity={self.sensitivity})")
        return self

    def to_igraph(self, k=None):
        """
        Build directed igraph from KNN at given k.

        KNN graphs are inherently directed: A having B as a neighbor
        does not imply B has A as a neighbor.

        Parameters
        ----------
        k : int or None
            Number of neighbors. None uses self.k_ if available,
            otherwise falls back to max_k_ (full neighborhood).
            Does not modify state when k is specified explicitly.

        Returns
        -------
        ig.Graph
            Directed graph with edges weighted by cosine similarity.
        """
        import igraph as ig

        check_is_fitted(self, ["similarities_", "indices_"])

        if k is None:
            k = self.k_ if self.k_ is not None else self.max_k_

        if k > self.max_k_:
            raise ValueError(f"k={k} exceeds max_k_={self.max_k_}")

        # Skip self (column 0), take k neighbors
        I = self.indices_[:, 1:k + 1]
        D = self.similarities_[:, 1:k + 1]
        n = I.shape[0]

        sources = np.repeat(np.arange(n), k)
        targets = I.flatten()
        weights = D.flatten()

        if self.labels_ is not None:
            labels_array = np.asarray(self.labels_)
            sources = labels_array[sources]
            targets = labels_array[targets]

        edges = list(zip(sources, targets, weights))
        graph = ig.Graph.TupleList(edges, weights=True, directed=True)
        return graph

    def to_kneighbors_graph(self, k=None, mode="connectivity", include_self=False):
        """
        Convert KNN indices to a sparse connectivity or distance matrix.

        Equivalent to ``sklearn.neighbors.kneighbors_graph``.

        Parameters
        ----------
        k : int or None
            Number of neighbors. None uses ``k_`` if available,
            otherwise ``max_k_``. Does not modify state.
        mode : {'connectivity', 'distance'}, default='connectivity'
            ``'connectivity'`` returns binary adjacency.
            ``'distance'`` returns cosine distances (1 - similarity) as weights.
        include_self : bool, default=False
            If True, include self-connections (diagonal entries).

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_samples, n_samples)
            Asymmetric sparse matrix (A[i,j] = 1 or distance if j is
            a k-neighbor of i).
        """
        from scipy.sparse import csr_matrix

        check_is_fitted(self, ["indices_"])

        if mode not in ("connectivity", "distance"):
            raise ValueError(
                f"mode must be 'connectivity' or 'distance', got '{mode}'"
            )

        if k is None:
            k = self.k_ if self.k_ is not None else self.max_k_
        if k > self.max_k_:
            raise ValueError(f"k={k} exceeds max_k_={self.max_k_}")

        n = self.n_samples_fit_

        if include_self:
            idx_slice = slice(0, k + 1)
        else:
            idx_slice = slice(1, k + 1)

        neighbor_idx = self.indices_[:, idx_slice]
        n_per_row = neighbor_idx.shape[1]

        rows = np.repeat(np.arange(n), n_per_row)
        cols = neighbor_idx.ravel()

        if mode == "connectivity":
            data = np.ones(len(rows), dtype=np.float64)
        else:
            data = np.clip(1 - self.similarities_[:, idx_slice].ravel(), 0, 2)

        return csr_matrix((data, (rows, cols)), shape=(n, n))

    def plot(self, ax=None, figsize=(8, 5),
             xlabel=None, ylabel=None, title=None,
             curve_color="black", iqr_color="steelblue",
             vline_color="firebrick"):
        """
        Diagnostic plot: k-similarity elbow with IQR and detected knee.

        Requires that knee detection has been run (either via
        detect_optimal_k=True during fit, or by calling detect_knee()).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        figsize : tuple, default=(8, 5)
            Figure size if creating new axes.
        xlabel : str or None
            X-axis label. Default: '$N_{Neighbors}$ [k]'.
        ylabel : str or None
            Y-axis label. Default: '{Aggregation} cosine similarity to k-th neighbor'.
        title : str or None
            Plot title.
        curve_color : str, default='black'
            Color for the aggregation curve.
        iqr_color : str, default='steelblue'
            Color for the IQR fill.
        vline_color : str, default='firebrick'
            Color for the knee vertical line.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if self.kneedle_ is None:
            raise ValueError(
                "Knee detection has not been run. "
                "Use detect_optimal_k=True during fit or call detect_knee() first."
            )

        if self.k_similarity_curve_ is None:
            self._build_k_similarity_curve()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        k_vals = self.k_similarity_curve_.index.values

        ax.plot(
            k_vals, self.k_similarity_curve_.values,
            color=curve_color, linewidth=2,
            label=self.aggregation.capitalize(),
        )

        if self.k_similarity_q25_ is not None:
            ax.fill_between(
                k_vals,
                self.k_similarity_q25_,
                self.k_similarity_q75_,
                alpha=0.2, color=iqr_color, label="IQR",
            )

        if self.k_ is not None:
            ax.axvline(
                self.k_, color=vline_color, linestyle="--",
                linewidth=1.5, label=f"Knee (k={self.k_})",
            )

        ax.set_xlabel(xlabel or "$N_{Neighbors}$ [k]")
        ax.set_ylabel(
            ylabel or f"{self.aggregation.capitalize()} cosine similarity to k-th neighbor"
        )
        if title is not None:
            ax.set_title(title)

        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return ax

    def transform(self, X):
        """Find neighbors for new data. Returns (similarities, indices)."""
        check_is_fitted(self)
        X_arr, _ = _to_numpy_with_index(X)
        X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}.")

        return _search_index(
            self.index_, X_arr, self.n_neighbors, self.backend_,
            X_fit=self.X_fit_,
        )

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.similarities_, self.indices_

    def __repr__(self):
        if hasattr(self, "k_") and self.k_ is not None:
            k_str = f"k_={self.k_}"
        elif hasattr(self, "max_k_"):
            k_str = f"max_k_={self.max_k_}"
        else:
            k_str = "unfitted"
        return (
            f"KNeighborsCosineGraph(n_neighbors={self.n_neighbors}, {k_str}, "
            f"detect_optimal_k={self.detect_optimal_k})"
        )
 
 
# ============================================================================
# SharedNearestNeighborsGraph
# ============================================================================
class SharedNearestNeighborsGraph:
    """
    Shared Nearest Neighbor graph from precomputed KNN.
 
    Parameters
    ----------
    indices : np.ndarray, shape (n_samples, max_k)
        KNN neighbor indices.
    distances : np.ndarray, shape (n_samples, max_k)
        Distances to neighbors (for knee detection).
    labels : pd.Index, list, or np.ndarray, optional
        Node names/IDs.
    k : int or None
        Number of neighbors for SNN. None = auto via knee detection.
    sensitivity : float
        Kneedle sensitivity for knee detection (higher = less sensitive).
    aggregation : {'median', 'mean'}, default='median'
        Aggregation function for the k-distance curve used in knee detection.
    direction : {'increasing', 'decreasing', 'auto'}, default='increasing'
        Direction of the k-distance curve for KneeLocator.
        'auto' uses kneed.find_shape to infer the direction.
    curve : {'concave', 'convex', 'auto'}, default='concave'
        Curvature of the k-distance curve for KneeLocator.
        'auto' uses kneed.find_shape to infer the curvature.
    skip_self : bool
        If True, column 0 contains self-references and is excluded.
 
    Attributes (set by fit)
    -----------------------
    k_ : int
        Selected k.
    k_distance_curve_ : pd.Series
        Aggregated distance to k-th neighbor.
    k_distance_q25_ : np.ndarray or None
        25th percentile of k-th neighbor distance (for IQR plotting).
    k_distance_q75_ : np.ndarray or None
        75th percentile of k-th neighbor distance (for IQR plotting).
    kneedle_ : KneeLocator or None
        Kneedle object (if k was auto-detected).
    snn_adjacency_ : scipy.sparse.csr_matrix
        SNN Jaccard-weighted adjacency matrix.
    graph_ : ig.Graph
        Undirected weighted igraph.
 
    Examples
    --------
    >>> knn = KNeighborsCosineGraph(n_neighbors=150)
    >>> knn.fit(X_l2)
    >>>
    >>> # Auto k with median aggregation (default)
    >>> snn = SharedNearestNeighborsGraph.from_kneighbors_cosine_graph(knn)
    >>> graph = snn.fit_transform()
    >>>
    >>> # Auto k with mean aggregation
    >>> snn = SharedNearestNeighborsGraph.from_kneighbors_cosine_graph(knn, aggregation="mean")
    >>> graph = snn.fit_transform()
    >>>
    >>> # Specified k
    >>> snn = SharedNearestNeighborsGraph.from_kneighbors_cosine_graph(knn, k=30)
    >>> graph = snn.fit_transform()
    >>>
    >>> # Parameter sweep (stateless)
    >>> snn = SharedNearestNeighborsGraph.from_kneighbors_cosine_graph(knn)
    >>> for k in [20, 30, 50]:
    ...     graph = snn.to_igraph(k=k)
    ...     leiden.fit(graph)
    """
 
    _VALID_AGGREGATIONS = {"median", "mean"}
    _VALID_DIRECTIONS = {"increasing", "decreasing", "auto"}
    _VALID_CURVES = {"concave", "convex", "auto"}

    def __init__(self, indices, distances, labels=None, k=None, sensitivity=1.0,
                 aggregation="median", direction="increasing", curve="concave",
                 skip_self=True):
        if aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {self._VALID_AGGREGATIONS}, got '{aggregation}'"
            )
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {self._VALID_DIRECTIONS}, got '{direction}'"
            )
        if curve not in self._VALID_CURVES:
            raise ValueError(
                f"curve must be one of {self._VALID_CURVES}, got '{curve}'"
            )

        self.indices_ = np.asarray(indices)
        self.distances_ = np.asarray(distances)
        self.skip_self_ = skip_self
        self.n_samples_ = self.indices_.shape[0]
        self.k = k
        self.sensitivity = sensitivity
        self.aggregation = aggregation
        self.direction = direction
        self.curve = curve
 
        offset = 1 if skip_self else 0
        self.max_k_ = self.indices_.shape[1] - offset
        self._offset = offset
 
        if labels is not None:
            self.labels_ = labels if isinstance(labels, pd.Index) else pd.Index(labels)
        else:
            self.labels_ = pd.Index(range(self.n_samples_))
 
        if len(self.labels_) != self.n_samples_:
            raise ValueError(
                f"labels length ({len(self.labels_)}) != n_samples ({self.n_samples_})"
            )
 
        # Set by fit
        self.k_ = None
        self.k_distance_curve_ = None
        self.k_distance_q25_ = None
        self.k_distance_q75_ = None
        self.kneedle_ = None
        self.snn_adjacency_ = None
        self.graph_ = None
 
    @classmethod
    def from_kneighbors_cosine_graph(cls, knn, k=None, sensitivity=1.0,
                                      aggregation="median", direction="increasing",
                                      curve="concave"):
        """
        Construct from a fitted KNeighborsCosineGraph.

        Converts similarities -> distances (1 - similarity) internally.

        Parameters
        ----------
        knn : KNeighborsCosineGraph
            Fitted KNN model.
        k : int or None
            Number of neighbors. None = auto.
        sensitivity : float
            Kneedle sensitivity.
        aggregation : {'median', 'mean'}, default='median'
            Aggregation function for knee detection.
        direction : {'increasing', 'decreasing', 'auto'}, default='increasing'
            Direction of the k-distance curve for KneeLocator.
            'auto' uses kneed.find_shape to infer the direction.
        curve : {'concave', 'convex', 'auto'}, default='concave'
            Curvature of the k-distance curve for KneeLocator.
            'auto' uses kneed.find_shape to infer the curvature.

        Returns
        -------
        SharedNearestNeighborsGraph
        """
        check_is_fitted(knn, ["indices_", "similarities_"])
        return cls(
            indices=knn.indices_,
            distances=np.clip(1 - knn.similarities_, 0, 2),
            labels=getattr(knn, "labels_", None),
            k=k,
            sensitivity=sensitivity,
            aggregation=aggregation,
            direction=direction,
            curve=curve,
            skip_self=True,
        )
 
    def _build_k_distance_curve(self):
        """Compute aggregated distance to k-th neighbor (with IQR) for each k."""
        k_values = np.arange(1, self.max_k_ + 1)
        data = self.distances_[:, self._offset:]

        if self.aggregation == "median":
            agg_values = np.median(data, axis=0)
        else:
            agg_values = np.mean(data, axis=0)

        self.k_distance_curve_ = pd.Series(
            agg_values, 
            index=k_values, 
            name=f"{self.aggregation}_kth_neighbor_distance",
        )
        self.k_distance_curve_.index.name = "k"

        # IQR
        self.k_distance_q25_ = np.percentile(data, 25, axis=0)
        self.k_distance_q75_ = np.percentile(data, 75, axis=0)
 
    def _build_snn(self, k):
        """Build SNN Jaccard-weighted sparse matrix at given k."""
        if k > self.max_k_:
            raise ValueError(f"k={k} exceeds max_k_={self.max_k_}")
        return kneighbors_to_snn_graph(
            self.indices_, k=k, skip_self=self.skip_self_,
        )
 
    def fit(self):
        """
        Build the SNN graph and igraph.
 
        If k is None, auto-detects via knee in k-distance curve.
        Curve shape is determined by self.direction and self.curve;
        'auto' defers to kneed.find_shape.
 
        Returns
        -------
        self
        """
        # k-distance curve (always computed for plotting)
        self._build_k_distance_curve()
 
        # Resolve k
        if self.k is None:
            from kneed import KneeLocator

            direction = self.direction
            curve = self.curve

            if direction == "auto" or curve == "auto":
                from kneed import find_shape
                auto_direction, auto_curve = find_shape(
                    self.k_distance_curve_.index.values,
                    self.k_distance_curve_.values,
                )
                if direction == "auto":
                    direction = auto_direction
                if curve == "auto":
                    curve = auto_curve

            logger.info(f"Curve shape: direction={direction}, curve={curve}")

            self.kneedle_ = KneeLocator(
                self.k_distance_curve_.index.values,
                self.k_distance_curve_.values,
                curve=curve,
                direction=direction,
                S=self.sensitivity,
            )
            if self.kneedle_.knee is None:
                logger.warning("No knee detected, defaulting to midpoint")
                self.k_ = int(self.k_distance_curve_.index[len(self.k_distance_curve_) // 2])
            else:
                self.k_ = int(self.kneedle_.knee)
            logger.info(f"Auto k={self.k_} (knee detection, sensitivity={self.sensitivity})")
        else:
            self.k_ = int(self.k)
            logger.info(f"User-specified k={self.k_}")
 
        # Build SNN and igraph
        self.snn_adjacency_ = self._build_snn(self.k_)
        self.graph_ = adjacency_to_igraph(
            self.snn_adjacency_, labels=self.labels_,
        )
 
        n_edges = self.graph_.ecount()
        logger.info(f"SNN graph: {self.n_samples_} nodes, {n_edges} edges")
 
        return self
 
    def fit_transform(self):
        """Fit and return the igraph."""
        return self.fit().graph_
 
    def to_igraph(self, k):
        """
        Build igraph at arbitrary k without modifying state.
 
        For parameter sweeps — does not change self.k_ or self.graph_.
 
        Parameters
        ----------
        k : int
            Number of neighbors.
 
        Returns
        -------
        ig.Graph
        """
        A_snn = self._build_snn(k)
        return adjacency_to_igraph(A_snn, labels=self.labels_)

    def plot(self, axes=None, figsize=(13, 5),
             xlabel_left=None, ylabel_left=None, title_left=None,
             xlabel_right=None, ylabel_right=None, title_right=None,
             curve_color="black", iqr_color="steelblue",
             vline_color="firebrick", hist_color="steelblue",
             n_bins=100, panel_labels=True):
        """
        Two-panel diagnostic: k-distance elbow (left) and Jaccard distribution (right).

        Parameters
        ----------
        axes : array-like of matplotlib.axes.Axes or None
            Two axes [left, right]. If None, creates a new figure.
        figsize : tuple, default=(13, 5)
            Figure size if creating new axes.
        xlabel_left : str or None
            Left panel x-label. Default: '$N_{Neighbors}$ [k]'.
        ylabel_left : str or None
            Left panel y-label. Default: '{Aggregation} distance to k-th neighbor'.
        title_left : str or None
            Left panel title. Default: None.
        xlabel_right : str or None
            Right panel x-label. Default: 'Jaccard similarity'.
        ylabel_right : str or None
            Right panel y-label. Default: '$N_{Edges}$'.
        title_right : str or None
            Right panel title. Default: None.
        curve_color : str, default='black'
            Color for the aggregation curve.
        iqr_color : str, default='steelblue'
            Color for the IQR fill.
        vline_color : str, default='firebrick'
            Color for the knee vertical line.
        hist_color : str, default='steelblue'
            Color for the Jaccard histogram.
        n_bins : int, default=100
            Number of histogram bins.
        panel_labels : bool, default=True
            Whether to add A/B panel labels.

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if self.k_distance_curve_ is None:
            self._build_k_distance_curve()

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax_left, ax_right = axes[0], axes[1]

        # ── Panel A: K-distance elbow ──
        k_vals = self.k_distance_curve_.index.values

        ax_left.plot(
            k_vals, self.k_distance_curve_.values,
            color=curve_color, linewidth=2,
            label=self.aggregation.capitalize(),
        )
        ax_left.fill_between(
            k_vals, self.k_distance_q25_, self.k_distance_q75_,
            alpha=0.2, color=iqr_color, label="IQR",
        )

        if self.k_ is not None:
            ax_left.axvline(
                self.k_, color=vline_color, linestyle="--",
                linewidth=1.5, label=f"Knee (k={self.k_})",
            )

        if xlabel_left is None:
            xlabel_left = "$N_{Neighbors}$ [k]"
        if ylabel_left is None:
            ylabel_left = f"{self.aggregation.capitalize()} distance to k-th neighbor"

        ax_left.set_xlabel(xlabel_left)
        ax_left.set_ylabel(ylabel_left)
        if title_left is not None:
            ax_left.set_title(title_left)
        ax_left.legend(frameon=False)

        # ── Panel B: Jaccard distribution ──
        if self.snn_adjacency_ is not None:
            A_upper = sps.triu(self.snn_adjacency_, k=1)
            ax_right.hist(A_upper.data, bins=n_bins, color=hist_color, edgecolor="none")
            n_edges = A_upper.nnz
            ax_right.text(
                0.95, 0.95,
                f"Edges at k={self.k_}: {n_edges:,}",
                transform=ax_right.transAxes,
                ha="right", va="top", fontsize=10,
            )

        if xlabel_right is None:
            xlabel_right = "Jaccard similarity"
        if ylabel_right is None:
            ylabel_right = "$N_{Edges}$"

        ax_right.set_xlabel(xlabel_right)
        ax_right.set_ylabel(ylabel_right)
        if title_right is not None:
            ax_right.set_title(title_right)

        # ── Formatting ──
        if panel_labels:
            for ax, label in zip(axes, ["A", "B"]):
                ax.text(
                    -0.12, 1.05, label,
                    transform=ax.transAxes,
                    fontsize=16, fontweight="bold",
                )

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].get_figure().tight_layout()
        return axes
 
    def __repr__(self):
        k_str = f"k={self.k_}" if self.k_ is not None else f"k={self.k} (unfitted)"
        return (
            f"SharedNearestNeighborsGraph(n_samples={self.n_samples_}, "
            f"max_k={self.max_k_}, {k_str}, aggregation='{self.aggregation}')"
        )

# ══════════════════════════════════════════════════════════════════════════════
# FaissKNNClassifier and FaissKNNTransformer Wrapper from DESlib
# ══════════════════════════════════════════════════════════════════════════════
#
# Strategy: Composition + direct FAISS index access.
#   - deslib is used ONLY during fit() to build the FAISS index
#   - All queries go through self._index.search() directly
#   - This avoids deslib's kneighbors/predict triggering sklearn ≥1.6 tag checks

class FaissKNNClassifier(BaseEstimator):
    """
    Sklearn-compatible kNN classifier using FAISS via deslib.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors.
    n_jobs : int or None, default=None
        Number of parallel jobs. If -1, uses all cores.
    algorithm : {'brute', 'voronoi', 'hierarchical'}, default='brute'
        - 'brute': IndexFlatL2 (exact search)
        - 'voronoi': IndexIVFFlat (faster inference, slower training)
        - 'hierarchical': IndexHNSWFlat (fast + accurate, higher memory)
    n_cells : int, default=100
        Number of voronoi cells. Only used when algorithm='voronoi'.
    n_probes : int, default=1
        Number of cells visited during search. Only used when algorithm='voronoi'.
    """

    def __init__(self, n_neighbors=5, n_jobs=None, algorithm="brute",
                 n_cells=100, n_probes=1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.n_cells = n_cells
        self.n_probes = n_probes

    def _build_model(self):
        """Instantiate the deslib model with current parameters."""
        return _FaissKNNClassifier(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            algorithm=self.algorithm,
            n_cells=self.n_cells,
            n_probes=self.n_probes,
        )

    def fit(self, X, y):
        _check_deslib()
        X = check_array(X, accept_sparse=False, dtype=[np.float32, np.float64])
        y = np.asarray(y)

        model = self._build_model()
        model.fit(X, y)
        self._index = model.index_

        self.n_samples_fit_ = X.shape[0]
        self.classes_ = np.unique(y)
        self.y_fit_ = y
        return self

    def kneighbors(self, X, n_neighbors=None):
        """
        Find k-nearest neighbors.

        Returns
        -------
        distances : np.ndarray of shape (n_samples, n_neighbors)
            Squared L2 distances (FAISS convention).
        indices : np.ndarray of shape (n_samples, n_neighbors)
        """
        check_is_fitted(self, ["_index"])
        X = check_array(X, accept_sparse=False, dtype=[np.float32, np.float64])
        X_query = np.ascontiguousarray(X, dtype=np.float32)
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        return self._index.search(X_query, k)

    def predict(self, X):
        """Predict class labels via majority vote of k-nearest neighbors."""
        check_is_fitted(self, ["_index", "classes_", "y_fit_"])
        _, indices = self.kneighbors(X)
        neighbor_labels = self.y_fit_[indices]
        preds = np.array([
            np.bincount(row, minlength=len(self.classes_)).argmax()
            for row in neighbor_labels
        ])
        return self.classes_[preds]


class FaissKNNTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible kNN transformer using FAISS via deslib.

    Outputs a sparse distance matrix matching sklearn's KNeighborsTransformer.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors.
    n_jobs : int or None, default=None
        Number of parallel jobs. If -1, uses all cores.
    algorithm : {'brute', 'voronoi', 'hierarchical'}, default='brute'
        - 'brute': IndexFlatL2 (exact search)
        - 'voronoi': IndexIVFFlat (faster inference, slower training)
        - 'hierarchical': IndexHNSWFlat (fast + accurate, higher memory)
    n_cells : int, default=100
        Number of voronoi cells. Only used when algorithm='voronoi'.
    n_probes : int, default=1
        Number of cells visited during search. Only used when algorithm='voronoi'.
    """

    def __init__(self, n_neighbors=5, n_jobs=None, algorithm="brute",
                 n_cells=100, n_probes=1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.n_cells = n_cells
        self.n_probes = n_probes

    def _build_model(self):
        """Instantiate the deslib model with current parameters."""
        return _FaissKNNClassifier(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            algorithm=self.algorithm,
            n_cells=self.n_cells,
            n_probes=self.n_probes,
        )

    def fit(self, X, y=None):
        _check_deslib()
        X = check_array(X, accept_sparse=False, dtype=[np.float32, np.float64])

        # deslib requires y; mock it for unsupervised use
        if y is None:
            y = np.zeros(X.shape[0], dtype=int)

        model = self._build_model()
        model.fit(X, y)
        self._index = model.index_

        self.n_samples_fit_ = X.shape[0]
        return self

    def transform(self, X):
        """Return sparse csr_matrix of true L2 distances to k-nearest neighbors."""
        check_is_fitted(self, ["_index"])
        X = check_array(X, accept_sparse=False, dtype=[np.float32, np.float64])
        X_query = np.ascontiguousarray(X, dtype=np.float32)

        sq_distances, indices = self._index.search(X_query, self.n_neighbors)

        # FAISS returns squared L2; convert to true L2
        distances = np.sqrt(np.maximum(sq_distances, 0.0))

        n_query = X.shape[0]
        rows = np.repeat(np.arange(n_query), self.n_neighbors)
        cols = indices.ravel()
        data = distances.ravel()

        # Filter FAISS -1 padding (approximate search edge case)
        valid = cols != -1
        rows, cols, data = rows[valid], cols[valid], data[valid]

        return sps.csr_matrix(
            (data, (rows, cols)),
            shape=(n_query, self.n_samples_fit_),
        )

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ══════════════════════════════════════════════════════════════════════════════
# EnsembleKNeighborsClassifier
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """
    KNN classifier that ensembles predictions across multiple k values.

    Computes a single cosine similarity matrix (via dot product on
    L2-normalized embeddings) and a single argsort, then slices the
    top-k neighbors for each k value. Class probabilities are averaged
    across all k values to produce the final prediction.

    Parameters
    ----------
    k_values : list of int
        The k values to ensemble over (e.g., [3, 5, 7, 11]).
    metric : str, default="cosine"
        Similarity metric. Only "cosine" is currently supported.
    weights : {"distance", "uniform"}, default="distance"
        How to weight neighbors within each k.
        - "distance": weight each neighbor by its cosine similarity.
        - "uniform": all neighbors contribute equally.

    Attributes
    ----------
    X_train_ : ndarray of shape (n_samples_train, n_features)
        Training embeddings (assumed L2-normalized).
    y_train_ : ndarray of shape (n_samples_train,)
        Training labels.
    classes_ : ndarray
        Unique class labels sorted in ascending order.
    class_to_index_ : dict
        Mapping from class label to column index in probability vectors.
    """

    def __init__(self, k_values, metric="cosine", weights="distance"):
        self.k_values = k_values
        self.metric = metric
        self.weights = weights

    def _validate_params(self):
        if self.metric != "cosine":
            raise NotImplementedError(
                f"Only metric='cosine' is supported, got '{self.metric}'"
            )
        if self.weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got '{self.weights}'"
            )
        k_arr = np.asarray(self.k_values)
        if k_arr.ndim != 1 or len(k_arr) == 0:
            raise ValueError("k_values must be a non-empty 1D list of integers")
        if not np.issubdtype(k_arr.dtype, np.integer) or np.any(k_arr <= 0):
            raise ValueError("All k_values must be positive integers")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError("k_values must not contain duplicates")

    def fit(self, X, y):
        """
        Store training data and derive class metadata.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training embeddings (assumed L2-normalized).
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self
        """
        self._validate_params()
        X, y = check_X_y(X, y)

        if max(self.k_values) > X.shape[0]:
            raise ValueError(
                f"max(k_values)={max(self.k_values)} exceeds "
                f"n_samples_train={X.shape[0]}"
            )

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}

        logger.info(
            f"EnsembleKNeighborsClassifier.fit | "
            f"n_samples={X.shape[0]}, n_features={X.shape[1]}, "
            f"n_classes={len(self.classes_)}, k_values={self.k_values}"
        )
        return self

    def predict_proba_per_k(self, X):
        """
        Compute class probabilities for each k value independently.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query embeddings (assumed L2-normalized).

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes, n_k_values)
            Class probability vectors for each sample and each k.
        """
        check_is_fitted(self)
        X = check_array(X)

        S = X @ self.X_train_.T
        max_k = max(self.k_values)
        # O(n) partition to find top-max_k, then O(max_k log max_k) sort within
        top_k_unsorted = np.argpartition(-S, max_k, axis=1)[:, :max_k]
        top_k_sims_unsorted = np.take_along_axis(S, top_k_unsorted, axis=1)
        sort_within = np.argsort(-top_k_sims_unsorted, axis=1)
        neighbor_indices = np.take_along_axis(top_k_unsorted, sort_within, axis=1)
        sorted_similarities = np.take_along_axis(top_k_sims_unsorted, sort_within, axis=1)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_k = len(self.k_values)
        label_indices = np.array([self.class_to_index_[c] for c in self.y_train_])

        proba = np.zeros((n_samples, n_classes, n_k), dtype=np.float64)

        for ki, k in enumerate(self.k_values):
            top_k_indices = neighbor_indices[:, :k]
            top_k_labels = label_indices[top_k_indices]

            if self.weights == "uniform":
                for c_idx in range(n_classes):
                    proba[:, c_idx, ki] = np.mean(top_k_labels == c_idx, axis=1)
            else:
                top_k_sims = sorted_similarities[:, :k]
                for c_idx in range(n_classes):
                    mask = top_k_labels == c_idx
                    proba[:, c_idx, ki] = np.sum(top_k_sims * mask, axis=1)
                row_sums = proba[:, :, ki].sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                proba[:, :, ki] /= row_sums

        return proba

    def predict_proba(self, X):
        """
        Compute class probabilities averaged across all k values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query embeddings (assumed L2-normalized).

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Mean class probability vectors.
        """
        return self.predict_proba_per_k(X).mean(axis=2)

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query embeddings (assumed L2-normalized).

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
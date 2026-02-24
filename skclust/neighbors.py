# -*- coding: utf-8 -*-
# skclust/kneighbors.py

from __future__ import annotations
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from typing import Union
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.base import clone, BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.neighbors import KNeighborsTransformer
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted, check_array
from tqdm import tqdm

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


def _compute_pairwise_similarities_chunked(X, chunk_size=10000, show_progress=True):
    """
    Compute upper-triangular pairwise cosine similarities in memory-efficient chunks.
    
    Returns flattened array of all unique pairwise similarities (excluding self).
    """
    n = X.shape[0]
    n_pairs = n * (n - 1) // 2
    
    if n_pairs == 0:
        return np.array([], dtype=np.float32)
    
    similarities = []
    
    iterator = range(0, n, chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing pairwise similarities", leave=False)
    
    for i in iterator:
        i_end = min(i + chunk_size, n)
        chunk = X[i:i_end]
        
        # Compute similarities to all samples from i onwards
        sims_block = chunk @ X[i:].T  # shape: (chunk_size, n - i)
        
        # Extract upper triangular for each row in chunk
        for local_row, global_row in enumerate(range(i, i_end)):
            start_col = local_row + 1
            row_sims = sims_block[local_row, start_col:]
            similarities.append(row_sims)
    
    return np.concatenate(similarities).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# KNeighborsCosineSimilarity
# ══════════════════════════════════════════════════════════════════════════════

class KNeighborsCosineSimilarity(BaseEstimator, TransformerMixin):
    """
    K-Nearest Neighbors using cosine similarity.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to find.
    mode : {'exact', 'ivf', 'pq'}, default='exact'
        Search strategy.
    backend : {'auto', 'faiss', 'sklearn'}, default='auto'
        Which library to use. 'auto' prefers FAISS, falls back to sklearn.
    n_voronoi_cells : int or 'auto', default='auto'
        Number of IVF cells. If 'auto', uses sqrt(n_samples).
    n_probes : int, default=1
        Number of cells to search in IVF.
    n_subvectors : int or None, default=None
        Number of sub-vectors for PQ. If None, uses d//16.
    n_bits : int, default=8
        Bits per sub-vector for PQ.

    Attributes
    ----------
    backend_ : str
    index_ : faiss.Index or None
    similarities_ : np.ndarray, shape (n_samples_fit, n_neighbors)
    indices_ : np.ndarray, shape (n_samples_fit, n_neighbors)
    """

    def __init__(
        self,
        n_neighbors,
        mode="exact",
        backend="auto",
        n_voronoi_cells="auto",
        n_probes=1,
        n_subvectors=None,
        n_bits=8,
    ):
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.backend = backend
        self.n_voronoi_cells = n_voronoi_cells
        self.n_probes = n_probes
        self.n_subvectors = n_subvectors
        self.n_bits = n_bits

    def fit(self, X, y=None):
        """
        Fit the k-NN model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            L2-normalized training data.
        y : Ignored
        """
        X_arr, row_index = _to_numpy_with_index(X)
        X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)

        if row_index is not None:
            self.index_labels_ = row_index

        self.n_samples_fit_ = X_arr.shape[0]
        self.n_features_in_ = X_arr.shape[1]
        self.backend_ = _determine_backend(self.backend)

        if self.backend_ == "faiss":
            self.index_ = _build_faiss_index(
                X_arr,
                mode=self.mode,
                n_voronoi_cells=self.n_voronoi_cells,
                n_probes=self.n_probes,
                n_subvectors=self.n_subvectors,
                n_bits=self.n_bits,
            )
            self.X_fit_ = None
        else:
            if self.mode != "exact":
                warnings.warn(
                    f"sklearn backend only supports exact search, ignoring mode='{self.mode}'.",
                    UserWarning,
                )
            self.X_fit_ = X_arr
            self.index_ = None

        self.similarities_, self.indices_ = self.transform(X_arr)
        return self

    def transform(self, X):
        """
        Find k-nearest neighbors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            L2-normalized query vectors.

        Returns
        -------
        similarities : np.ndarray, shape (n_samples, n_neighbors)
        indices : np.ndarray, shape (n_samples, n_neighbors)
        """
        check_is_fitted(self)
        X_arr, _ = _to_numpy_with_index(X)
        X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}.")
        if self.n_neighbors > self.n_samples_fit_:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) > n_samples ({self.n_samples_fit_})."
            )

        return _search_index(
            self.index_, X_arr, self.n_neighbors, self.backend_,
            X_fit=self.X_fit_,
        )

    def fit_transform(self, X, y=None):
        """Fit and return neighbors for training data."""
        self.fit(X, y)
        return self.similarities_, self.indices_

    def to_igraph(self, index="auto", include_self=False):
        """
        Convert fitted k-NN results to igraph.

        Parameters
        ----------
        index : array-like or 'auto'
            Node labels. 'auto' uses DataFrame index if available, else integers.
        include_self : bool, default=False

        Returns
        -------
        ig.Graph
        """
        check_is_fitted(self, ["similarities_", "indices_"])

        if index == "auto":
            index = getattr(self, "index_labels_", None)
        elif isinstance(index, pd.Index):
            index = list(index)

        return kneighbors_to_igraph(
            self.similarities_,
            self.indices_,
            index=index,
            include_self=include_self,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CosineSimilarityClassifier
# ══════════════════════════════════════════════════════════════════════════════

class CosineSimilarityClassifier(BaseEstimator, ClassifierMixin):
    """
    Cosine similarity-based nearest neighbor classifier with per-class thresholds.

    Uses confidence intervals on within- and between-class cosine similarities
    to derive per-class thresholds for assignment. Points below threshold
    are assigned label -1 (unknown).

    Parameters
    ----------
    confidence_interval : float
        CI percentage (e.g., 95 → uses p2.5/p97.5). Default is 95.
    backend : {'auto', 'faiss', 'sklearn'}
        Search backend. Default is 'auto' (prefers FAISS).
    max_samples_per_class : int or None
        Maximum samples per class for within-class similarity estimation.
        None uses all samples. Default is None.
    max_samples_between_classes : int or None
        Maximum between-class pairs to keep per class (most similar ones).
        None keeps all pairs. Default is None.
    chunk_size : int
        Chunk size for pairwise similarity computation. Default is 10000.
    show_progress : bool
        Whether to show tqdm progress bars. Default is True.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels.
    class_to_indices_ : dict
        Mapping from class label to array of training indices.
    index_labels_ : pd.Index or None
        Row labels from training DataFrame (if provided).
    within_class_similarities_ : dict
        Mapping from class label to array of within-class similarities.
    between_class_similarities_ : dict
        Mapping from class label to array of between-class similarities.
    within_ci_ : dict
        Mapping from class label to (lower, upper) CI bounds.
    between_ci_ : dict
        Mapping from class label to (lower, upper) CI bounds.
    cutoffs_ : dict
        Mapping from class label to similarity cutoff threshold.
    n_neighbors_query_ : int
        Auto-computed number of neighbors for between-class queries.
    """

    def __init__(
        self,
        confidence_interval=95,
        backend="auto",
        max_samples_per_class=None,
        max_samples_between_classes=None,
        chunk_size=10000,
        show_progress=True,
    ):
        self.confidence_interval = confidence_interval
        self.backend = backend
        self.max_samples_per_class = max_samples_per_class
        self.max_samples_between_classes = max_samples_between_classes
        self.chunk_size = chunk_size
        self.show_progress = show_progress

    def fit(self, X, y):
        """
        Compute per-class within- and between-class similarity distributions.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            L2-normalized embeddings, shape (n_samples, n_features).
        y : np.ndarray or pd.Series
            Class labels.

        Returns
        -------
        self
        """
        X_arr, row_index = _to_numpy_with_index(X)
        X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)
        y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        self.X_fit_ = X_arr
        self.y_fit_ = y_arr
        self.n_features_in_ = X_arr.shape[1]
        self.backend_ = _determine_backend(self.backend)
        
        # Store training index labels for search() output
        self.index_labels_ = row_index

        # Build index (always exact for classifier)
        if self.backend_ == "faiss":
            self.index_ = _build_faiss_index(X_arr, mode="exact")
        else:
            self.index_ = None

        # Build class-to-indices mapping
        self.class_to_indices_ = {}
        for cls in self.classes_:
            self.class_to_indices_[cls] = np.where(y_arr == cls)[0]

        # Auto-compute n_neighbors for between-class queries
        # Must exceed largest class size to guarantee cross-class neighbors
        max_class_size = max(len(indices) for indices in self.class_to_indices_.values())
        self.n_neighbors_query_ = min(max_class_size + 100, len(X_arr))

        # Compute per-class distributions
        self.within_class_similarities_ = {}
        self.between_class_similarities_ = {}
        self.within_ci_ = {}
        self.between_ci_ = {}
        self.cutoffs_ = {}

        lower_pct = (100 - self.confidence_interval) / 2
        upper_pct = 100 - lower_pct

        class_iterator = self.classes_
        if self.show_progress:
            class_iterator = tqdm(self.classes_, desc="Processing classes")

        for cls in class_iterator:
            indices = self.class_to_indices_[cls]
            n_class = len(indices)

            # Warn if class is very large and no sampling
            if self.max_samples_per_class is None and n_class > 5000:
                n_pairs = n_class * (n_class - 1) // 2
                warnings.warn(
                    f"Class '{cls}' has {n_class} samples ({n_pairs:,} pairs). "
                    f"Consider setting max_samples_per_class to reduce memory usage.",
                    UserWarning,
                )

            # Sample if needed
            if self.max_samples_per_class is not None and n_class > self.max_samples_per_class:
                rng = np.random.default_rng(42)
                sample_indices = rng.choice(indices, self.max_samples_per_class, replace=False)
            else:
                sample_indices = indices

            X_class = X_arr[sample_indices]

            # Within-class similarities (chunked pairwise)
            within_sims = _compute_pairwise_similarities_chunked(
                X_class,
                chunk_size=self.chunk_size,
                show_progress=False,
            )
            self.within_class_similarities_[cls] = within_sims

            # Between-class similarities (query-based, most similar non-class neighbors)
            between_sims = self._compute_between_class_similarities(
                X_class, cls, X_arr, y_arr
            )
            self.between_class_similarities_[cls] = between_sims

            # Compute CIs
            if len(within_sims) > 0:
                self.within_ci_[cls] = (
                    np.percentile(within_sims, lower_pct),
                    np.percentile(within_sims, upper_pct),
                )
            else:
                self.within_ci_[cls] = (0.0, 1.0)
                warnings.warn(
                    f"Class '{cls}' has insufficient samples for within-class CI.",
                    UserWarning,
                )

            if len(between_sims) > 0:
                self.between_ci_[cls] = (
                    np.percentile(between_sims, lower_pct),
                    np.percentile(between_sims, upper_pct),
                )
            else:
                self.between_ci_[cls] = (0.0, 0.0)

            # Cutoff: lower bound of within-class CI
            self.cutoffs_[cls] = self.within_ci_[cls][0]

            # Warn if distributions overlap
            if self.within_ci_[cls][0] < self.between_ci_[cls][1]:
                warnings.warn(
                    f"Class '{cls}': within-class lower bound ({self.within_ci_[cls][0]:.3f}) < "
                    f"between-class upper bound ({self.between_ci_[cls][1]:.3f}). "
                    "Threshold assignment may be ambiguous.",
                    UserWarning,
                )

        return self

    def _compute_between_class_similarities(self, X_class, cls, X_all, y_all):
        """
        Compute between-class similarities for a given class.
        
        Queries each sample in X_class against the full index, filters to
        neighbors from other classes, and keeps the most similar pairs.
        """
        k = self.n_neighbors_query_
        
        sims, idxs = _search_index(
            self.index_, X_class, k, self.backend_, X_fit=X_all
        )

        # Collect between-class similarities
        between_sims = []
        for i in range(len(X_class)):
            for j in range(k):
                neighbor_idx = idxs[i, j]
                if y_all[neighbor_idx] != cls:
                    between_sims.append(sims[i, j])

        between_sims = np.array(between_sims, dtype=np.float32)

        # Keep only top most similar (these are the "confusable" pairs)
        if self.max_samples_between_classes is not None and len(between_sims) > self.max_samples_between_classes:
            # Partial sort to get top-k
            partition_idx = len(between_sims) - self.max_samples_between_classes
            between_sims = np.partition(between_sims, partition_idx)[partition_idx:]

        return between_sims

    def predict(self, X, k=10):
        """
        Predict class labels for query points.

        Iterates through k nearest neighbors until finding one whose similarity
        exceeds that class's cutoff threshold. Returns -1 if no neighbor passes.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            L2-normalized query embeddings.
        k : int
            Maximum neighbors to consider. Default is 10.

        Returns
        -------
        np.ndarray or pd.Series
            Predicted class labels. -1 if no confident assignment.
            Returns pd.Series if input was pd.DataFrame.
        """
        check_is_fitted(self)
        X_arr, row_index = _to_numpy_with_index(X)
        X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)

        k = min(k, len(self.X_fit_))
        sims, idxs = _search_index(
            self.index_, X_arr, k, self.backend_, X_fit=self.X_fit_
        )

        results = np.empty(len(X_arr), dtype=object)

        for i in range(len(X_arr)):
            assigned = -1
            for j in range(k):
                neighbor_idx = idxs[i, j]
                neighbor_class = self.y_fit_[neighbor_idx]
                neighbor_sim = sims[i, j]
                cutoff = self.cutoffs_[neighbor_class]

                if neighbor_sim >= cutoff:
                    assigned = neighbor_class
                    break

            results[i] = assigned

        # Return Series if input was DataFrame
        if row_index is not None:
            return pd.Series(results, index=row_index)
        return results

    def search(
        self,
        X,
        k=10,
        filter_by_cutoff: Union[bool, float, Mapping] = False,
    ):
        """
        Return top-k nearest neighbors for each query point.

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or pd.Series
            L2-normalized query embeddings.
        k : int
            Number of nearest neighbors. Default is 10.
        filter_by_cutoff : bool, float, or Mapping
            How to filter results:
            - False: no filtering, return all k neighbors
            - True: filter using stored per-class cutoffs
            - float: use this value as cutoff for all classes
            - Mapping (dict or pd.Series): per-class cutoffs (must contain all classes)

        Returns
        -------
        OrderedDict
            Keys are row index/name (or integer position).
            Values are lists of (neighbor_id, class_label, similarity) tuples,
            sorted by similarity descending. neighbor_id uses training index
            labels if available, otherwise integer indices.
        """
        check_is_fitted(self)
        X_arr, row_index = _to_numpy_with_index(X)
        X_arr = check_array(X_arr, dtype=np.float32, ensure_2d=True)
        k = min(k, len(self.X_fit_))

        # Resolve cutoffs
        if filter_by_cutoff is False:
            cutoffs = None
        elif filter_by_cutoff is True:
            cutoffs = self.cutoffs_
        elif isinstance(filter_by_cutoff, (int, float)):
            cutoffs = {cls: float(filter_by_cutoff) for cls in self.classes_}
        elif isinstance(filter_by_cutoff, (Mapping, pd.Series)):
            missing = set(self.classes_) - set(filter_by_cutoff.keys())
            if missing:
                raise ValueError(f"Missing cutoffs for classes: {missing}")
            cutoffs = dict(filter_by_cutoff)
        else:
            raise TypeError(
                f"filter_by_cutoff must be bool, float, or Mapping, got {type(filter_by_cutoff)}"
            )

        sims, idxs = _search_index(
            self.index_, X_arr, k, self.backend_, X_fit=self.X_fit_
        )

        results = OrderedDict()
        for qi in range(len(X_arr)):
            row_key = row_index[qi] if row_index is not None else qi
            hits = []

            for j in range(k):
                neighbor_idx = int(idxs[qi, j])
                neighbor_class = self.y_fit_[neighbor_idx]
                neighbor_sim = float(sims[qi, j])

                # Apply filtering if cutoffs specified
                if cutoffs is not None:
                    if neighbor_sim < cutoffs[neighbor_class]:
                        continue

                # Use training index labels if available
                if self.index_labels_ is not None:
                    neighbor_id = self.index_labels_[neighbor_idx]
                else:
                    neighbor_id = neighbor_idx

                hits.append((neighbor_id, neighbor_class, neighbor_sim))

            results[row_key] = hits

        return results

    def get_cutoff(self, cls=None):
        """
        Get cutoff threshold(s).

        Parameters
        ----------
        cls : class label or None
            If provided, returns cutoff for that class.
            If None, returns dict of all cutoffs.

        Returns
        -------
        float or dict
        """
        check_is_fitted(self)
        if cls is None:
            return dict(self.cutoffs_)
        return self.cutoffs_[cls]

    def summary(self):
        """
        Return a DataFrame summarizing per-class statistics.

        Returns
        -------
        pd.DataFrame
        """
        check_is_fitted(self)

        records = []
        for cls in self.classes_:
            within_sims = self.within_class_similarities_[cls]
            between_sims = self.between_class_similarities_[cls]

            records.append({
                "class": cls,
                "n_samples": len(self.class_to_indices_[cls]),
                "n_within_pairs": len(within_sims),
                "n_between_pairs": len(between_sims),
                "within_mean": within_sims.mean() if len(within_sims) > 0 else np.nan,
                "within_ci_lower": self.within_ci_[cls][0],
                "within_ci_upper": self.within_ci_[cls][1],
                "between_mean": between_sims.mean() if len(between_sims) > 0 else np.nan,
                "between_ci_lower": self.between_ci_[cls][0],
                "between_ci_upper": self.between_ci_[cls][1],
                "cutoff": self.cutoffs_[cls],
            })

        return pd.DataFrame(records).set_index("class")


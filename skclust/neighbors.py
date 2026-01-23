# -*- coding: utf-8 -*-
# skclust/kneighbors.py

from __future__ import annotations
import warnings
from typing import Union
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsTransformer
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted, check_array

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

class KNeighborsCosineSimilarity(BaseEstimator, TransformerMixin):
    """
    K-Nearest Neighbors using cosine similarity.
    
    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to find
    mode : {'exact', 'ivf', 'pq'}, default='exact'
        Search strategy:
        - 'exact': Brute force exact search
        - 'ivf': Inverted file index (approximate)
        - 'pq': Product quantization (compressed, approximate)
    backend : {'auto', 'faiss', 'sklearn'}, default='auto'
        Which library to use. 'auto' tries FAISS, falls back to sklearn
    n_voronoi_cells : int or 'auto', default='auto'
        Number of IVF cells. If 'auto', uses sqrt(n_samples)
    n_probes : int, default=1
        Number of cells to search in IVF (FAISS default is 1)
    n_subvectors : int or None, default=None
        Number of sub-vectors for PQ. If None, uses d//16
    n_bits : int, default=8
        Bits per sub-vector for PQ
    
    Attributes
    ----------
    backend_ : str
        Actual backend used ('faiss' or 'sklearn')
    index_ : faiss index or None
        Fitted FAISS index (None if using sklearn)
    similarities_ : np.ndarray, shape (n_samples_fit, n_neighbors)
        Cosine similarities to k nearest neighbors (higher = more similar)
    indices_ : np.ndarray, shape (n_samples_fit, n_neighbors)
        Indices of k nearest neighbors
    """
    
    def __init__(
        self,
        n_neighbors,
        mode='exact',
        backend='auto',
        n_voronoi_cells='auto',
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
    
    def _determine_backend(self):
        """Determine which backend to use."""
        if self.backend == 'sklearn':
            return 'sklearn'
        elif self.backend == 'faiss':
            try:
                import faiss
                return 'faiss'
            except ImportError:
                raise ImportError("FAISS not available")
        else:  # auto
            try:
                import faiss
                return 'faiss'
            except ImportError:
                warnings.warn("FAISS not available, falling back to sklearn", UserWarning)
                return 'sklearn'
    
    def fit(self, X, y=None):
        """
        Fit the k-NN model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (must be L2-normalized for cosine similarity)
        y : Ignored
        
        Returns
        -------
        self : object
        """
        if isinstance(X, pd.DataFrame):
            self.index_labels_ = X.index

        X = check_array(X, dtype=np.float32, ensure_2d=True)
        
        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.backend_ = self._determine_backend()
        
        if self.backend_ == 'faiss':
            self._fit_faiss(X)
        else:
            self._fit_sklearn(X)
        
        # Store the training data neighbors
        self.similarities_, self.indices_ = self.transform(X)
        
        return self
    
    def _fit_faiss(self, X):
        """Fit using FAISS backend."""
        import faiss
        
        d = self.n_features_in_
        
        if self.mode == 'exact':
            self.index_ = faiss.IndexFlatIP(d)
            self.index_.add(X)
        
        elif self.mode == 'ivf':
            # Determine n_voronoi_cells
            if self.n_voronoi_cells == 'auto':
                nlist = int(np.sqrt(self.n_samples_fit_))
            else:
                nlist = self.n_voronoi_cells
            
            quantizer = faiss.IndexFlatIP(d)
            self.index_ = faiss.IndexIVFFlat(quantizer, d, nlist)
            self.index_.train(X)
            self.index_.add(X)
            self.index_.nprobe = self.n_probes
        
        elif self.mode == 'pq':
            # Determine n_subvectors
            if self.n_subvectors is None:
                m = d // 16
                while d % m != 0 and m > 1:
                    m -= 1
                if m == 1:
                    raise ValueError(
                        f"Cannot determine n_subvectors for dimension {d}. "
                        f"Please specify n_subvectors that divides {d} evenly."
                    )
            else:
                m = self.n_subvectors
                if d % m != 0:
                    raise ValueError(f"n_subvectors ({m}) must divide dimension ({d}) evenly")
            
            self.index_ = faiss.IndexPQ(d, m, self.n_bits)
            self.index_.train(X)
            self.index_.add(X)
        
        else:
            raise ValueError(f"mode must be 'exact', 'ivf', or 'pq', got '{self.mode}'")
    
    def _fit_sklearn(self, X):
        """Fit using sklearn backend."""
        if self.mode != 'exact':
            warnings.warn(
                f"sklearn backend only supports exact search, ignoring mode='{self.mode}'",
                UserWarning
            )
        self.X_fit_ = X
        self.index_ = None
    
    def transform(self, X):
        """
        Find k-nearest neighbors.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Query vectors (must be L2-normalized)
        
        Returns
        -------
        similarities : np.ndarray, shape (n_samples, n_neighbors)
            Cosine similarities to k nearest neighbors (higher = more similar)
        indices : np.ndarray, shape (n_samples, n_neighbors)
            Indices of k nearest neighbors
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32, ensure_2d=True)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        
        if self.n_neighbors > self.n_samples_fit_:
            raise ValueError(f"n_neighbors ({self.n_neighbors}) > n_samples ({self.n_samples_fit_})")
        
        if self.backend_ == 'faiss':
            similarities, indices = self.index_.search(X, self.n_neighbors)
        else:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
            nn.fit(self.X_fit_)
            distances, indices = nn.kneighbors(X)
            similarities = 1 - distances  # Convert distance to similarity
        
        return similarities, indices
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def to_igraph(self, index="auto", include_self=False):
        """
        Convert fitted k-NN results to igraph.
        
        Parameters
        ----------
        index : array-like or None, default=None
            Node labels. If None, uses integers 0 to n-1
        include_self : bool, default=False
            Whether to include self-loops
        
        Returns
        -------
        ig.Graph
            Directed graph with edges weighted by cosine similarity
        """
        check_is_fitted(self, ['similarities_', 'indices_'])
        if index == "auto":
            if hasattr(self,"index_labels_"):
                index = self.index_labels_
            else:
                index = None
        return kneighbors_to_igraph(
            self.similarities_,
            self.indices_,
            index=index,
            include_self=include_self
        )
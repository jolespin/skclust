# -*- coding: utf-8 -*-
# skclust/graph.py

import numpy as np
import pandas as pd
import igraph as ig
from itertools import combinations
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm


def compute_membership_cooccurrence(
    df: pd.DataFrame,
    edge_type: str = "Edge",
    iteration_type: str = "Iteration"
) -> pd.DataFrame:
    """
    Compute pairwise cluster membership co-occurrence across iterations.
    
    For each pair of nodes, determines whether they belong to the same cluster
    in each iteration. Returns a boolean DataFrame where True indicates the
    node pair shares cluster membership in that iteration.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where rows are nodes and columns are iterations.
        Values are cluster/partition assignments.
        Index name will be preserved or set to 'Node' if None.
    edge_type : str, default="Edge"
        Name for the index of the output DataFrame (node pairs)
    iteration_type : str, default="Iteration"
        Name for the columns of the output DataFrame (iterations)
    
    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with shape (n_node_pairs, n_iterations) where
        n_node_pairs = n_nodes * (n_nodes - 1) / 2.
        Index contains frozensets of node pairs.
        Values are True if nodes share cluster membership, False otherwise.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create example partition data
    >>> df = pd.DataFrame({
    ...     0: [0, 0, 1, 1],
    ...     1: [0, 0, 1, 1],
    ...     2: [0, 1, 1, 0]
    ... }, index=['A', 'B', 'C', 'D'])
    >>> df.columns.name = 'Iteration'
    >>> df.index.name = 'Node'
    >>> 
    >>> # Compute co-occurrence
    >>> cooccur = compute_membership_cooccurrence(df)
    >>> 
    >>> # Nodes A and B are always together
    >>> print(cooccur.loc[frozenset(['A', 'B'])])
    >>> # Iteration
    >>> # 0    True
    >>> # 1    True
    >>> # 2    False
    >>> 
    >>> # Get consensus edges (100% co-occurrence)
    >>> consensus = cooccur.mean(axis=1)
    >>> perfect_pairs = consensus[consensus == 1.0].index
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Extract arrays and metadata
    X = df.values
    nodes = df.index.values
    iterations = df.columns
    n_nodes = len(nodes)
    n_iterations = len(iterations)
    
    # Pre-allocate boolean array for results
    n_pairs = (n_nodes * (n_nodes - 1)) // 2
    result = np.empty((n_pairs, n_iterations), dtype=bool)
    
    # Generate all node pairs once (upper triangle, no diagonal)
    pairs = np.array(list(combinations(range(n_nodes), 2)), dtype=np.int32)
    
    # Vectorized comparison across all iterations
    # For each iteration, check if pair[0] cluster == pair[1] cluster
    for i in range(n_iterations):
        col = X[:, i]
        result[:, i] = col[pairs[:, 0]] == col[pairs[:, 1]]
    
    # Create edge labels as frozensets for hashable, undirected pairs
    edge_labels = [frozenset([nodes[i], nodes[j]]) for i, j in pairs]
    
    # Construct output DataFrame
    return pd.DataFrame(
        data=result,
        index=pd.Index(edge_labels, name=edge_type),
        columns=pd.Index(iterations, name=iteration_type),
    )


def _leiden_worker(args):
    """
    Worker function for parallel Leiden execution.
    
    Must be at module level for pickling (multiprocessing requirement).
    Receives all arguments as tuple to work with Pool.map().
    """
    graph, weight, random_seed, leiden_kws_full, nodes_list = args
    
    try:
        from leidenalg import find_partition
    except ModuleNotFoundError:
        raise ImportError("Install leidenalg: pip install leidenalg")
    
    # Run Leiden
    partition = find_partition(
        graph, 
        weights=weight,
        seed=random_seed,
        **leiden_kws_full
    )
    
    # Convert to node->partition mapping
    node_to_partition = {}
    for partition_id, node_indices in enumerate(partition):
        for idx in node_indices:
            node_to_partition[nodes_list[idx]] = partition_id
            
    return node_to_partition


class ConsensusLeidenClustering(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for consensus Leiden clustering.
    
    Runs multiple iterations of Leiden with different random seeds in parallel,
    then returns only edges with consistent cluster membership across all iterations.
    
    Parameters
    ----------
    n_iter : int, default=100
        Number of Leiden iterations with different random seeds
    weight : str or None, default=None
        Edge weight attribute name in graph. If None, unweighted clustering is used.
    random_state : int, default=0
        Base random seed (actual seeds: random_state to random_state + n_iter - 1)
    partition_type : leidenalg partition class, default=None
        Leiden partition type to use. If None, uses RBConfigurationVertexPartition
        with resolution_parameter=1.0 (equivalent to modularity).
        Common options:
        - RBConfigurationVertexPartition: Reichardt-Bornholdt quality (recommended)
        - ModularityVertexPartition: Classic modularity optimization
        - CPMVertexPartition: Constant Potts Model for weighted graphs
        - SignificanceVertexPartition: Statistical significance-based
    resolution_parameter : float, default=1.0
        Resolution parameter for RBConfigurationVertexPartition.
        Only used if partition_type is None or RBConfigurationVertexPartition.
        - 1.0: Standard modularity
        - >1.0: Smaller, more clusters
        - <1.0: Larger, fewer clusters
    n_iterations : int, default=-1
        Number of iterations for Leiden convergence (-1 for auto convergence)
    n_jobs : int, default=1
        Number of parallel processes. 
        - 1: Sequential execution (no multiprocessing)
        - -1: Use all available CPUs
        - >1: Use specific number of processes
    verbose : bool, default=True
        Show progress bar
    leiden_kws : dict, optional
        Additional keyword arguments passed to leidenalg.find_partition.
        Note: resolution_parameter should be set via the resolution_parameter
        parameter rather than leiden_kws for proper sklearn compatibility.
        
    Attributes
    ----------
    partitions_ : pd.DataFrame
        Node assignments for each iteration (shape: n_nodes x n_iter)
    membership_matrix_ : pd.DataFrame
        Boolean matrix of edge co-membership across iterations (shape: n_edges x n_iter)
    consensus_edges_ : set of frozenset
        Edge pairs with 100% consistent cluster membership
    consensus_ratio_ : pd.Series
        Proportion of iterations each edge had consistent membership
    graph_ : ig.Graph
        Original input graph (stored for reference)
    consensus_graph_ : ig.Graph
        Subgraph containing only consensus edges
        
    Notes
    -----
    Multiprocessing uses 'spawn' context for cross-platform compatibility.
    Each process gets a copy of the graph, which is memory-intensive for large graphs.
    For very large graphs (>100k nodes), consider using n_jobs=1 or smaller n_iter.
    
    The leidenalg library is thread-safe but multiprocessing provides better
    performance since each process runs independently without GIL contention.
    
    Examples
    --------
    >>> import igraph as ig
    >>> 
    >>> # Create graph
    >>> graph = ig.Graph.Famous('Zachary')
    >>> graph.vs['name'] = [f'node_{i}' for i in range(graph.vcount())]
    >>> 
    >>> # Fit transformer with default RBConfigurationVertexPartition
    >>> leiden = ConsensusLeidenClustering(n_iter=100, n_jobs=-1, random_state=42)
    >>> leiden.fit(graph)
    >>> 
    >>> # Get consensus graph
    >>> consensus_graph = leiden.transform(graph)
    >>> 
    >>> # Find finer-grained clusters
    >>> leiden = ConsensusLeidenClustering(
    ...     n_iter=100,
    ...     resolution_parameter=1.5,
    ...     n_jobs=-1
    ... )
    >>> consensus_graph = leiden.fit_transform(graph)
    >>> 
    >>> # Use classic modularity
    >>> from leidenalg import ModularityVertexPartition
    >>> leiden = ConsensusLeidenClustering(
    ...     n_iter=100,
    ...     partition_type=ModularityVertexPartition,
    ...     n_jobs=-1
    ... )
    >>> consensus_graph = leiden.fit_transform(graph)
    """
    
    def __init__(
        self,
        n_iter: int = 100,
        weight: Optional[str] = None,
        random_state: int = 0,
        partition_type=None,
        resolution_parameter: float = 1.0,
        n_iterations: int = -1,
        n_jobs: int = 1,
        verbose: bool = True,
        leiden_kws: Optional[dict] = None,
    ):
        self.n_iter = n_iter
        self.weight = weight
        self.random_state = random_state
        self.partition_type = partition_type
        self.resolution_parameter = resolution_parameter
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.leiden_kws = leiden_kws or {}
    
    def fit(self, X, y=None):
        """
        Run N iterations of Leiden clustering in parallel.
        
        Parameters
        ----------
        X : ig.Graph
            Input graph with named vertices
        y : None
            Ignored, exists for sklearn compatibility
            
        Returns
        -------
        self
            Fitted transformer
        """
        # Validate graph
        if not isinstance(X, ig.Graph):
            raise TypeError("Graph must be igraph.Graph instance")
        if 'name' not in X.vs.attributes():
            raise ValueError("Graph vertices must have 'name' attribute")
        if self.weight is not None and self.weight not in X.es.attributes():
            raise ValueError(f"Weight attribute '{self.weight}' not found in graph edges")
        
        self.graph_ = X
        nodes_list = np.asarray(X.vs['name'])
        
        # Import and setup partition type
        try:
            from leidenalg import find_partition, RBConfigurationVertexPartition
        except ModuleNotFoundError:
            raise ImportError("Install leidenalg: pip install leidenalg")
        
        # Determine partition type
        partition_type = self.partition_type if self.partition_type is not None else RBConfigurationVertexPartition
        
        # Build leiden kwargs
        leiden_kws_full = {
            'partition_type': partition_type,
            'n_iterations': self.n_iterations,
            **self.leiden_kws
        }
        
        # Add resolution_parameter for RB partition if not already specified
        if (self.partition_type is None or partition_type is RBConfigurationVertexPartition):
            if 'resolution_parameter' not in self.leiden_kws:
                leiden_kws_full['resolution_parameter'] = self.resolution_parameter
        
        # Determine number of jobs
        n_jobs = cpu_count() if self.n_jobs == -1 else self.n_jobs
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or >= 1, got {self.n_jobs}")
        
        # Prepare worker arguments
        random_seeds = list(range(self.random_state, self.random_state + self.n_iter))
        weight_attr = self.weight if self.weight is not None else None
        worker_args = [
            (X, weight_attr, seed, leiden_kws_full, nodes_list)
            for seed in random_seeds
        ]
        
        # Run partitions
        if n_jobs == 1:
            # Sequential execution
            if self.verbose:
                partitions = [
                    _leiden_worker(args) 
                    for args in tqdm(worker_args, desc="Leiden clustering")
                ]
            else:
                partitions = [_leiden_worker(args) for args in worker_args]
        else:
            # Parallel execution
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            
            if self.verbose:
                with ctx.Pool(processes=n_jobs) as pool:
                    partitions = list(
                        tqdm(
                            pool.imap(_leiden_worker, worker_args),
                            total=len(worker_args),
                            desc="Leiden clustering"
                        )
                    )
            else:
                with ctx.Pool(processes=n_jobs) as pool:
                    partitions = pool.map(_leiden_worker, worker_args)
        
        # Convert to DataFrame
        self.partitions_ = pd.DataFrame(partitions).T
        self.partitions_.index.name = "Node"
        self.partitions_.columns.name = "Iteration"
        
        # Compute membership co-occurrence matrix
        self.membership_matrix_ = compute_membership_cooccurrence(self.partitions_)
        
        # Compute consensus metrics
        self.consensus_ratio_ = self.membership_matrix_.mean(axis=1)
        self.consensus_edges_ = set(
            self.consensus_ratio_[self.consensus_ratio_ == 1.0].index
        )
        
        return self
    
    def transform(self, X) -> ig.Graph:
        """
        Return subgraph with only consensus edges (100% consistent membership).
        
        Parameters
        ----------
        X : ig.Graph
            Input graph (should be same as fit input)
            
        Returns
        -------
        ig.Graph
            Subgraph containing only edges with consistent cluster membership
        """
        if not hasattr(self, 'consensus_edges_'):
            raise RuntimeError("Must call fit() before transform()")
        
        # Find edges to keep
        edges_to_keep = []
        for edge in X.es:
            source_name = X.vs[edge.source]['name']
            target_name = X.vs[edge.target]['name']
            edge_set = frozenset([source_name, target_name])
            
            if edge_set in self.consensus_edges_:
                edges_to_keep.append(edge.index)
        
        # Create subgraph
        self.consensus_graph_ = X.subgraph_edges(edges_to_keep, delete_vertices=False)
        
        return self.consensus_graph_
    
    def fit_transform(self, X, y=None) -> ig.Graph:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : ig.Graph
            Input graph
        y : None
            Ignored
            
        Returns
        -------
        ig.Graph
            Consensus subgraph
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return edge names for sklearn compatibility"""
        if not hasattr(self, 'consensus_edges_'):
            raise RuntimeError("Must call fit() before get_feature_names_out()")
        return np.array([str(edge) for edge in self.consensus_edges_])
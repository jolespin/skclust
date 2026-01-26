# -*- coding: utf-8 -*-
# skclust/graph.py

import numpy as np
import pandas as pd
import igraph as ig
from itertools import combinations
from typing import Optional
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from loguru import logger
import sys


def cluster_membership_cooccurrence(
    df: pd.DataFrame,
    edge_list: Optional[list] = None,  # NEW: Only compute for these edges
    edge_type: str = "Edge",
    iteration_type: str = "Iteration"
) -> pd.DataFrame:
    """
    Compute pairwise cluster membership co-occurrence across iterations.
    
    OPTIMIZED: If edge_list provided, only computes for actual graph edges
    instead of all possible node pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where rows are nodes and columns are iterations.
    edge_list : list of frozenset, optional
        Only compute co-occurrence for these edges. If None, computes for all pairs.
    edge_type : str, default="Edge"
        Name for the index of the output DataFrame (node pairs)
    iteration_type : str, default="Iteration"
        Name for the columns of the output DataFrame (iterations)
    
    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with co-occurrence for each edge/pair.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    X = df.values
    nodes = df.index.values
    iterations = df.columns
    n_nodes = len(nodes)
    n_iterations = len(iterations)
    
    # Create node name to index mapping
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    if edge_list is not None:
        # OPTIMIZED PATH: Only compute for specified edges
        n_pairs = len(edge_list)
        result = np.empty((n_pairs, n_iterations), dtype=bool)
        
        # Convert edges to index pairs
        edge_idx_pairs = []
        valid_edges = []
        
        for edge in edge_list:
            edge_nodes = list(edge)
            if len(edge_nodes) != 2:
                continue  # Skip self-loops
            
            node_a, node_b = edge_nodes
            if node_a in node_to_idx and node_b in node_to_idx:
                idx_a = node_to_idx[node_a]
                idx_b = node_to_idx[node_b]
                edge_idx_pairs.append((idx_a, idx_b))
                valid_edges.append(edge)
        
        edge_idx_pairs = np.array(edge_idx_pairs, dtype=np.int32)
        
        # Vectorized comparison for only these edges
        for i in range(n_iterations):
            col = X[:, i]
            result[:len(valid_edges), i] = col[edge_idx_pairs[:, 0]] == col[edge_idx_pairs[:, 1]]
        
        return pd.DataFrame(
            data=result[:len(valid_edges)],
            index=pd.Index(valid_edges, name=edge_type),
            columns=pd.Index(iterations, name=iteration_type),
        )
    
    else:
        # ORIGINAL PATH: Compute for all possible pairs
        n_pairs = (n_nodes * (n_nodes - 1)) // 2
        result = np.empty((n_pairs, n_iterations), dtype=bool)
        
        pairs = np.array(list(combinations(range(n_nodes), 2)), dtype=np.int32)
        
        for i in range(n_iterations):
            col = X[:, i]
            result[:, i] = col[pairs[:, 0]] == col[pairs[:, 1]]
        
        edge_labels = [frozenset([nodes[i], nodes[j]]) for i, j in pairs]
        
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


class ConsensusLeidenClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    Sklearn-compatible transformer for consensus Leiden clustering.
    
    Runs multiple iterations of Leiden with different random seeds in parallel,
    then returns only edges with consistent cluster membership across all iterations.
    Final cluster labels are determined by connected components in the consensus graph.
    
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
    cluster_prefix : str, default="leiden_"
        Prefix for cluster labels (e.g., "leiden_1", "leiden_2", ...)
    n_jobs : int, default=1
        Number of parallel processes. 
        - 1: Sequential execution (no multiprocessing)
        - -1: Use all available CPUs
        - >1: Use specific number of processes
    verbose : int, default=0
        Verbosity level (sklearn-style):
        - 0: Silent
        - 1: Progress bars only (tqdm)
        - 2: Stage information (loguru INFO)
        - 3: Detailed timing (loguru DEBUG)
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
    labels_ : pd.Series
        Cluster labels for each node (dtype: str with cluster_prefix)
    graph_ : ig.Graph
        Original input graph (stored for reference)
    consensus_graph_ : ig.Graph
        Subgraph containing only consensus edges (100% co-occurrence)
    n_clusters_ : int
        Number of clusters found
        
    Notes
    -----
    Multiprocessing uses 'spawn' context for cross-platform compatibility.
    Each process gets a copy of the graph, which is memory-intensive for large graphs.
    For very large graphs (>100k nodes), consider using n_jobs=1 or smaller n_iter.
    
    The leidenalg library is thread-safe but multiprocessing provides better
    performance since each process runs independently without GIL contention.
    
    This class inherits from ClusterMixin, providing the fit_predict() method.
    """
    
    def __init__(
        self,
        n_iter: int = 100,
        weight: Optional[str] = None,
        random_state: int = 0,
        partition_type=None,
        resolution_parameter: float = 1.0,
        n_iterations: int = -1,
        cluster_prefix: str = "leiden_",
        n_jobs: int = 1,
        verbose: int = 0,
        leiden_kws: Optional[dict] = None,
    ):
        self.n_iter = n_iter
        self.weight = weight
        self.random_state = random_state
        self.partition_type = partition_type
        self.resolution_parameter = resolution_parameter
        self.n_iterations = n_iterations
        self.cluster_prefix = cluster_prefix
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.leiden_kws = leiden_kws or {}
    
    def _log(self, message: str, level: str = "info"):
        """Log message based on verbosity level"""
        if self.verbose == 0:
            return
        
        if level == "info" and self.verbose >= 2:
            logger.info(message)
        elif level == "debug" and self.verbose >= 3:
            logger.debug(message)
    
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
        import time
        start_time = time.time()
        
        # Validate graph
        self._log("Validating input graph", "info")
        if not isinstance(X, ig.Graph):
            raise TypeError("Graph must be igraph.Graph instance")
        if 'name' not in X.vs.attributes():
            raise ValueError("Graph vertices must have 'name' attribute")
        if self.weight is not None and self.weight not in X.es.attributes():
            raise ValueError(f"Weight attribute '{self.weight}' not found in graph edges")
        
        self.graph_ = X
        nodes_list = np.asarray(X.vs['name'])
        
        self._log(f"Graph: {X.vcount()} nodes, {X.ecount()} edges", "info")
        
        # Import and setup partition type
        self._log("Setting up Leiden algorithm", "debug")
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
        
        self._log(f"Resolution parameter: {leiden_kws_full.get('resolution_parameter', 'N/A')}", "debug")
        
        # Determine number of jobs
        n_jobs = cpu_count() if self.n_jobs == -1 else self.n_jobs
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or >= 1, got {self.n_jobs}")
        
        self._log(f"Using {n_jobs} parallel jobs", "info")
        
        # Prepare worker arguments
        random_seeds = list(range(self.random_state, self.random_state + self.n_iter))
        weight_attr = self.weight if self.weight is not None else None
        worker_args = [
            (X, weight_attr, seed, leiden_kws_full, nodes_list)
            for seed in random_seeds
        ]
        
        # Run partitions
        partition_start = time.time()
        self._log(f"Running {self.n_iter} Leiden iterations", "info")
        
        if n_jobs == 1:
            # Sequential execution
            if self.verbose >= 1:
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
            
            if self.verbose >= 1:
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
        
        partition_time = time.time() - partition_start
        self._log(f"Leiden iterations completed in {partition_time:.2f}s", "info")
        
        # Convert to DataFrame
        df_start = time.time()
        self._log("Converting partitions to DataFrame", "debug")
        self.partitions_ = pd.DataFrame(partitions).T
        self.partitions_.index.name = "Node"
        self.partitions_.columns.name = "Iteration"
        self._log(f"DataFrame conversion: {time.time() - df_start:.2f}s", "debug")
        
        # Compute membership co-occurrence matrix
        cooccur_start = time.time()
        self._log("Computing cluster membership co-occurrence matrix", "info")

        # OPTIMIZATION: Only compute for edges that actually exist in graph
        edge_list = [frozenset([X.vs[e.source]['name'], X.vs[e.target]['name']]) 
                    for e in X.es]

        self.membership_matrix_ = cluster_membership_cooccurrence(
            self.partitions_,
            edge_list=edge_list  # NEW: Only compute for actual edges
        )

        cooccur_time = time.time() - cooccur_start
        self._log(f"Co-occurrence matrix: {self.membership_matrix_.shape}, computed in {cooccur_time:.2f}s", "info")
        
        # Compute consensus metrics
        consensus_start = time.time()
        self._log("Computing consensus metrics", "debug")
        self.consensus_ratio_ = self.membership_matrix_.mean(axis=1)
        self.consensus_edges_ = set(
            self.consensus_ratio_[self.consensus_ratio_ == 1.0].index
        )
        self._log(f"Found {len(self.consensus_edges_)} consensus edges (100% co-occurrence)", "info")
        self._log(f"Consensus metrics: {time.time() - consensus_start:.2f}s", "debug")
        
        # Create consensus graph - OPTIMIZED VERSION
        graph_start = time.time()
        self._log("Building consensus graph from edges", "info")
        
        # BOTTLENECK FIX: Create node->index mapping once
        name_to_idx = {v['name']: v.index for v in X.vs}
        
        # Pre-filter edges efficiently using vectorized operations
        edges_to_keep = []
        
        # Convert consensus_edges to a format optimized for lookup
        # Use tuple pairs instead of frozensets for faster comparison
        consensus_edge_tuples = set()
        for edge in self.consensus_edges_:
            nodes = tuple(sorted(edge))  # Sort to handle undirected
            consensus_edge_tuples.add(nodes)
        
        # Iterate through graph edges only once
        if self.verbose >= 1:
            edge_iter = tqdm(X.es, desc="Filtering consensus edges", total=X.ecount())
        else:
            edge_iter = X.es
            
        for edge in edge_iter:
            source_name = X.vs[edge.source]['name']
            target_name = X.vs[edge.target]['name']
            edge_tuple = tuple(sorted([source_name, target_name]))
            
            if edge_tuple in consensus_edge_tuples:
                edges_to_keep.append(edge.index)
        
        self._log(f"Edge filtering: {time.time() - graph_start:.2f}s", "debug")
        
        # Create subgraph
        subgraph_start = time.time()
        self._log(f"Creating subgraph with {len(edges_to_keep)} edges", "debug")
        self.consensus_graph_ = X.subgraph_edges(edges_to_keep, delete_vertices=False)
        self._log(f"Subgraph creation: {time.time() - subgraph_start:.2f}s", "debug")
        
        graph_time = time.time() - graph_start
        self._log(f"Consensus graph built in {graph_time:.2f}s", "info")
        
        # Generate cluster labels from connected components
        label_start = time.time()
        self._log("Computing connected components for cluster labels", "info")
        components = self.consensus_graph_.connected_components()
        
        # Build cluster labels, sorted by cluster size (largest first)
        node_to_cluster = {}
        cluster_sizes = []
        
        for component in components:
            nodes_in_component = [self.consensus_graph_.vs[idx]['name'] for idx in component]
            cluster_sizes.append((len(nodes_in_component), nodes_in_component))
        
        # Sort by size (descending)
        cluster_sizes.sort(key=lambda x: x[0], reverse=True)
        
        # Assign labels
        for i, (size, nodes) in enumerate(cluster_sizes, start=1):
            cluster_label = f"{self.cluster_prefix}{i}"
            for node in nodes:
                node_to_cluster[node] = cluster_label
        
        # Create series with all nodes from original graph
        all_nodes = [v['name'] for v in X.vs]
        self.labels_ = pd.Series(node_to_cluster, name="Cluster")
        self.labels_ = self.labels_.reindex(all_nodes)  # Ensures all nodes are included
        self.labels_.index.name = "Node"
        self.n_clusters_ = len(cluster_sizes)
        
        label_time = time.time() - label_start
        self._log(f"Found {self.n_clusters_} clusters in {label_time:.2f}s", "info")
        
        total_time = time.time() - start_time
        self._log(f"Total fit time: {total_time:.2f}s", "info")
        
        # Summary
        if self.verbose >= 2:
            logger.info("=" * 60)
            logger.info("CONSENSUS LEIDEN CLUSTERING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Input: {X.vcount()} nodes, {X.ecount()} edges")
            logger.info(f"Iterations: {self.n_iter} (parallel jobs: {n_jobs})")
            logger.info(f"Consensus edges: {len(self.consensus_edges_)} ({100*len(self.consensus_edges_)/X.ecount():.2f}%)")
            logger.info(f"Clusters: {self.n_clusters_}")
            logger.info(f"Nodes in clusters: {self.labels_.notna().sum()} ({100*self.labels_.notna().sum()/X.vcount():.2f}%)")
            logger.info("=" * 60)
            logger.info("TIMING BREAKDOWN")
            logger.info("=" * 60)
            logger.info(f"Leiden iterations: {partition_time:.2f}s ({100*partition_time/total_time:.1f}%)")
            logger.info(f"Co-occurrence matrix: {cooccur_time:.2f}s ({100*cooccur_time/total_time:.1f}%)")
            logger.info(f"Consensus graph: {graph_time:.2f}s ({100*graph_time/total_time:.1f}%)")
            logger.info(f"Cluster labels: {label_time:.2f}s ({100*label_time/total_time:.1f}%)")
            logger.info(f"Total: {total_time:.2f}s")
            logger.info("=" * 60)
        
        return self
    
    def transform(self, X) -> pd.Series:
        """
        Return cluster labels based on connected components in consensus graph.
        
        Parameters
        ----------
        X : ig.Graph
            Input graph (should be same as fit input)
            
        Returns
        -------
        pd.Series
            Cluster labels for each node, indexed by node name.
            Labels are formatted as "{cluster_prefix}{i}" where i is the
            cluster number (sorted by size, largest first).
        """
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Must call fit() before transform()")
        
        return self.labels_
    
    def fit_transform(self, X, y=None) -> pd.Series:
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
        pd.Series
            Cluster labels
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return edge names for sklearn compatibility"""
        if not hasattr(self, 'consensus_edges_'):
            raise RuntimeError("Must call fit() before get_feature_names_out()")
        return np.array([str(edge) for edge in self.consensus_edges_])
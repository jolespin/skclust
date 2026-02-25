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


def leiden_stability(consensus_ratio: pd.Series) -> pd.Series:
    """
    Compute consensus stability metrics from edge consensus ratios.
    
    Parameters
    ----------
    consensus_ratio : pd.Series
        Consensus ratio for each edge (fraction of iterations with co-clustering)
    
    Returns
    -------
    pd.Series
        Stability metrics with entries:
        - n_edges: Total number of edges
        - mean_consensus: Mean consensus ratio
        - median_consensus: Median consensus ratio
        - std_consensus: Standard deviation of consensus ratio
        - min_consensus: Minimum consensus ratio
        - max_consensus: Maximum consensus ratio
        - pct_100: Percentage of edges with 100% consensus
        - pct_90plus: Percentage of edges with ≥90% consensus
        - pct_80plus: Percentage of edges with ≥80% consensus
        - pct_70plus: Percentage of edges with ≥70% consensus
        - pct_60plus: Percentage of edges with ≥60% consensus
        - pct_below_50: Percentage of edges with <50% consensus
        - q25: 25th percentile of consensus ratio
        - q75: 75th percentile of consensus ratio
    """
    values = consensus_ratio.values
    
    return pd.Series({
        'n_edges': len(values),
        'mean_consensus': values.mean(),
        'median_consensus': np.median(values),
        'std_consensus': values.std(),
        'min_consensus': values.min(),
        'max_consensus': values.max(),
        'pct_100': (values == 1.0).mean() * 100,
        'pct_90plus': (values >= 0.9).mean() * 100,
        'pct_80plus': (values >= 0.8).mean() * 100,
        'pct_70plus': (values >= 0.7).mean() * 100,
        'pct_60plus': (values >= 0.6).mean() * 100,
        'pct_below_50': (values < 0.5).mean() * 100,
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75),
    }, name="Stability")


def cluster_membership_cooccurrence(
    df: pd.DataFrame,
    edge_list: Optional[list] = None,
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
                continue
            
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


def _compute_modularity(graph, labels, weight=None):
    """
    Compute modularity for a graph given cluster labels.
    
    Nodes present in the graph but absent from labels (or NaN) are 
    each assigned a unique singleton cluster ID.
    
    Parameters
    ----------
    graph : ig.Graph
        Graph with named vertices
    labels : pd.Series
        Cluster labels indexed by node name
    weight : str or None
        Edge weight attribute name
        
    Returns
    -------
    float
        Modularity score
    """
    unique_labels = labels.dropna().unique()
    label_to_int = {label: i for i, label in enumerate(sorted(unique_labels))}
    next_id = len(unique_labels)
    
    membership = []
    for v in graph.vs:
        name = v['name']
        label = labels.get(name)
        if pd.notna(label) and label in label_to_int:
            membership.append(label_to_int[label])
        else:
            membership.append(next_id)
            next_id += 1
    
    return graph.modularity(membership, weights=weight)


class ConsensusLeidenClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    Sklearn-compatible transformer for consensus Leiden clustering.
    
    Runs multiple iterations of Leiden with different random seeds in parallel,
    computes edge consensus ratios, and builds a consensus graph from edges 
    meeting the consensus threshold. Final cluster labels are determined by 
    connected components in the consensus graph, optionally filtered by 
    minimum cluster size.
    
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
    consensus_threshold : float, default=1.0
        Minimum consensus ratio (0.0-1.0) for edges to include in the
        consensus graph. 1.0 means only edges with 100% co-clustering
        across all iterations are retained.
    minimum_cluster_size : int, default=1
        Minimum number of nodes for a cluster to be retained.
        Clusters smaller than this are removed from labels_ and 
        their nodes are excluded from filtered_graph_.
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
        Edge pairs meeting the consensus_threshold
    consensus_ratio_ : pd.Series
        Proportion of iterations each edge had consistent membership
    stability_report_ : pd.Series
        Consensus stability metrics (percentiles, thresholds, etc.)
    labels_ : pd.Series
        Cluster labels for each node in qualifying clusters, indexed by 
        node name. Only contains nodes assigned to clusters meeting 
        minimum_cluster_size. Use excluded_nodes_ to see what was dropped.
    excluded_nodes_ : pd.Index
        Node names excluded from clustering results. Includes nodes with 
        no consensus edges and nodes from clusters below minimum_cluster_size.
    consensus_graph_ : ig.Graph
        Subgraph containing only consensus edges (isolated nodes removed).
        Represents the raw consensus structure.
    filtered_graph_ : ig.Graph
        Subgraph of the original graph induced on nodes in qualifying 
        clusters (>= minimum_cluster_size). Retains original edge structure.
        This is the graph to use for downstream analysis.
    modularity_ : pd.Series
        Modularity scores with keys:
        - 'initial': On the full input graph
        - 'consensus': On the consensus graph (near 1.0 by construction)
        - 'filtered': On the filtered graph (the meaningful metric)
    n_clusters_ : int
        Number of clusters meeting minimum_cluster_size
        
    Notes
    -----
    The original input graph is NOT stored. Use filtered_graph_ for 
    downstream analysis and consensus_graph_ for inspecting consensus structure.
    
    Multiprocessing uses 'spawn' context for cross-platform compatibility.
    Each process gets a copy of the graph, which is memory-intensive for large graphs.
    For very large graphs (>100k nodes), consider using n_jobs=1 or smaller n_iter.
    """
    
    def __init__(
        self,
        n_iter: int = 100,
        weight: Optional[str] = None,
        random_state: int = 0,
        partition_type=None,
        resolution_parameter: float = 1.0,
        n_iterations: int = -1,
        consensus_threshold: float = 1.0,
        minimum_cluster_size: int = 1,
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
        self.consensus_threshold = consensus_threshold
        self.minimum_cluster_size = minimum_cluster_size
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
        Run N iterations of Leiden clustering and compute consensus.
        
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
        import os
        os.environ.pop('MallocStackLogging', None)
        os.environ.pop('MallocStackLoggingOutputFilename', None)
        import time
        start_time = time.time()
        
        # ================================================================
        # Phase 1: Validate input
        # ================================================================
        self._log("Validating input graph", "info")
        if not isinstance(X, ig.Graph):
            raise TypeError("Graph must be igraph.Graph instance")
        if 'name' not in X.vs.attributes():
            raise ValueError("Graph vertices must have 'name' attribute")
        if self.weight is not None and self.weight not in X.es.attributes():
            raise ValueError(f"Weight attribute '{self.weight}' not found in graph edges")
        if not 0.0 <= self.consensus_threshold <= 1.0:
            raise ValueError(f"consensus_threshold must be between 0.0 and 1.0, got {self.consensus_threshold}")
        if self.minimum_cluster_size < 1:
            raise ValueError(f"minimum_cluster_size must be >= 1, got {self.minimum_cluster_size}")
        
        nodes_list = np.asarray(X.vs['name'])
        
        self._log(f"Graph: {X.vcount()} nodes, {X.ecount()} edges", "info")
        
        # ================================================================
        # Phase 2: Setup Leiden algorithm
        # ================================================================
        self._log("Setting up Leiden algorithm", "debug")
        try:
            from leidenalg import find_partition, RBConfigurationVertexPartition
        except ModuleNotFoundError:
            raise ImportError("Install leidenalg: pip install leidenalg")
        
        partition_type = self.partition_type if self.partition_type is not None else RBConfigurationVertexPartition
        
        leiden_kws_full = {
            'partition_type': partition_type,
            'n_iterations': self.n_iterations,
            **self.leiden_kws
        }
        
        if (self.partition_type is None or partition_type is RBConfigurationVertexPartition):
            if 'resolution_parameter' not in self.leiden_kws:
                leiden_kws_full['resolution_parameter'] = self.resolution_parameter
        
        self._log(f"Resolution parameter: {leiden_kws_full.get('resolution_parameter', 'N/A')}", "debug")
        
        # ================================================================
        # Phase 3: Run Leiden iterations
        # ================================================================
        n_jobs = cpu_count() if self.n_jobs == -1 else self.n_jobs
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or >= 1, got {self.n_jobs}")
        
        self._log(f"Using {n_jobs} parallel jobs", "info")
        
        random_seeds = list(range(self.random_state, self.random_state + self.n_iter))
        weight_attr = self.weight if self.weight is not None else None
        worker_args = [
            (X, weight_attr, seed, leiden_kws_full, nodes_list)
            for seed in random_seeds
        ]
        
        partition_start = time.time()
        self._log(f"Running {self.n_iter} Leiden iterations", "info")
        
        if n_jobs == 1:
            if self.verbose >= 1:
                partitions = [
                    _leiden_worker(args) 
                    for args in tqdm(worker_args, desc="Leiden clustering")
                ]
            else:
                partitions = [_leiden_worker(args) for args in worker_args]
        else:
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
        
        # ================================================================
        # Phase 4: Build partitions DataFrame and co-occurrence matrix
        # ================================================================
        df_start = time.time()
        self._log("Converting partitions to DataFrame", "debug")
        self.partitions_ = pd.DataFrame(partitions).T
        self.partitions_.index.name = "Node"
        self.partitions_.columns.name = "Iteration"
        self._log(f"DataFrame conversion: {time.time() - df_start:.2f}s", "debug")
        
        cooccur_start = time.time()
        self._log("Computing cluster membership co-occurrence matrix", "info")
        
        edge_list = [frozenset([X.vs[e.source]['name'], X.vs[e.target]['name']]) 
                    for e in X.es]
        
        self.membership_matrix_ = cluster_membership_cooccurrence(
            self.partitions_,
            edge_list=edge_list,
        )
        
        cooccur_time = time.time() - cooccur_start
        self._log(f"Co-occurrence matrix: {self.membership_matrix_.shape}, computed in {cooccur_time:.2f}s", "info")
        
        # ================================================================
        # Phase 5: Consensus metrics and consensus graph
        # ================================================================
        consensus_start = time.time()
        self._log("Computing consensus metrics", "debug")
        self.consensus_ratio_ = self.membership_matrix_.mean(axis=1)
        self.consensus_edges_ = set(
            self.consensus_ratio_[self.consensus_ratio_ >= self.consensus_threshold].index
        )
        self.stability_report_ = leiden_stability(self.consensus_ratio_)
        
        self._log(f"Found {len(self.consensus_edges_)} consensus edges (>= {self.consensus_threshold} threshold)", "info")
        self._log(f"Consensus metrics: {time.time() - consensus_start:.2f}s", "debug")
        
        # Build consensus graph
        graph_start = time.time()
        self._log("Building consensus graph", "info")
        
        consensus_edge_tuples = set()
        for edge in self.consensus_edges_:
            consensus_edge_tuples.add(tuple(sorted(edge)))
        
        edges_to_keep = []
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
        
        # Always remove isolated nodes from consensus graph
        self.consensus_graph_ = X.subgraph_edges(edges_to_keep, delete_vertices=True)
        
        graph_time = time.time() - graph_start
        self._log(f"Consensus graph: {self.consensus_graph_.vcount()} nodes, {self.consensus_graph_.ecount()} edges ({graph_time:.2f}s)", "info")
        
        # ================================================================
        # Phase 6: Cluster labels from connected components
        # ================================================================
        label_start = time.time()
        self._log("Computing connected components for cluster labels", "info")
        components = self.consensus_graph_.connected_components()
        
        # Assign labels sorted by cluster size (largest first)
        node_to_cluster = {}
        cluster_sizes = []
        
        for component in components:
            nodes_in_component = [self.consensus_graph_.vs[idx]['name'] for idx in component]
            cluster_sizes.append((len(nodes_in_component), nodes_in_component))
        
        cluster_sizes.sort(key=lambda x: x[0], reverse=True)
        
        for i, (size, nodes) in enumerate(cluster_sizes, start=1):
            cluster_label = f"{self.cluster_prefix}{i}"
            for node in nodes:
                node_to_cluster[node] = cluster_label
        
        # All labels before minimum_cluster_size filtering
        all_nodes = [v['name'] for v in X.vs]
        labels_all = pd.Series(node_to_cluster, name="Cluster")
        labels_all = labels_all.reindex(all_nodes)
        labels_all.index.name = "Node"
        
        # ================================================================
        # Phase 7: Filter by minimum_cluster_size
        # ================================================================
        cluster_counts = labels_all.value_counts()
        valid_clusters = set(cluster_counts[cluster_counts >= self.minimum_cluster_size].index)
        
        # labels_ only contains nodes in qualifying clusters (no NaN rows)
        self.labels_ = labels_all[labels_all.isin(valid_clusters)].copy()
        self.labels_.index.name = "Node"
        self.n_clusters_ = len(valid_clusters)
        
        # Track excluded nodes (no consensus edges OR cluster too small)
        self.excluded_nodes_ = pd.Index(
            sorted(set(all_nodes) - set(self.labels_.index)),
            name="Node",
        )
        
        n_excluded_small_cluster = labels_all.notna().sum() - len(self.labels_)
        n_excluded_no_consensus = labels_all.isna().sum()
        if n_excluded_small_cluster > 0:
            self._log(f"Excluded {n_excluded_small_cluster} nodes from {cluster_counts.shape[0] - len(valid_clusters)} clusters below minimum_cluster_size={self.minimum_cluster_size}", "info")
        if n_excluded_no_consensus > 0:
            self._log(f"Excluded {n_excluded_no_consensus} nodes with no consensus edges", "info")
        
        label_time = time.time() - label_start
        self._log(f"Found {self.n_clusters_} clusters in {label_time:.2f}s", "info")
        
        # ================================================================
        # Phase 8: Build filtered_graph_ and compute modularity
        # ================================================================
        filtered_start = time.time()
        self._log("Building filtered graph and computing modularity", "info")
        
        # Filtered graph: original edges between nodes in qualifying clusters
        valid_node_names = set(self.labels_.index)
        valid_node_indices = [v.index for v in X.vs if v['name'] in valid_node_names]
        self.filtered_graph_ = X.induced_subgraph(valid_node_indices)
        
        self._log(f"Filtered graph: {self.filtered_graph_.vcount()} nodes, {self.filtered_graph_.ecount()} edges", "info")
        
        # Compute modularity on all three graphs
        modularity_initial = _compute_modularity(X, self.labels_, weight=weight_attr)
        modularity_consensus = _compute_modularity(self.consensus_graph_, self.labels_, weight=weight_attr)
        modularity_filtered = _compute_modularity(self.filtered_graph_, self.labels_, weight=weight_attr)
        
        self.modularity_ = pd.Series({
            'initial': modularity_initial,
            'consensus': modularity_consensus,
            'filtered': modularity_filtered,
        }, name="Modularity")
        
        filtered_time = time.time() - filtered_start
        self._log(f"Modularity (initial={modularity_initial:.4f}, consensus={modularity_consensus:.4f}, filtered={modularity_filtered:.4f})", "info")
        self._log(f"Filtered graph + modularity: {filtered_time:.2f}s", "debug")
        
        # ================================================================
        # Phase 9: Summary
        # ================================================================
        total_time = time.time() - start_time
        self._log(f"Total fit time: {total_time:.2f}s", "info")
        
        if self.verbose >= 2:
            logger.info("=" * 60)
            logger.info("CONSENSUS LEIDEN CLUSTERING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Input: {X.vcount()} nodes, {X.ecount()} edges")
            logger.info(f"Iterations: {self.n_iter} (parallel jobs: {n_jobs})")
            logger.info(f"Consensus threshold: {self.consensus_threshold}")
            logger.info(f"Minimum cluster size: {self.minimum_cluster_size}")
            logger.info(f"Consensus edges: {len(self.consensus_edges_)} ({100*len(self.consensus_edges_)/X.ecount():.2f}%)")
            logger.info(f"Consensus graph: {self.consensus_graph_.vcount()} nodes, {self.consensus_graph_.ecount()} edges")
            logger.info(f"Filtered graph: {self.filtered_graph_.vcount()} nodes, {self.filtered_graph_.ecount()} edges")
            logger.info(f"Clusters: {self.n_clusters_}")
            logger.info(f"Nodes in clusters: {len(self.labels_)} ({100*len(self.labels_)/X.vcount():.2f}%)")
            logger.info(f"Excluded nodes: {len(self.excluded_nodes_)}")
            logger.info("=" * 60)
            logger.info("STABILITY REPORT")
            logger.info("=" * 60)
            rep = self.stability_report_
            logger.info(f"Consensus ratio: median={rep['median_consensus']:.3f}, mean={rep['mean_consensus']:.3f}")
            logger.info(f"100% consensus: {rep['pct_100']:.1f}% of edges")
            logger.info(f"≥80% consensus: {rep['pct_80plus']:.1f}% of edges")
            logger.info(f"<50% consensus: {rep['pct_below_50']:.1f}% of edges")
            logger.info("=" * 60)
            logger.info("MODULARITY")
            logger.info("=" * 60)
            logger.info(f"Initial graph:   {self.modularity_['initial']:.4f}")
            logger.info(f"Consensus graph: {self.modularity_['consensus']:.4f}")
            logger.info(f"Filtered graph:  {self.modularity_['filtered']:.4f}")
            logger.info("=" * 60)
            logger.info("TIMING BREAKDOWN")
            logger.info("=" * 60)
            logger.info(f"Leiden iterations: {partition_time:.2f}s ({100*partition_time/total_time:.1f}%)")
            logger.info(f"Co-occurrence matrix: {cooccur_time:.2f}s ({100*cooccur_time/total_time:.1f}%)")
            logger.info(f"Consensus graph: {graph_time:.2f}s ({100*graph_time/total_time:.1f}%)")
            logger.info(f"Cluster labels: {label_time:.2f}s ({100*label_time/total_time:.1f}%)")
            logger.info(f"Filtered graph + modularity: {filtered_time:.2f}s ({100*filtered_time/total_time:.1f}%)")
            logger.info(f"Total: {total_time:.2f}s")
            logger.info("=" * 60)
        
        return self
    
    def transform(self, X) -> pd.Series:
        """
        Return cluster labels.
        
        Parameters
        ----------
        X : ig.Graph
            Input graph (ignored, exists for sklearn compatibility)
            
        Returns
        -------
        pd.Series
            Cluster labels for each node in qualifying clusters, 
            indexed by node name.
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
    
    def get_feature_names_out(self, input_features=None) -> pd.Index:
        """
        Return consensus edges as a pd.Index of frozensets.
        
        Parameters
        ----------
        input_features : None
            Ignored, exists for sklearn compatibility
            
        Returns
        -------
        pd.Index
            Index of frozenset edge pairs from the consensus graph
        """
        if not hasattr(self, 'consensus_graph_'):
            raise RuntimeError("Must call fit() before get_feature_names_out()")
        
        edges = [
            frozenset([self.consensus_graph_.vs[e.source]['name'], 
                       self.consensus_graph_.vs[e.target]['name']])
            for e in self.consensus_graph_.es
        ]
        return pd.Index(edges, name="Edge")
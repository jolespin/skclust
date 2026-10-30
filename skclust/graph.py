# -*- coding: utf-8 -*-
# skclust/graph.py

import inspect
from typing import Optional
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import igraph as ig
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClusterMixin
from tqdm.auto import tqdm
from loguru import logger

from .metrics import pielou_evenness


# ============================================================================
# Leiden workers (module level for multiprocessing pickling)
# ============================================================================
def _leiden_worker(args):
    """
    Sequential Leiden worker. All arguments bundled for self-contained execution.

    Returns the vertex-ordered membership array (membership[i] = community of
    vertex i), which aligns with igraph vertex ids.
    """
    graph, weight, random_seed, leiden_kws_full = args

    try:
        from leidenalg import find_partition
    except ModuleNotFoundError:
        raise ImportError("Install leidenalg: pip install leidenalg")

    partition = find_partition(
        graph,
        weights=weight,
        seed=random_seed,
        **leiden_kws_full,
    )
    return np.asarray(partition.membership)


def _leiden_pool_init(graph, weight, leiden_kws_full):
    """
    Pool initializer storing shared data as process-level globals.

    Called once per worker, so the graph is pickled once per process instead
    of once per iteration.
    """
    global _LEIDEN_GRAPH, _LEIDEN_WEIGHT, _LEIDEN_KWS
    _LEIDEN_GRAPH = graph
    _LEIDEN_WEIGHT = weight
    _LEIDEN_KWS = leiden_kws_full


def _leiden_worker_parallel(random_seed):
    """Parallel Leiden worker. Reads graph/params from globals; only seed varies."""
    try:
        from leidenalg import find_partition
    except ModuleNotFoundError:
        raise ImportError("Install leidenalg: pip install leidenalg")

    partition = find_partition(
        _LEIDEN_GRAPH,
        weights=_LEIDEN_WEIGHT,
        seed=random_seed,
        **_LEIDEN_KWS,
    )
    return np.asarray(partition.membership)


# ============================================================================
# Consensus + quality helpers
# ============================================================================
def _compute_consensus(
    partitions_by_iter: np.ndarray,
    edgelist: np.ndarray,
    n_iter: int,
    store_disagreement_matrix: bool = False,
):
    """
    One-pass edge consensus from per-iteration memberships.

    Parameters
    ----------
    partitions_by_iter : np.ndarray
        Shape (n_iter, n_nodes). Row i is the membership for iteration i,
        indexed by igraph vertex id.
    edgelist : np.ndarray
        Shape (n_edges, 2). Vertex-id endpoints for each edge, in edge-id order.
    n_iter : int
        Number of iterations.
    store_disagreement_matrix : bool
        If True, also build the edge x iteration disagreement matrix as a scipy
        CSR (True where the endpoints land in different clusters). Built directly
        from the disagreements, so it is genuinely sparse for high-consensus data.

    Returns
    -------
    n_agree : np.ndarray
        Shape (n_edges,). Number of iterations in which each edge co-clustered.
    disagreement_matrix : scipy.sparse.csr_matrix or None
        Boolean (n_edges x n_iter) disagreement matrix, or None.

    Notes
    -----
    The loop is over iterations on purpose. Comparing all iterations at once
    (partitions_by_iter[:, a] == partitions_by_iter[:, b]) would materialise an
    (n_iter x n_edges) boolean array. The loop keeps peak memory at O(n_edges).
    """
    idx_a = edgelist[:, 0]
    idx_b = edgelist[:, 1]
    n_edges = edgelist.shape[0]

    n_agree = np.zeros(n_edges, dtype=np.int32)

    rows = []
    cols = []
    for i in range(n_iter):
        membership_i = partitions_by_iter[i]  # contiguous (n_nodes,)
        same = membership_i[idx_a] == membership_i[idx_b]
        n_agree += same
        if store_disagreement_matrix:
            nz = np.flatnonzero(~same)
            rows.append(nz)
            cols.append(np.full(nz.shape, i, dtype=np.int32))

    disagreement_matrix = None
    if store_disagreement_matrix:
        if rows:
            row_idx = np.concatenate(rows)
            col_idx = np.concatenate(cols)
            data = np.ones(row_idx.shape, dtype=bool)
        else:
            row_idx = np.array([], dtype=np.int32)
            col_idx = np.array([], dtype=np.int32)
            data = np.array([], dtype=bool)
        disagreement_matrix = sp.csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(n_edges, n_iter),
        )

    return n_agree, disagreement_matrix


def _compute_quality(graph, labels, partition_type, resolution_parameter, weight=None):
    """
    Quality of a partition under its own objective function.

    Builds a leidenalg partition of `partition_type` from `labels` (without
    re-optimising) and returns `.quality()`. Whatever objective the chosen type
    implements (modularity, CPM, significance, surprise, ...) is what is
    reported. Vertices missing from `labels` are each placed in a singleton.

    `resolution_parameter` and `weight` are only passed to partition types whose
    constructor accepts them, so any leidenalg partition type works unchanged.
    """
    node_to_label = labels.dropna().to_dict()
    unique_labels = sorted(set(node_to_label.values()))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    next_id = len(unique_labels)

    membership = []
    for v in graph.vs:
        label = node_to_label.get(v["name"])
        if label is None:
            membership.append(next_id)
            next_id += 1
        else:
            membership.append(label_to_int[label])

    sig_params = inspect.signature(partition_type).parameters
    partition_kws = {"initial_membership": membership}
    if "resolution_parameter" in sig_params:
        partition_kws["resolution_parameter"] = resolution_parameter
    if weight is not None and "weights" in sig_params:
        partition_kws["weights"] = weight

    return partition_type(graph, **partition_kws).quality()


# ============================================================================
# Consensus Leiden clustering
# ============================================================================
class ConsensusLeidenClustering(BaseEstimator, ClusterMixin):
    """
    Consensus Leiden clustering with unanimous edge agreement.

    Runs `n_iter` Leiden partitions with different random seeds, then keeps only
    edges that co-cluster in *every* iteration (unanimous consensus). Final
    clusters are the connected components of the unanimous-edge graph, optionally
    filtered by `minimum_cluster_size`.

    Unanimous agreement (rather than a tunable threshold) is deliberate: the
    "all iterations agree" relation is an equivalence relation, so its components
    are exactly the cliques of fully-agreeing nodes. Any threshold below 1.0
    would break transitivity and degrade into single-linkage clustering on the
    co-association matrix, so no threshold parameter is exposed. The full
    distribution is still available in `consensus_ratio_` for inspection.

    Parameters
    ----------
    n_iter : int, default=10
        Number of Leiden iterations (random seeds). Lower values make unanimity
        easier to reach, i.e. a weaker consensus filter.
    weight : str or None, default=None
        Edge weight attribute name. None => unweighted.
    random_state : int, default=0
        Seeds are random_state .. random_state + n_iter - 1.
    partition_type : leidenalg partition class, default=None
        None => CPMVertexPartition. Any leidenalg partition type is
        supported (e.g. RBConfigurationVertexPartition, ModularityVertexPartition).
    resolution_parameter : float
        Passed through to any partition type whose constructor accepts it
        (RBConfiguration, CPM, ...). Ignored by types that do not (Modularity,
        Significance, Surprise).
    n_iterations : int, default=-1
        Leiden internal convergence iterations (-1 => until convergence).
    minimum_cluster_size : int, default=1
        Clusters smaller than this are dropped from `labels_`.
    cluster_prefix : str, default="leiden_"
        Cluster label prefix (clusters numbered largest-first).
    n_jobs : int, default=1
        1 => sequential; -1 => all CPUs; >1 => that many processes.
    verbose : int, default=0
        0 silent, 1 progress bars, 2 stage summary, 3 detailed timing.
    store_disagreement_matrix : bool, default=False
        If True, build and retain the edge x iteration disagreement matrix
        (CSR; True where the endpoints land in different clusters). Access via
        get_disagreement_matrix(). For high-consensus data this is mostly False,
        so CSR is genuinely sparse.
    leiden_kws : dict, optional
        Extra kwargs forwarded to leidenalg.find_partition. Set
        resolution_parameter via the parameter above, not here.

    Attributes
    ----------
    partitions_ : pd.DataFrame
        Per-iteration cluster ids (n_nodes x n_iter), int32.
    consensus_ratio_ : pd.Series
        Fraction of iterations each edge co-clustered, indexed by frozenset edge.
    consensus_edges_ : pd.Index
        Unanimous edges (consensus_ratio_ == 1.0), in canonical order.
    disagreement_matrix_ : scipy.sparse.csr_matrix or None
        Disagreement matrix if store_disagreement_matrix else None.
    labels_ : pd.Series
        Cluster label per node in qualifying clusters (no NaN rows).
    cluster_sizes_ : pd.Series
        Node count per cluster, largest first.
    unstable_nodes_ : pd.Index
        Nodes with no unanimous edges (never in the consensus graph).
    discarded_nodes_ : pd.Index
        Nodes in stable-but-too-small clusters.
    consensus_graph_ : ig.Graph
        Unanimous edges among qualifying nodes.
    consensus_graph_discarded_ : ig.Graph
        Unanimous edges among discarded (small-cluster) nodes.
    filtered_graph_ : ig.Graph
        Original edges induced on qualifying nodes (use this downstream).
    quality_ : pd.Series
        Partition quality under the chosen objective: 'initial' (full graph)
        and 'filtered' (filtered graph). Resolution/weight handled per type.
    summary_ : pd.Series
        Graph sizes, cluster counts, mean_consensus_ratio, stable-node and
        stable-edge fractions, discard rate, Pielou cluster-size evenness,
        partition_type / quality_metric / resolution_parameter (context for
        interpreting quality), and quality_initial / quality_filtered. Mixed
        dtype (contains strings).
    n_clusters_, n_nodes_initial_, n_edges_initial_, ... : int
        Size scalars (also in summary_).

    Notes
    -----
    The original input graph is not retained. Multiprocessing uses 'spawn'.
    """

    def __init__(
        self,
        n_iter: int = 10,
        weight: Optional[str] = None,
        resolution_parameter: float = "auto",
        random_state: int = 0,
        partition_type=None,
        n_iterations: int = -1,
        minimum_cluster_size: int = 1,
        cluster_prefix: str = "leiden_",
        n_jobs: int = 1,
        verbose: int = 0,
        store_disagreement_matrix: bool = False,
        leiden_kws: Optional[dict] = None,
    ):
        self.n_iter = n_iter
        self.weight = weight
        self.random_state = random_state
        self.partition_type = partition_type
        self.resolution_parameter = resolution_parameter
        self.n_iterations = n_iterations
        self.minimum_cluster_size = minimum_cluster_size
        self.cluster_prefix = cluster_prefix
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.store_disagreement_matrix = store_disagreement_matrix
        self.leiden_kws = leiden_kws

    def _log(self, message: str, level: str = "info"):
        if self.verbose == 0:
            return
        if level == "info" and self.verbose >= 2:
            logger.info(message)
        elif level == "debug" and self.verbose >= 3:
            logger.debug(message)

    def fit(self, X, y=None):
        """
        Run n_iter Leiden iterations and build the unanimous consensus.

        Parameters
        ----------
        X : ig.Graph
            Graph with named vertices.
        y : None
            Ignored (sklearn compatibility).

        Returns
        -------
        self
        """
        import os
        import time

        os.environ.pop("MallocStackLogging", None)
        os.environ.pop("MallocStackLoggingOutputFilename", None)

        start_time = time.time()

        # Resolve resolution_parameter
        if self.resolution_parameter == "auto":
            logger.warning(
                "resolution_parameter='auto' resolved to median edge weight ({:.4f}). "
                "This is a conservative default; consider a resolution sweep with "
                "domain-informed selection for production analyses.".format(self.resolution_parameter_)
            )
            if self.weight is not None:
                self.resolution_parameter_ = float(np.median(X.es[self.weight]))
            else:
                n = X.vcount()
                self.resolution_parameter_ = X.ecount() / (n * (n - 1) / 2)
            self._log(f"Auto resolution_parameter: {self.resolution_parameter_:.4f}", "info")
        else:
            self.resolution_parameter_ = self.resolution_parameter

        # --- Phase 1: validate input -------------------------------------
        self._log("Validating input graph", "info")
        if not isinstance(X, ig.Graph):
            raise TypeError("Graph must be igraph.Graph instance")
        if "name" not in X.vs.attributes():
            raise ValueError("Graph vertices must have 'name' attribute")
        if self.weight is not None and self.weight not in X.es.attributes():
            raise ValueError(f"Weight attribute '{self.weight}' not found in graph edges")
        if self.minimum_cluster_size < 1:
            raise ValueError(f"minimum_cluster_size must be >= 1, got {self.minimum_cluster_size}")

        nodes_list = np.asarray(X.vs["name"])
        self.n_nodes_initial_ = X.vcount()
        self.n_edges_initial_ = X.ecount()
        self._log(f"Graph: {self.n_nodes_initial_} nodes, {self.n_edges_initial_} edges", "info")

        # --- Phase 2: set up Leiden --------------------------------------
        self._log("Setting up Leiden algorithm", "debug")
        try:
            from leidenalg import CPMVertexPartition
        except ModuleNotFoundError:
            raise ImportError("Install leidenalg: pip install leidenalg")

        partition_type = (
            self.partition_type if self.partition_type is not None
            else CPMVertexPartition
        )
        leiden_kws = self.leiden_kws or {}
        leiden_kws_full = {
            "partition_type": partition_type,
            "n_iterations": self.n_iterations,
            **leiden_kws,
        }
        # Wire resolution_parameter for any partition type that accepts it
        # (RBConfiguration, CPM, ...), unless the user already set it.
        if "resolution_parameter" not in leiden_kws:
            if "resolution_parameter" in inspect.signature(partition_type).parameters:
                leiden_kws_full["resolution_parameter"] = self.resolution_parameter
        self._log(
            f"Partition: {partition_type.__name__}, "
            f"resolution={leiden_kws_full.get('resolution_parameter', 'N/A')}",
            "debug",
        )

        # --- Phase 3: run Leiden iterations ------------------------------
        n_jobs = cpu_count() if self.n_jobs == -1 else self.n_jobs
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or >= 1, got {self.n_jobs}")
        self._log(f"Using {n_jobs} parallel jobs", "info")

        random_seeds = list(range(self.random_state, self.random_state + self.n_iter))
        weight_attr = self.weight

        partition_start = time.time()
        self._log(f"Running {self.n_iter} Leiden iterations", "info")

        if n_jobs == 1:
            worker_args = [
                (X, weight_attr, seed, leiden_kws_full) for seed in random_seeds
            ]
            if self.verbose >= 1:
                results = [
                    _leiden_worker(args)
                    for args in tqdm(worker_args, desc="Leiden clustering")
                ]
            else:
                results = [_leiden_worker(args) for args in worker_args]
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=n_jobs,
                initializer=_leiden_pool_init,
                initargs=(X, weight_attr, leiden_kws_full),
            ) as pool:
                if self.verbose >= 1:
                    results = list(
                        tqdm(
                            pool.imap(_leiden_worker_parallel, random_seeds),
                            total=len(random_seeds),
                            desc="Leiden clustering",
                        )
                    )
                else:
                    results = pool.map(_leiden_worker_parallel, random_seeds)

        partition_time = time.time() - partition_start
        self._log(f"Leiden iterations completed in {partition_time:.2f}s", "info")

        # --- Phase 4: partitions matrix ----------------------------------
        # (n_iter, n_nodes): row access per iteration is contiguous for Phase 5.
        partitions_by_iter = np.asarray(results, dtype=np.int32)
        self.partitions_ = pd.DataFrame(
            partitions_by_iter.T,
            index=pd.Index(nodes_list, name="Node"),
            columns=pd.Index(range(self.n_iter), name="Iteration"),
        )

        # --- Phase 5: edge consensus -------------------------------------
        consensus_start = time.time()
        self._log("Computing edge consensus", "info")

        edgelist = np.asarray(X.get_edgelist(), dtype=np.int64)
        n_agree, disagreement_matrix = _compute_consensus(
            partitions_by_iter,
            edgelist,
            self.n_iter,
            store_disagreement_matrix=self.store_disagreement_matrix,
        )
        unanimous_mask = n_agree == self.n_iter

        # frozenset edge index (built once, reused for ratio + edges + matrix)
        edge_name_pairs = nodes_list[edgelist]
        edge_index = pd.Index(
            [frozenset(pair) for pair in edge_name_pairs.tolist()],
            name="Edge",
        )
        self.consensus_ratio_ = pd.Series(
            n_agree / self.n_iter, index=edge_index, name="ConsensusRatio"
        )
        self.consensus_edges_ = pd.Index(
            sorted(edge_index[unanimous_mask], key=lambda fs: tuple(sorted(fs))),
            name="Edge",
        )

        self.disagreement_matrix_ = disagreement_matrix
        if self.store_disagreement_matrix:
            self._disagreement_edge_index_ = edge_index
            self._disagreement_iteration_index_ = pd.Index(
                range(self.n_iter), name="Iteration"
            )

        self._log(
            f"Found {len(self.consensus_edges_)} unanimous edges "
            f"of {self.n_edges_initial_} ({time.time() - consensus_start:.2f}s)",
            "info",
        )

        # --- Phase 6: consensus graph + components -----------------------
        graph_start = time.time()
        self._log("Building consensus graph", "info")

        edges_to_keep = np.flatnonzero(unanimous_mask).tolist()
        _consensus_graph = X.subgraph_edges(edges_to_keep, delete_vertices=True)
        self.n_nodes_consensus_before_filter_ = _consensus_graph.vcount()
        self.n_edges_consensus_before_filter_ = _consensus_graph.ecount()

        components = _consensus_graph.connected_components()
        cluster_sizes = [
            (len(component), [_consensus_graph.vs[idx]["name"] for idx in component])
            for component in components
        ]
        cluster_sizes.sort(key=lambda x: x[0], reverse=True)

        node_to_cluster = {}
        for i, (_, nodes) in enumerate(cluster_sizes, start=1):
            cluster_label = f"{self.cluster_prefix}{i}"
            for node in nodes:
                node_to_cluster[node] = cluster_label

        all_nodes = nodes_list.tolist()
        labels_all = pd.Series(node_to_cluster, name="Cluster").reindex(all_nodes)
        labels_all.index.name = "Node"
        self._log(
            f"Consensus graph (before filter): "
            f"{self.n_nodes_consensus_before_filter_} nodes, "
            f"{self.n_edges_consensus_before_filter_} edges ({time.time() - graph_start:.2f}s)",
            "info",
        )

        # --- Phase 7: minimum_cluster_size filter ------------------------
        cluster_counts = labels_all.value_counts()
        valid_clusters = set(cluster_counts[cluster_counts >= self.minimum_cluster_size].index)
        discarded_clusters = set(cluster_counts.index) - valid_clusters

        self.labels_ = labels_all[labels_all.isin(valid_clusters)].copy()
        self.labels_.index.name = "Node"
        self.n_clusters_ = len(valid_clusters)

        self.cluster_sizes_ = self.labels_.value_counts()
        self.cluster_sizes_.index.name = "Cluster"
        self.cluster_sizes_.name = "Size"

        consensus_node_names = set(_consensus_graph.vs["name"])
        self.unstable_nodes_ = pd.Index(
            sorted(set(all_nodes) - consensus_node_names), name="Node"
        )
        self.discarded_nodes_ = pd.Index(
            sorted(labels_all[labels_all.isin(discarded_clusters)].dropna().index),
            name="Node",
        )
        if len(self.unstable_nodes_) > 0:
            self._log(f"Unstable nodes (no unanimous edges): {len(self.unstable_nodes_)}", "info")
        if len(self.discarded_nodes_) > 0:
            self._log(
                f"Discarded nodes (cluster size < {self.minimum_cluster_size}): "
                f"{len(self.discarded_nodes_)}",
                "info",
            )

        # --- Phase 8: final graphs + quality -----------------------------
        filtered_start = time.time()
        self._log("Building final graphs and computing quality", "info")

        valid_node_names = set(self.labels_.index)
        discarded_node_names = set(self.discarded_nodes_)

        consensus_names = np.asarray(_consensus_graph.vs["name"])
        valid_consensus_indices = np.flatnonzero(
            np.isin(consensus_names, list(valid_node_names))
        ).tolist()
        self.consensus_graph_ = _consensus_graph.induced_subgraph(valid_consensus_indices)
        self.n_nodes_consensus_after_filter_ = self.consensus_graph_.vcount()
        self.n_edges_consensus_after_filter_ = self.consensus_graph_.ecount()

        discarded_consensus_indices = np.flatnonzero(
            np.isin(consensus_names, list(discarded_node_names))
        ).tolist()
        self.consensus_graph_discarded_ = _consensus_graph.induced_subgraph(
            discarded_consensus_indices
        )

        valid_node_indices = np.flatnonzero(
            np.isin(nodes_list, list(valid_node_names))
        ).tolist()
        self.filtered_graph_ = X.induced_subgraph(valid_node_indices)
        self.n_nodes_filtered_ = self.filtered_graph_.vcount()
        self.n_edges_filtered_ = self.filtered_graph_.ecount()

        quality_initial = _compute_quality(
            X, self.labels_, partition_type, self.resolution_parameter, weight=weight_attr
        )
        quality_filtered = _compute_quality(
            self.filtered_graph_, self.labels_, partition_type,
            self.resolution_parameter, weight=weight_attr,
        )
        self.quality_ = pd.Series(
            {"initial": quality_initial, "filtered": quality_filtered}, name="Quality"
        )
        self._log(
            f"Quality (initial={quality_initial:.4f}, filtered={quality_filtered:.4f}) "
            f"in {time.time() - filtered_start:.2f}s",
            "info",
        )

        # --- Phase 9: summary --------------------------------------------
        n_unstable = len(self.unstable_nodes_)
        n_discarded = len(self.discarded_nodes_)
        self.summary_ = pd.Series(
            {
                "n_nodes_initial": self.n_nodes_initial_,
                "n_edges_initial": self.n_edges_initial_,
                "n_nodes_consensus_before_filter": self.n_nodes_consensus_before_filter_,
                "n_edges_consensus_before_filter": self.n_edges_consensus_before_filter_,
                "n_nodes_consensus_after_filter": self.n_nodes_consensus_after_filter_,
                "n_edges_consensus_after_filter": self.n_edges_consensus_after_filter_,
                "n_nodes_filtered": self.n_nodes_filtered_,
                "n_edges_filtered": self.n_edges_filtered_,
                "n_clusters": self.n_clusters_,
                "n_unstable": n_unstable,
                "n_discarded": n_discarded,
                "mean_consensus_ratio": self.consensus_ratio_.mean(),
                "ratio_stable_nodes": 1 - n_unstable / self.n_nodes_initial_,
                "ratio_stable_edges": (
                    len(self.consensus_edges_) / self.n_edges_initial_
                    if self.n_edges_initial_ else float("nan")
                ),
                "discard_rate": n_discarded / self.n_nodes_initial_,
                "pielou_evenness": pielou_evenness(self.cluster_sizes_.to_numpy()),
                "partition_type": partition_type.__name__,
                "quality_metric": partition_type.__name__.replace("VertexPartition", ""),
                "resolution_parameter": leiden_kws_full.get("resolution_parameter", float("nan")),
                "quality_initial": self.quality_["initial"],
                "quality_filtered": self.quality_["filtered"],
            },
            name="Summary",
        )

        total_time = time.time() - start_time
        self._log(f"Total fit time: {total_time:.2f}s", "info")
        if self.verbose >= 2:
            logger.info(
                "ConsensusLeiden | {} clusters | {} iters x {} jobs | unanimous consensus".format(
                    self.n_clusters_, self.n_iter, n_jobs
                )
            )
            logger.info(
                "Nodes: {} initial -> {} clustered | {:.1%} stable, {:.1%} discarded".format(
                    self.n_nodes_initial_, self.n_nodes_filtered_,
                    self.summary_["ratio_stable_nodes"], self.summary_["discard_rate"],
                )
            )
            logger.info(
                "Quality ({}, resolution={}): initial={:.4f}, filtered={:.4f}".format(
                    partition_type.__name__, self.resolution_parameter,
                    self.quality_["initial"], self.quality_["filtered"],
                )
            )

        return self

    def transform(self, X) -> pd.Series:
        """Return cluster labels (X ignored; sklearn convenience)."""
        if not hasattr(self, "labels_"):
            raise RuntimeError("Must call fit() before transform()")
        return self.labels_

    def fit_transform(self, X, y=None) -> pd.Series:
        """Fit and return cluster labels."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> pd.Index:
        """Unanimous consensus edges (convenience alias for consensus_edges_)."""
        if not hasattr(self, "consensus_edges_"):
            raise RuntimeError("Must call fit() before get_feature_names_out()")
        return self.consensus_edges_

    def get_disagreement_matrix(self) -> pd.DataFrame:
        """
        Return the disagreement matrix as a boolean sparse DataFrame.

        True where an edge's endpoints landed in different clusters in that
        iteration. Requires store_disagreement_matrix=True at fit time.
        """
        if not hasattr(self, "disagreement_matrix_"):
            raise RuntimeError("Must call fit() before get_disagreement_matrix()")
        if self.disagreement_matrix_ is None:
            raise RuntimeError(
                "Disagreement matrix not stored. Set store_disagreement_matrix=True before fitting."
            )
        return pd.DataFrame(
            self.disagreement_matrix_.toarray(),
            index=self._disagreement_edge_index_,
            columns=self._disagreement_iteration_index_,
            dtype=pd.SparseDtype(bool),
        )
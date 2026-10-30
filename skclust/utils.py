# -*- coding: utf-8 -*-
# skclust/utils.py

from __future__ import annotations
 
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sps
from loguru import logger
 
 
def adjacency_to_igraph(A, labels=None, weight_attr="weight"):
    """
    Convert a symmetric weighted sparse matrix to an undirected igraph.
 
    Works for any sparse adjacency matrix (KNN, SNN, etc.).
 
    Parameters
    ----------
    A : scipy.sparse matrix, shape (n, n)
        Symmetric weighted adjacency matrix.
    labels : array-like, optional
        Node names.
    weight_attr : str
        Edge attribute name.
 
    Returns
    -------
    ig.Graph
    """
    import igraph as ig
 
    A_upper = sps.triu(A, k=1, format="coo")
    edges = list(zip(A_upper.row.tolist(), A_upper.col.tolist()))
 
    n = A.shape[0]
    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.es[weight_attr] = A_upper.data.tolist()
 
    if labels is not None:
        graph.vs["name"] = [
            str(v) for v in (labels.tolist() if hasattr(labels, "tolist") else labels)
        ]
    else:
        graph.vs["name"] = [str(i) for i in range(n)]
 
    return graph

def choose_minimum_cluster_size(clusters: pd.Series, coverage_target=0.95, verbosity=0):
    """
    Determine a minimum cluster size threshold from a coverage target.

    Clusters are ranked by size (descending) and accumulated until
    ``coverage_target`` of all assigned nodes are covered. The size of
    the last cluster needed to reach that threshold is returned as the
    minimum cluster size.

    Parameters
    ----------
    clusters : pd.Series
        Cluster labels indexed by node. ``NaN`` entries are treated as
        unassigned and excluded from all calculations.
    coverage_target : float, optional
        Fraction of assigned nodes that must be retained by the size
        filter (default 0.95).
    verbosity : int, optional
        If > 0, log the chosen threshold and retention statistics via
        ``loguru.logger`` (default 0).

    Returns
    -------
    int
        Minimum cluster size at which ``coverage_target`` is met.

    Examples
    --------
    >>> import pandas as pd
    >>> labels = pd.Series([0]*50 + [1]*30 + [2]*10 + [3]*5 + [4]*5)
    >>> choose_minimum_cluster_size(labels, coverage_target=0.95)
    5
    """
    cluster_counts = clusters.dropna().value_counts()  # already sorted descending
    total = cluster_counts.sum()

    crossed = (cluster_counts.cumsum() >= total * coverage_target).idxmax()
    minimum_cluster_size = int(cluster_counts[crossed])

    n_passing = (cluster_counts >= minimum_cluster_size).sum()
    n_nodes_passing = cluster_counts[cluster_counts >= minimum_cluster_size].sum()
    if verbosity > 0:
        logger.info(f"Minimum cluster size: {minimum_cluster_size} (coverage_target={coverage_target})")
        logger.info(f"  Clusters retained: {n_passing}/{len(cluster_counts)}")
        logger.info(f"  Nodes retained:    {n_nodes_passing}/{total} ({n_nodes_passing / total:.1%})")

    return minimum_cluster_size
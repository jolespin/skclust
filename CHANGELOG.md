# Change Log
* [2026.2.25] - Added `consensus_threshold` parameter directly to `ConsensusLeidenClustering.__init__` to apply thresholding during `fit()`. Removed `get_consensus_graph()` and `get_labels()` as they are no longer needed for post-hoc extraction.
* [2026.2.25] - Added `minimum_cluster_size` parameter to filter out small communities. Nodes are now cleanly categorized into `labels_` (valid), `discarded_nodes_` (stable but too small), and `unstable_nodes_` (never achieved consensus).
* [2026.2.25] - Replaced `stability_report_` DataFrame with a comprehensive `summary_` Series that includes graph sizes, node categorizations, consensus metrics, and modularity scores.
* [2026.2.25] - Added automatic modularity computation on the `initial`, `consensus`, and `filtered` graphs, accessible via the new `modularity_` attribute.
* [2026.2.25] - Added `filtered_graph_` attribute which retains the original edge structure but only includes nodes from clusters meeting the `minimum_cluster_size`. Added `consensus_graph_discarded_` for inspecting dropped components.
* [2026.2.25] - Optimized `membership_matrix_` by changing it from a dense DataFrame to a memory-efficient `scipy.sparse.csr_matrix`. Added `get_membership_matrix()` to retrieve it as a pandas DataFrame with `SparseDtype`.
* [2026.2.25] - Updated `consensus_edges_` to be a `pd.Index` instead of a set, and modified `get_feature_names_out()` to return it directly for better scikit-learn compatibility.
* [2026.2.25] - Removed `graph_` attribute from being stored directly on the estimator object to save memory, as it is now logically split into `filtered_graph_` and `consensus_graph_`.
* [2026.2.23] - Added `CosineSimilarityClassifier` and reimplemented to `KNeighborsSimilarity` so the overlapping methods are shared
* [2026.2.23] - Cleaned up `metrics` and removed `distinctiveness_score`
* [2026.2.4] - Added `metrics` submodule with `cv_score`, `eta_squared_score`, `entropy_score`, `cramers_v_score`, `distinctiveness_score`
* [2026.2.1] - Added `leiden_stability(consensus_ratio)` - Computes comprehensive stability metrics from Leiden consensus clustering (n_edges, mean/median/std consensus, percentile thresholds at 100%/90%/80%/70%/60%, percentage below 50%, quartiles)
* [2026.2.1] - Added `stability_report_` automatically computed during `ConsensusLeidenClustering.fit()` - Single-row DataFrame with 14 stability metrics for easy analysis and comparison
* [2026.2.1] - Added `get_consensus_graph(threshold)` - Extract consensus graph at any threshold (0.0-1.0) without refitting, enables post-hoc threshold exploration
* [2026.2.1] - Added `get_labels(threshold)` - Extract cluster labels at any threshold (0.0-1.0) without refitting, computed from connected components in consensus graph
* [2026.2.1] - Fixed `if index == "auto"` issue when providing `pd.Index`
* [2025.1.24] - **Major performance optimization** for `ConsensusLeidenClustering`: Co-occurrence matrix now only computed for edges that exist in graph instead of all possible node pairs (55-45,000x speedup for sparse graphs)
* [2025.1.24] - **Breaking change**: `verbose` parameter in `ConsensusLeidenClustering` changed from `bool` to `int` (0-3) for sklearn-style verbosity levels with `loguru` logging (backward compatible: `verbose=False` → 0, `verbose=True` → 1)
* [2025.1.24] - Added `edge_list` parameter to `cluster_membership_cooccurrence()` to compute co-occurrence only for specified edges
* [2025.1.24] - Optimized consensus graph construction using sorted tuples instead of frozenset lookups (10-100x faster)
* [2025.1.24] - Added comprehensive timing breakdown and performance summary in `ConsensusLeidenClustering` verbose mode
* [2026.1.23] - Added `index="auto"` to `KNeighborsCosine.to_igraph` which uses `.index_labels_` if `X` is a `pd.DataFrame`
* [2026.1.23] - Changed `kneighbors` to `neighbors`
* [2026.1.23] - Added `verbose` to `ConsensusLeidenClustering` to track progress
* [2026.1.9] - Added `graph` submodule with `compute_membership_cooccurrence`,`_leiden_worker`, and `ConsensusLeidenClustering`
* [2026.1.9] - Added `kneighbors` submodule with `kneighbors_graph_from_transformer`,`brute_force_kneighbors_graph_from_rectangular_distance`,`pairwise_distances_kneighbors`,`convert_distance_matrix_to_kneighbors_matrix`,`kneighbors_to_igraph`, and `KNeighborsCosineSimilarity`
* [2026.1.8] - Removed `KMeansRepresentativeSampler` and added a `hierarchical` submodule
* [2025.8.12.post1] - Removed `RepresentativeSampler` to have simplified `KMeansRepresentativeSampler` and later will add `GMMRepresentativeSampler` and `AgglomerativeRepresentativeSampler`
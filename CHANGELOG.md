# Change Log
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
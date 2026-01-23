# Change Log
* [2026.1.23] - Added `index="auto"` to `KNeighborsCosine.to_igraph` which uses `.index_labels_` if `X` is a `pd.DataFrame`
* [2026.1.23] - Changed `kneighbors` to `neighbors`
* [2026.1.23] - Added `verbose` to `ConsensusLeidenClustering` to track progress
* [2026.1.9] - Added `graph` submodule with `compute_membership_cooccurrence`,`_leiden_worker`, and `ConsensusLeidenClustering`
* [2026.1.9] - Added `kneighbors` submodule with `kneighbors_graph_from_transformer`,`brute_force_kneighbors_graph_from_rectangular_distance`,`pairwise_distances_kneighbors`,`convert_distance_matrix_to_kneighbors_matrix`,`kneighbors_to_igraph`, and `KNeighborsCosineSimilarity`
* [2026.1.8] - Removed `KMeansRepresentativeSampler` and added a `hierarchical` submodule
* [2025.8.12.post1] - Removed `RepresentativeSampler` to have simplified `KMeansRepresentativeSampler` and later will add `GMMRepresentativeSampler` and `AgglomerativeRepresentativeSampler`
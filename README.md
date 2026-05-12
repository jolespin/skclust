# skclust

A comprehensive clustering toolkit with hierarchical clustering, k-nearest neighbors, and consensus network analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![scikit-learn compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org)
![Beta](https://img.shields.io/badge/status-beta-orange)
![Not Production Ready](https://img.shields.io/badge/production-not%20ready-red)

## Features

- **Scikit-learn compatible** API for seamless integration
- **Hierarchical clustering** with multiple linkage methods and tree cutting strategies
- **K-nearest neighbors** with cosine similarity using FAISS or sklearn backends
- **Consensus Leiden clustering** with parallel execution and edge co-occurrence analysis
- **Cluster validation metrics** for continuous and binary features
- **FAISS-based kNN classifiers and transformers** via DESlib integration
- **Rich visualizations** with dendrograms and metadata tracks
- **Distance matrix utilities** for kNN graph construction and conversion

## Installation

```bash
pip install skclust
```

### Optional Dependencies

```bash
# For enhanced hierarchical clustering
pip install dynamicTreeCut fastcluster skbio

# For visualization
pip install matplotlib seaborn

# For Leiden clustering
pip install leidenalg igraph

# For fast k-NN with large datasets 
pip install faiss-cpu  # or faiss-gpu (Python < 3.13)

# For FAISS-based kNN classifier/transformer (via DESlib)
pip install deslib

# For ensemble network analysis
pip install ensemble-networkx
```

## Quick Start

### Hierarchical Clustering

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from skclust.hierarchical import HierarchicalClustering

# Generate sample data
X, y = make_blobs(n_samples=100, centers=4, random_state=42)
X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])

# Perform hierarchical clustering with dynamic tree cutting
hc = HierarchicalClustering(
    method='ward',
    cut_method='dynamic',
    min_cluster_size=5,
    cluster_prefix='C'
)

# Fit and get cluster labels
labels = hc.fit_transform(X_df)
print(f"Found {hc.n_clusters_} clusters")

# Plot dendrogram with clusters
fig, axes = hc.plot(figsize=(12, 6), show_clusters=True)
```

**Output:** Cluster labels as numpy array (e.g., `['C1', 'C1', 'C2', ...]`) with `hc.n_clusters_` indicating the number of clusters found.

### Consensus Leiden Clustering

```python
import igraph as ig
from skclust.graph import ConsensusLeidenClustering

# Create graph
graph = ig.Graph.Famous('Zachary')
graph.vs['name'] = [f'node_{i}' for i in range(graph.vcount())]

# Run consensus clustering with 100 iterations in parallel
leiden = ConsensusLeidenClustering(
    n_iter=100,
    resolution_parameter=1.0,
    consensus_threshold=1.0,
    minimum_cluster_size=3,
    n_jobs=-1,
    random_state=42
)

labels = leiden.fit_transform(graph)
print(f"Found {leiden.n_clusters_} clusters")
print(f"Consensus edges: {leiden.consensus_graph_.ecount()}")
print(f"Modularity (filtered): {leiden.modularity_['filtered']:.3f}")
```

**Output:** Returns pandas Series with cluster labels indexed by node names. The `consensus_graph_` contains only edges where nodes consistently clustered together across all iterations. Nodes in clusters smaller than `minimum_cluster_size` are available via `discarded_nodes_`.

### K-Nearest Neighbors with Cosine Similarity

```python
import numpy as np
from skclust.neighbors import KNeighborsCosineSimilarity

# L2-normalized embeddings (required for cosine similarity)
embeddings = np.random.randn(1000, 128).astype(np.float32)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Exact search
knn = KNeighborsCosineSimilarity(n_neighbors=10, mode='exact')
similarities, indices = knn.fit_transform(embeddings)

# Convert to igraph for network analysis
graph = knn.to_igraph(include_self=False)
```

**Output:** `similarities` is shape (n_samples, k) with cosine similarity values (higher = more similar). `indices` contains the neighbor indices for each sample.

## Module Overview

### skclust.hierarchical

**HierarchicalClustering**

Hierarchical clustering with multiple linkage methods and tree cutting strategies.

**Key Parameters:**
- `method`: Linkage method ('ward', 'complete', 'average', 'single')
- `cut_method`: Tree cutting strategy ('dynamic', 'height', 'maxclust')
- `min_cluster_size`: Minimum cluster size for dynamic cutting
- `cluster_prefix`: String prefix for cluster labels (e.g., "C" produces "C1", "C2")

**Key Methods:**
- `fit(X)`: Fit clustering to data (accepts arrays or DataFrames)
- `transform()`: Return cluster labels
- `add_track(name, data, track_type)`: Add metadata for visualization
- `plot()`: Generate dendrogram with optional tracks and cluster colors
- `summary()`: Print clustering statistics

**Attributes:**
- `labels_`: Cluster assignments for each sample
- `n_clusters_`: Number of clusters found
- `linkage_matrix_`: Scipy linkage matrix
- `dendrogram_`: Dendrogram data structure

### skclust.graph

**ConsensusLeidenClustering**

Runs Leiden clustering multiple times with different random seeds, computes edge consensus ratios, and builds a consensus graph from edges meeting the threshold. Final labels come from connected components, optionally filtered by minimum cluster size.

**Key Parameters:**
- `n_iter`: Number of Leiden iterations (default: 100)
- `resolution_parameter`: Controls cluster size (1.0 = modularity, >1.0 = smaller clusters)
- `consensus_threshold`: Minimum consensus ratio for edges (default: 1.0, i.e. 100% agreement)
- `minimum_cluster_size`: Clusters smaller than this are removed from `labels_` (default: 1)
- `weight`: Edge weight attribute name (default: None for unweighted)
- `n_jobs`: Number of parallel processes (-1 = use all CPUs)
- `cluster_prefix`: String prefix for cluster labels (default: "leiden_")
- `verbose`: Verbosity level 0-3 (0=silent, 1=progress bars, 2=stage info, 3=detailed timing)

**Key Methods:**
- `fit(graph)`: Fit on igraph.Graph with named vertices
- `transform(graph)`: Return cluster labels as pandas Series
- `get_membership_matrix()`: Return membership matrix as a pandas DataFrame with SparseDtype

**Attributes:**
- `labels_`: Cluster labels for nodes in qualifying clusters (pd.Series)
- `n_clusters_`: Number of clusters meeting `minimum_cluster_size`
- `partitions_`: Node assignments for each iteration (DataFrame)
- `membership_matrix_`: Sparse boolean edge co-membership matrix (scipy.sparse.csr_matrix)
- `consensus_ratio_`: Proportion of iterations each edge had consistent membership (pd.Series)
- `consensus_edges_`: Edge pairs meeting `consensus_threshold` (pd.Index)
- `consensus_graph_`: Subgraph with only consensus edges, pruned to qualifying clusters
- `consensus_graph_discarded_`: Consensus edges for small clusters that were filtered out
- `filtered_graph_`: Original edge structure induced on nodes in qualifying clusters
- `modularity_`: Modularity scores for initial, consensus, and filtered graphs (pd.Series)
- `summary_`: Comprehensive summary with graph sizes, consensus metrics, and modularity
- `unstable_nodes_`: Nodes with no consensus edges across iterations
- `discarded_nodes_`: Nodes in stable but too-small clusters

**leiden_stability(consensus_ratio)**

Computes comprehensive stability metrics from consensus ratio values.

**cluster_membership_cooccurrence(df)**

Compute edge-wise cluster co-occurrence across iterations.

**Parameters:**
- `df`: DataFrame where rows are nodes and columns are iterations

**Returns:** Sparse boolean matrix showing whether each node pair shared cluster membership in each iteration.

### skclust.neighbors

**KNeighborsCosineSimilarity**

K-nearest neighbors using cosine similarity with FAISS or sklearn backend.

**Key Parameters:**
- `n_neighbors`: Number of neighbors to find
- `mode`: Search strategy ('exact', 'ivf', 'pq')
- `backend`: Library to use ('auto', 'faiss', 'sklearn')

**Key Methods:**
- `fit(X)`: Fit on L2-normalized embeddings
- `transform(X)`: Return (similarities, indices) for query vectors
- `to_igraph()`: Convert to directed igraph

**Attributes:**
- `similarities_`: Cosine similarities to k nearest neighbors
- `indices_`: Indices of k nearest neighbors

**FaissKNNClassifier**

Sklearn-compatible kNN classifier using FAISS via DESlib. Supports exact (brute), Voronoi (IVF), and hierarchical (HNSW) search algorithms.

**Key Parameters:**
- `n_neighbors`: Number of neighbors (default: 5)
- `algorithm`: Search strategy ('brute', 'voronoi', 'hierarchical')
- `n_jobs`: Number of parallel jobs

**Key Methods:**
- `fit(X, y)`: Fit classifier
- `predict(X)`: Predict class labels via majority vote
- `kneighbors(X)`: Return (distances, indices) of k nearest neighbors

**FaissKNNTransformer**

Sklearn-compatible kNN transformer using FAISS via DESlib. Outputs a sparse distance matrix matching sklearn's KNeighborsTransformer interface.

**Key Parameters:**
- `n_neighbors`: Number of neighbors (default: 5)
- `algorithm`: Search strategy ('brute', 'voronoi', 'hierarchical')

**Key Methods:**
- `fit(X)`: Fit transformer
- `transform(X)`: Return sparse csr_matrix of L2 distances to k nearest neighbors

**Utility Functions:**

- `kneighbors_graph_from_transformer()`: Build kNN graph from any KNeighborsTransformer
- `brute_force_kneighbors_graph_from_rectangular_distance()`: Build kNN graph from distance matrix
- `pairwise_distances_kneighbors()`: Compute full or sparse pairwise distances
- `convert_distance_matrix_to_kneighbors_matrix()`: Convert dense distance matrix to sparse kNN matrix
- `kneighbors_to_igraph()`: Convert kNN results to igraph
- `kneighbors_classification_assignment_score()`: Compute assignment quality metrics (ambiguity detection) for kNN classification

### skclust.metrics

Cluster validation metrics for evaluating cluster quality.

**Continuous Features:**
- `cv_score(X, labels)`: Coefficient of variation within each cluster (lower = more compact)
- `eta_squared_score(X, labels)`: Eta-squared (η²), ratio of between-cluster variance to total variance (higher = more separated)

**Binary Features:**
- `entropy_score(X, labels)`: Shannon entropy within each cluster (lower = more consistent)
- `cramers_v_score(X, labels)`: Cramér's V association between cluster membership and feature values (higher = more separated)

## Advanced Usage

### Adding Metadata Tracks to Dendrograms

```python
# Add continuous metadata
sample_scores = pd.Series(np.random.randn(100), index=X_df.index)
hc.add_track('Quality Score', sample_scores, track_type='continuous')

# Add categorical metadata
sample_groups = pd.Series(['A', 'B', 'C'] * 34, index=X_df.index[:100])
hc.add_track('Group', sample_groups, track_type='categorical')

# Plot with all tracks
fig, axes = hc.plot(show_tracks=True, figsize=(12, 10))
```

**Output:** Multi-panel plot with dendrogram on top, followed by cluster assignments and metadata tracks below, all aligned to the same sample order.

### Custom Tree Cutting

```python
# Cut by height threshold
hc_height = HierarchicalClustering(
    method='ward',
    cut_method='height',
    cut_threshold=50.0
)
labels = hc_height.fit_transform(X_df)

# Force specific number of clusters
hc_maxclust = HierarchicalClustering(
    method='complete',
    cut_method='maxclust',
    cut_threshold=5
)
labels = hc_maxclust.fit_transform(X_df)
```

**Output:** `cut_method='height'` cuts tree at specified distance threshold. `cut_method='maxclust'` produces exactly the specified number of clusters.

### Using Distance Matrices

```python
from scipy.spatial.distance import pdist, squareform

# Compute custom distance matrix
distances = pdist(X_df, metric='cosine')
distance_matrix = pd.DataFrame(
    squareform(distances),
    index=X_df.index,
    columns=X_df.index
)

# Cluster using precomputed distances
hc = HierarchicalClustering(method='average')
labels = hc.fit_transform(distance_matrix)
```

**Output:** Works identically to feature-based clustering but uses pre-computed distances. Useful for custom metrics.

### Approximate k-NN with FAISS

```python
# For large datasets, use approximate search
knn_ivf = KNeighborsCosineSimilarity(
    n_neighbors=50,
    mode='ivf',
    n_voronoi_cells='auto',
    n_probes=4
)
similarities, indices = knn_ivf.fit_transform(embeddings)

# Product quantization for memory efficiency
knn_pq = KNeighborsCosineSimilarity(
    n_neighbors=50,
    mode='pq',
    n_subvectors=16,
    n_bits=8
)
similarities, indices = knn_pq.fit_transform(embeddings)
```

**Output:** Faster but approximate nearest neighbor search. IVF uses inverted file index, PQ uses compressed representations. Trade accuracy for speed on large datasets.

## Author

Josh L. Espinoza

## License

Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Original Implementation

The hierarchical clustering implementation is based on the [Soothsayer](https://github.com/jolespin/soothsayer) framework:

**Espinoza JL, Dupont CL, O'Rourke A, Beyhan S, Morales P, et al. (2021) Predicting antimicrobial mechanism-of-action from transcriptomes: A generalizable explainable artificial intelligence approach. PLOS Computational Biology 17(3): e1008857.** [https://doi.org/10.1371/journal.pcbi.1008857](https://doi.org/10.1371/journal.pcbi.1008857)

## Citation

If you use this package in your research, please cite:

```bibtex
@article{espinoza2021predicting,
  title={Predicting antimicrobial mechanism-of-action from transcriptomes: A generalizable explainable artificial intelligence approach},
  author={Espinoza, Josh L and Dupont, Chris L and O'Rourke, Aubrie and Beyhan, Seherzada and Morales, Paula and others},
  journal={PLOS Computational Biology},
  volume={17},
  number={3},
  pages={e1008857},
  year={2021},
  publisher={Public Library of Science San Francisco, CA USA},
  doi={10.1371/journal.pcbi.1008857},
  url={https://doi.org/10.1371/journal.pcbi.1008857}
}
```
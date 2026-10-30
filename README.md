# skclust

Scikit-learn compatible clustering toolkit for hierarchical clustering, k-nearest neighbors graphs, consensus network analysis, and cluster validation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![scikit-learn compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org)

## Installation

```bash
pip install skclust
```

Optional dependencies for specific features:

```bash
pip install dynamicTreeCut     # Dynamic tree cutting
pip install leidenalg igraph   # Leiden clustering
pip install faiss-cpu           # Fast approximate kNN (or faiss-gpu)
pip install deslib              # FAISS-based kNN classifier/transformer
pip install matplotlib seaborn  # Visualization
```

## Quick Start

### Hierarchical Clustering

```python
from sklearn.datasets import make_blobs
from skclust.hierarchical import HierarchicalClustering

X, y = make_blobs(n_samples=100, centers=4, random_state=42)

hc = HierarchicalClustering(method='ward', cut_method='dynamic', min_cluster_size=5)
labels = hc.fit_transform(X)

fig, axes = hc.plot(show_clusters=True)
```

### Consensus Leiden Clustering

```python
import igraph as ig
from skclust.graph import ConsensusLeidenClustering

graph = ig.Graph.Famous('Zachary')
graph.vs['name'] = [f'node_{i}' for i in range(graph.vcount())]

leiden = ConsensusLeidenClustering(n_iter=100, resolution_parameter=1.0, n_jobs=-1, random_state=42)
labels = leiden.fit_transform(graph)
```

### K-Nearest Neighbors Graph

```python
import numpy as np
from skclust.neighbors import KNeighborsCosineGraph

embeddings = np.random.randn(1000, 128).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

knn = KNeighborsCosineGraph(n_neighbors=10)
knn.fit(embeddings)
graph = knn.to_igraph()
```

## Modules

### `skclust.hierarchical`

| Class | Description |
|-------|-------------|
| `HierarchicalClustering` | Agglomerative clustering with multiple linkage methods, tree cutting strategies (dynamic, height, maxclust), dendrogram visualization, and metadata tracks |
| `ConnectivityHierarchicalClustering` | Hierarchical clustering with a connectivity constraint via sklearn backend |

### `skclust.graph`

| Class | Description |
|-------|-------------|
| `ConsensusLeidenClustering` | Repeated Leiden clustering with edge consensus thresholding and parallel execution |

### `skclust.neighbors`

| Class | Description |
|-------|-------------|
| `KNeighborsCosineGraph` | kNN graph using cosine similarity with automatic k selection via knee detection (FAISS or sklearn backend) |
| `SharedNearestNeighborsGraph` | Shared nearest neighbor graph from precomputed kNN with knee-based k selection |
| `EnsembleKNeighborsClassifier` | kNN classifier that ensembles predictions across multiple k values |
| `KNeighborsThresholdClassifier` | Distance-weighted kNN classifier with per-class reject option and margin-based confidence |
| `FaissKNNClassifier` | Sklearn-compatible kNN classifier using FAISS via DESlib |
| `FaissKNNTransformer` | Sklearn-compatible kNN transformer using FAISS via DESlib |

Utility functions: `kneighbors_graph_from_transformer`, `kneighbors_to_snn_graph`, `brute_force_kneighbors_graph_from_rectangular_distance`, `pairwise_distances_kneighbors`, `convert_distance_matrix_to_kneighbors_matrix`, `kneighbors_to_igraph`, `kneighbors_classification_assignment_score`

### `skclust.distance`

| Class / Function | Description |
|------------------|-------------|
| `ClusteredDistances` | Intra- and inter-cluster distance summaries with effect sizes and optional PERMANOVA testing |
| `pairwise_cosine_distances` | Pairwise cosine distance matrix |
| `pairwise_jaccard_distances` | Pairwise Jaccard distance matrix |
| `cosine_distances_to_representatives` | Cosine distances from samples to cluster representatives |

### `skclust.metrics`

| Function | Description |
|----------|-------------|
| `cv_score` | Coefficient of variation within clusters (continuous features) |
| `eta_squared_score` | Between-cluster variance ratio (continuous features) |
| `entropy_score` | Shannon entropy within clusters (binary features) |
| `cramers_v_score` | Association between cluster membership and feature values (binary features) |
| `pielou_evenness` | Evenness of cluster size distribution |

### `skclust.utils`

| Function | Description |
|----------|-------------|
| `adjacency_to_igraph` | Convert adjacency matrix to igraph |
| `choose_minimum_cluster_size` | Select minimum cluster size from a coverage target |

## Author

Josh L. Espinoza

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Citation

The hierarchical clustering implementation is based on the [Soothsayer](https://github.com/jolespin/soothsayer) framework:

> Espinoza JL, Dupont CL, O'Rourke A, Beyhan S, Morales P, et al. (2021) Predicting antimicrobial mechanism-of-action from transcriptomes: A generalizable explainable artificial intelligence approach. *PLOS Computational Biology* 17(3): e1008857. [doi:10.1371/journal.pcbi.1008857](https://doi.org/10.1371/journal.pcbi.1008857)

<details>
<summary>BibTeX</summary>

```bibtex
@article{espinoza2021predicting,
  title={Predicting antimicrobial mechanism-of-action from transcriptomes: A generalizable explainable artificial intelligence approach},
  author={Espinoza, Josh L and Dupont, Chris L and O'Rourke, Aubrie and Beyhan, Seherzada and Morales, Paula and others},
  journal={PLOS Computational Biology},
  volume={17},
  number={3},
  pages={e1008857},
  year={2021},
  doi={10.1371/journal.pcbi.1008857}
}
```

</details>

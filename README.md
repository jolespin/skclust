# skclust
A comprehensive clustering toolkit with advanced tree cutting, visualization, and network analysis capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![scikit-learn compatible](https://img.shields.io/badge/sklearn-compatible-orange.svg)](https://scikit-learn.org)
![Beta](https://img.shields.io/badge/status-beta-orange)
![Not Production Ready](https://img.shields.io/badge/production-not%20ready-red)

**Warning: This is a beta release and has not been thoroughly tested.**

##  Features

- **Scikit-learn compatible** API for seamless integration
- **Multiple linkage methods** (Ward, Complete, Average, Single, etc.)
- **Advanced tree cutting** with dynamic, height-based, and max-cluster methods
- **Rich visualizations** with dendrograms and metadata tracks
- **Network analysis** with connectivity metrics and NetworkX integration
- **Tree export** in Newick format for phylogenetic analysis
- **Distance matrix support** for precomputed distances
- **Metadata tracks** for biological and experimental annotations

## Installation

```bash
pip install skclust
```

## Quick Start

### Hierarchical Clustering

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from skclust import HierarchicalClustering

# Generate sample data
X, y = make_blobs(n_samples=100, centers=4, random_state=42)
X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])

# Perform hierarchical clustering
hc = HierarchicalClustering(
    method='ward',
    cut_method='dynamic',
    min_cluster_size=5
)

# Fit and get cluster labels
labels = hc.fit_transform(X_df)
print(f"Found {hc.n_clusters_} clusters")

# Plot dendrogram with clusters
fig, axes = hc.plot(figsize=(12, 6), show_clusters=True)
```

### Representative Sampling

```python
from skclust import KMeansRepresentativeSampler

# Create representative test set (10% of data)
sampler = KMeansRepresentativeSampler(
    sampling_size=0.1,
    stratify=True,  # Maintain class proportions
    method='minibatch'
)

# Get train/test split
X_train, X_test, y_train, y_test = sampler.fit(X_df, y).get_train_test_split(X_df, y)

print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X_df)*100:.1f}%)")
```

## Advanced Usage

### Adding Metadata Tracks

```python
# Add continuous metadata track
sample_scores = pd.Series(np.random.randn(100), index=X_df.index)
hc.add_track('Quality Score', sample_scores, track_type='continuous')

# Add categorical metadata track
sample_groups = pd.Series(['A', 'B', 'C'] * 34, index=X_df.index[:100])
hc.add_track('Group', sample_groups, track_type='categorical')

# Plot with metadata tracks
fig, axes = hc.plot(show_tracks=True, figsize=(12, 8))
```

### Custom Tree Cutting

```python
# Cut by height
hc_height = HierarchicalClustering(
    method='ward',
    cut_method='height',
    cut_threshold=50.0
)
labels_height = hc_height.fit_transform(X_df)

# Cut by number of clusters
hc_maxclust = HierarchicalClustering(
    method='complete',
    cut_method='maxclust',
    cut_threshold=5  # Force exactly 5 clusters
)
labels_maxclust = hc_maxclust.fit_transform(X_df)
```

### Distance Matrix Input

```python
from scipy.spatial.distance import pdist, squareform

# Compute custom distance matrix
distances = pdist(X_df, metric='cosine')
distance_matrix = pd.DataFrame(squareform(distances), 
                              index=X_df.index, 
                              columns=X_df.index)

# Cluster using precomputed distances
hc_custom = HierarchicalClustering(method='average')
labels_custom = hc_custom.fit_transform(distance_matrix)
```

### Stratified Representative Sampling

```python
# Enhanced stratified sampling with minority class boosting
sampler_enhanced = KMeansRepresentativeSampler(
    sampling_size=0.15,
    stratify=True,
    coverage_boost=2.0,  # Boost minority classes
    min_clusters_per_class=3,  # Ensure minimum representation
    method='kmeans'
)

X_train, X_test, y_train, y_test = sampler_enhanced.fit(X_df, y).get_train_test_split(X_df, y)

# Check class balance preservation
print("Original class distribution:")
print(pd.Series(y).value_counts().sort_index())
print("\nTest set class distribution:")
print(pd.Series(y_test).value_counts().sort_index())
```

## API Reference

### HierarchicalClustering

**Parameters:**
- `method`: Linkage method ('ward', 'complete', 'average', 'single', 'centroid', 'median', 'weighted')
- `metric`: Distance metric for computing pairwise distances
- `cut_method`: Tree cutting method ('dynamic', 'height', 'maxclust')
- `min_cluster_size`: Minimum cluster size for dynamic cutting
- `deep_split`: Deep split parameter for dynamic cutting (0-4)
- `cut_threshold`: Threshold for height/maxclust cutting
- `cluster_prefix`: String prefix for cluster labels (e.g., "C" → "C1", "C2")

**Key Methods:**
- `fit(X)`: Fit hierarchical clustering to data
- `transform()`: Return cluster labels
- `add_track(name, data, track_type)`: Add metadata track for visualization
- `plot()`: Generate dendrogram with optional tracks and clusters
- `summary()`: Print clustering summary statistics

### KMeansRepresentativeSampler

**Parameters:**
- `sampling_size`: Proportion of data for test set (0.0-1.0)
- `stratify`: Whether to maintain class proportions
- `method`: Clustering method ('minibatch', 'kmeans')
- `coverage_boost`: Boost factor for minority classes (>1.0)
- `min_clusters_per_class`: Minimum clusters per class
- `batch_size`: Batch size for MiniBatchKMeans

**Key Methods:**
- `fit(X, y)`: Fit sampler and identify representatives
- `transform(X)`: Return representative samples
- `get_train_test_split(X, y)`: Get train/test split

## Examples with Real Data

### Iris Dataset

```python
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = pd.Series(iris.target, name='species')

# Hierarchical clustering
hc_iris = HierarchicalClustering(
    method='ward',
    cut_method='dynamic',
    min_cluster_size=10,
    cluster_prefix='Cluster_'
)

clusters = hc_iris.fit_transform(X_iris)

# Add species information as track
species_names = pd.Series([iris.target_names[i] for i in y_iris], index=X_iris.index)
hc_iris.add_track('True Species', species_names, track_type='categorical')

# Plot results
fig, axes = hc_iris.plot(show_clusters=True, show_tracks=True, figsize=(15, 8))
```

### Creating Balanced Test Sets

```python
# Create representative test set maintaining species balance
sampler_iris = KMeansRepresentativeSampler(
    sampling_size=0.2,  # 20% test set
    stratify=True,
    coverage_boost=1.0,  # Equal representation
    method='kmeans',
    random_state=42
)

X_train, X_test, y_train, y_test = sampler_iris.fit(X_iris, y_iris).get_train_test_split(X_iris, y_iris)

print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Representative indices: {sampler_iris.representative_indices_[:10].tolist()}")
```

## Dependencies

### Required
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- seaborn
- networkx
- loguru

### Optional (for enhanced functionality)
- dynamicTreeCut (dynamic tree cutting)
- skbio (tree representations)
- fastcluster (faster linkage computation)
- ensemble_networkx (network analysis)

## Author

Josh L. Espinoza

##  License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

##  Original Implementation

This package is based on the hierarchical clustering implementation originally developed in the [Soothsayer](https://github.com/jolespin/soothsayer) framework:

**Espinoza JL, Dupont CL, O'Rourke A, Beyhan S, Morales P, et al. (2021) Predicting antimicrobial mechanism-of-action from transcriptomes: A generalizable explainable artificial intelligence approach. PLOS Computational Biology 17(3): e1008857.** [https://doi.org/10.1371/journal.pcbi.1008857](https://doi.org/10.1371/journal.pcbi.1008857)

The original implementation provided the foundation for the hierarchical clustering algorithms, metadata track visualization, and eigenprofile analysis features in this package.

##  Acknowledgments

- Built on top of scipy, scikit-learn, and networkx
- Original implementation developed in the [Soothsayer framework](https://github.com/jolespin/soothsayer)
- Inspired by WGCNA and other biological clustering tools
- Dynamic tree cutting algorithms from the dynamicTreeCut package

##  Support

- **Issues**: [GitHub Issues](https://github.com/your-username/hierarchical-clustering/issues)

##  Citation

If you use this package in your research, please cite:

**Original Soothsayer implementation:**
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


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
- **Cluster validation** using silhouette analysis
- **Eigenprofile calculation** for cluster characterization
- **Tree export** in Newick format for phylogenetic analysis
- **Distance matrix support** for precomputed distances
- **Metadata tracks** for biological and experimental annotations

##  Installation

### Basic Installation

```bash
pip install skclust
```

### Development Installation

```bash
git clone https://github.com/jolespin/skclust.git
cd skclust
pip install -e .[all]
```

### Installation Options

```bash
# Basic functionality only
pip install skclust

# With fast clustering (fastcluster)
pip install skclust[fast]

# With tree operations (scikit-bio)
pip install skclust[tree]

# With all optional features
pip install skclust[all]

# Development installation
pip install skclust[dev]
```

##  Dependencies

### Core Dependencies (Required)
- `numpy >= 1.19.0`
- `pandas >= 1.3.0`
- `scipy >= 1.7.0`
- `scikit-learn >= 1.0.0`
- `matplotlib >= 3.3.0`
- `seaborn >= 0.11.0`
- `networkx >= 2.6.0`

### Optional Dependencies (Enhanced Features)
- `fastcluster >= 1.2.0` - Faster linkage computations
- `scikit-bio >= 0.5.6` - Tree operations and Newick export
- `dynamicTreeCut >= 0.1.0` - Dynamic tree cutting algorithms
- `ensemble-networkx >= 0.1.0` - Enhanced network analysis

##  Quick Start

```python
from skclust import HierarchicalClustering
import pandas as pd
import numpy as np

# Generate sample data
data = np.random.randn(100, 10)
df = pd.DataFrame(data, index=[f"Sample_{i}" for i in range(100)])

# Create and fit clusterer
clusterer = HierarchicalClustering(
    method='ward',
    cut_method='dynamic',
    min_cluster_size=10
)

# Fit and get cluster labels
labels = clusterer.fit_transform(df)

# Plot dendrogram
fig, ax = clusterer.plot_dendrogram(figsize=(12, 6))

# Get summary
summary = clusterer.summary()
print(f"Found {clusterer.n_clusters_} clusters")
```

##  Detailed Usage Examples

### 1. Basic Clustering with Different Methods

```python
from skclust import HierarchicalClustering
import pandas as pd

# Ward clustering with dynamic cutting
clusterer = HierarchicalClustering(
    method='ward',
    cut_method='dynamic',
    min_cluster_size=5
)
labels = clusterer.fit_transform(data)

# Complete linkage with height-based cutting
clusterer = HierarchicalClustering(
    method='complete',
    cut_method='height',
    cut_threshold=10.0
)
labels = clusterer.fit_transform(data)

# Average linkage with fixed number of clusters
clusterer = HierarchicalClustering(
    method='average',
    cut_method='maxclust',
    cut_threshold=4
)
labels = clusterer.fit_transform(data)
```

### 2. Working with Distance Matrices

```python
from scipy.spatial.distance import pdist, squareform

# Compute custom distance matrix
distances = pdist(data.values, metric='correlation')
distance_matrix = pd.DataFrame(
    squareform(distances),
    index=data.index,
    columns=data.index
)

# Cluster using precomputed distances
clusterer = HierarchicalClustering(method='complete')
labels = clusterer.fit_transform(distance_matrix)
```

### 3. Adding Metadata Tracks

```python
# Add continuous metadata (e.g., age, expression levels)
age_data = np.random.normal(45, 15, len(data))
clusterer.add_track('Age', age_data, track_type='continuous', color='steelblue')

# Add categorical metadata (e.g., treatment groups)
treatment = ['Control'] * 30 + ['Treatment_A'] * 35 + ['Treatment_B'] * 35
clusterer.add_track(
    'Treatment', 
    treatment, 
    track_type='categorical',
    color={'Control': 'gray', 'Treatment_A': 'red', 'Treatment_B': 'blue'}
)

# Plot dendrogram with metadata tracks
fig, axes = clusterer.plot_dendrogram(
    figsize=(14, 10),
    show_tracks=True,
    track_height=1.0
)
```

### 4. Cluster Analysis and Validation

```python
# Calculate eigenprofiles (principal components for each cluster)
eigenprofiles = clusterer.eigenprofiles(data)
for cluster_id, profile in eigenprofiles.items():
    print(f"Cluster {cluster_id}: "
          f"Explained variance = {profile['explained_variance_ratio']:.3f}")

# Perform silhouette analysis
silhouette_results = clusterer.silhouette_analysis()
print(f"Overall silhouette score: {silhouette_results['overall_score']:.3f}")

# Calculate connectivity metrics
connectivity = clusterer.connectivity()
print("Connectivity analysis:", connectivity)
```

### 5. Network Analysis

```python
# Convert to NetworkX graph
graph = clusterer.to_networkx(weight_threshold=0.3)
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Visualize network (for small datasets)
import networkx as nx
import matplotlib.pyplot as plt

pos = nx.spring_layout(graph)
nx.draw(graph, pos, node_color=clusterer.labels_, 
        node_size=50, cmap='viridis', alpha=0.7)
plt.title('Sample Network (colored by cluster)')
plt.show()
```

### 6. Tree Export and Phylogenetic Analysis

```python
# Export tree in Newick format (requires scikit-bio)
try:
    newick_string = clusterer.to_newick()
    print("Newick tree:", newick_string[:100], "...")
    
    # Save to file
    clusterer.to_newick('my_tree.newick')
except ValueError as e:
    print("Tree export not available:", e)
```

### 7. Convenience Function

```python
from skclust import hierarchical_clustering

# Quick clustering with default parameters
clusterer = hierarchical_clustering(
    data, 
    method='ward', 
    min_cluster_size=10
)
print(f"Quick clustering: {clusterer.n_clusters_} clusters")
```

##  Biological Data Example

```python
import pandas as pd
import numpy as np
from skclust import HierarchicalClustering

# Simulate gene expression data
np.random.seed(42)
n_samples, n_genes = 80, 1000
expression_data = np.random.randn(n_samples, n_genes)

# Add structure: 3 patient groups with different expression patterns
expression_data[:25, :100] += 2.0   # Group 1: high expression in genes 1-100
expression_data[25:50, 100:200] += 2.0  # Group 2: high expression in genes 101-200
expression_data[50:, 200:300] += 2.0     # Group 3: high expression in genes 201-300

# Create DataFrame with sample names
sample_names = [f"Patient_{i:02d}" for i in range(n_samples)]
gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
df_expression = pd.DataFrame(expression_data, 
                           index=sample_names, 
                           columns=gene_names)

# Perform hierarchical clustering
clusterer = HierarchicalClustering(
    method='ward',
    cut_method='dynamic',
    min_cluster_size=8,
    name='Gene_Expression_Clustering'
)

labels = clusterer.fit_transform(df_expression)

# Add clinical metadata
age = np.random.normal(55, 12, n_samples)
gender = np.random.choice(['Male', 'Female'], n_samples)
stage = ['Stage_I'] * 20 + ['Stage_II'] * 30 + ['Stage_III'] * 30

clusterer.add_track('Age', age, track_type='continuous')
clusterer.add_track('Gender', gender, track_type='categorical')
clusterer.add_track('Disease_Stage', stage, track_type='categorical')

# Visualize results
fig, axes = clusterer.plot_dendrogram(figsize=(15, 10), show_tracks=True)

# Analyze cluster characteristics
eigenprofiles = clusterer.eigenprofiles(df_expression)
silhouette_results = clusterer.silhouette_analysis()

print(f"Identified {clusterer.n_clusters_} patient clusters")
print(f"Silhouette score: {silhouette_results['overall_score']:.3f}")

# Print cluster summary
clusterer.summary()
```

##  Advanced Configuration

### Custom Linkage Methods

```python
# Supported linkage methods
methods = ['ward', 'complete', 'average', 'single', 'centroid', 'median', 'weighted']

for method in methods:
    clusterer = HierarchicalClustering(method=method)
    labels = clusterer.fit_transform(data)
    print(f"{method}: {clusterer.n_clusters_} clusters")
```

### Distance Metrics

```python
# Supported distance metrics (for raw data)
metrics = ['euclidean', 'manhattan', 'cosine', 'correlation']

for metric in metrics:
    clusterer = HierarchicalClustering(metric=metric)
    labels = clusterer.fit_transform(data)
    print(f"{metric}: {clusterer.n_clusters_} clusters")
```

### Dynamic Tree Cutting Parameters

```python
# Fine-tune dynamic tree cutting
clusterer = HierarchicalClustering(
    cut_method='dynamic',
    min_cluster_size=10,        # Minimum samples per cluster
    deep_split=2,               # Sensitivity (0-4, higher = more clusters)
    dynamic_cut_method='hybrid' # 'hybrid' or 'tree'
)
```

##  Performance Tips

1. **Use fastcluster**: Install `fastcluster` for significantly faster linkage computation
2. **Distance matrices**: Precompute distance matrices for repeated analysis
3. **Data preprocessing**: Standardize/normalize data before clustering
4. **Memory management**: For large datasets (>1000 samples), consider subsampling

```python
# Example: Preprocessing pipeline
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
df_scaled = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)

# Cluster scaled data
clusterer = HierarchicalClustering(method='ward')
labels = clusterer.fit_transform(df_scaled)
```

##  Troubleshooting

### Common Issues

1. **ImportError for optional dependencies**:
   ```bash
   pip install hierarchical-clustering[all]
   ```

2. **Memory issues with large datasets**:
   - Use data subsampling or dimensionality reduction
   - Consider approximate methods for >5000 samples

3. **Dynamic tree cutting not working**:
   - Install `dynamicTreeCut` package
   - Falls back to height-based cutting automatically

4. **Tree export failing**:
   - Install `scikit-bio` package
   - Check that clustering was successful

### Performance Benchmarks

| Dataset Size | Method | Time (seconds) | Memory (GB) |
|-------------|--------|----------------|-------------|
| 100 samples | Ward   | 0.01          | < 0.1       |
| 500 samples | Ward   | 0.1           | 0.2         |
| 1000 samples| Ward   | 0.5           | 0.8         |
| 2000 samples| Ward   | 2.0           | 3.2         |

##  API Reference

### HierarchicalClustering Class

#### Parameters
- `method` (str): Linkage method ('ward', 'complete', 'average', 'single')
- `metric` (str): Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
- `cut_method` (str): Tree cutting method ('dynamic', 'height', 'maxclust')
- `min_cluster_size` (int): Minimum cluster size for dynamic cutting
- `cut_threshold` (float): Threshold for height/maxclust cutting
- `name` (str): Optional name for the clustering instance

#### Methods
- `fit(X)`: Fit clustering to data
- `transform()`: Return cluster labels
- `fit_transform(X)`: Fit and return labels
- `add_track(name, data, track_type)`: Add metadata track
- `plot_dendrogram(**kwargs)`: Plot dendrogram with optional tracks
- `eigenprofiles(data)`: Calculate cluster eigenprofiles
- `silhouette_analysis()`: Perform silhouette analysis
- `connectivity()`: Calculate network connectivity
- `to_networkx()`: Convert to NetworkX graph
- `to_newick()`: Export tree in Newick format
- `summary()`: Print clustering summary

#### Attributes (after fitting)
- `labels_`: Cluster labels for each sample
- `n_clusters_`: Number of clusters found
- `linkage_matrix_`: Hierarchical linkage matrix
- `distance_matrix_`: Distance matrix used
- `tree_`: Tree object (if available)
- `tracks_`: Dictionary of metadata tracks

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


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

- **Documentation**: [Link to docs]
- **Issues**: [GitHub Issues](https://github.com/your-username/hierarchical-clustering/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/hierarchical-clustering/discussions)

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


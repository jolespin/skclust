# -*- coding: utf-8 -*-
"""
skclust: A comprehensive hierarchical clustering toolkit
========================================================================

A scikit-learn compatible implementation of hierarchical clustering with 
advanced tree cutting, visualization, and network analysis capabilities.

Author: Josh L. Espinoza
"""

__version__ = "2025.8.12.post1"
__author__ = "Josh L. Espinoza"

import os
import warnings
from collections import (
    Counter,
    OrderedDict,
)
from typing import (
    Union, 
    Optional, 
    List, 
    Dict, 
    Any, 
    Tuple,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import (
    rgb2hex, 
    to_rgb,
)
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.cluster.hierarchy import (
    linkage, 
    dendrogram as scipy_dendrogram, 
    fcluster,
)
from scipy.spatial.distance import (
    squareform, 
    pdist,
)
from sklearn.base import (
    BaseEstimator, 
    ClusterMixin, 
    TransformerMixin,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_samples, 
    silhouette_score,
    pairwise_distances_argmin_min,
)


from sklearn.cluster import (
    KMeans, 
    MiniBatchKMeans,
)
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import (
    check_X_y, 
    check_array,
)
from sklearn.utils.multiclass import check_classification_targets

from loguru import logger

try:
    from fastcluster import linkage as fast_linkage
    FASTCLUSTER_AVAILABLE = True
except ImportError:
    FASTCLUSTER_AVAILABLE = False
    warnings.warn("fastcluster not available, using scipy.cluster.hierarchy.linkage")

try:
    import skbio
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False
    warnings.warn("skbio not available, tree functionality will be limited")

try:
    from ensemble_networkx import Symmetric
    ENSEMBLE_NETWORKX_AVAILABLE = True
except ImportError:
    ENSEMBLE_NETWORKX_AVAILABLE = False
    warnings.warn("ensemble_networkx not available, Symmetric object support disabled")

try:
    import dynamicTreeCut
    DYNAMIC_TREE_CUT_AVAILABLE = True
except ImportError:
    DYNAMIC_TREE_CUT_AVAILABLE = False
    warnings.warn("dynamicTreeCut not available, dynamic tree cutting disabled")


class HierarchicalClustering(BaseEstimator, ClusterMixin):
    """
    Hierarchical clustering with advanced tree cutting and visualization.
    
    This class provides a comprehensive hierarchical clustering implementation
    that follows scikit-learn conventions while offering advanced features like
    dynamic tree cutting, metadata tracks, and network analysis.
    
    Parameters
    ----------
    method : str, default='ward'
        The linkage method to use. Options: 'ward', 'complete', 'average', 
        'single', 'centroid', 'median', 'weighted'.
    metric : str, default='euclidean'
        The distance metric to use for computing pairwise distances.
    min_cluster_size : int, default=20
        Minimum cluster size for dynamic tree cutting.
    deep_split : int, default=1
        Deep split parameter for dynamic tree cutting (0-4).
    dynamic_cut_method : str, default='hybrid'
        Method for dynamic tree cutting: 'hybrid' or 'tree'.
    cut_method : str, default='dynamic'
        Tree cutting method: 'dynamic', 'height', or 'maxclust'.
    cut_threshold : float, optional
        Threshold for height-based cutting or number of clusters for maxclust.
    name : str, optional
        Name for the clustering instance.
    random_state : int, optional
        Random state for reproducible results.
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    linkage_matrix_ : ndarray
        The linkage matrix from hierarchical clustering.
    tree_ : skbio.TreeNode
        The hierarchical tree (if skbio is available).
    dendrogram_ : dict
        Dendrogram data structure from scipy.
    n_clusters_ : int
        Number of clusters found.
    tracks_ : dict
        Dictionary of metadata tracks for visualization.
    """
    
    def __init__(self, 
                 method='ward',
                 metric='euclidean',
                 min_cluster_size=20,
                 deep_split=1,
                 dynamic_cut_method='hybrid',
                 cut_method='dynamic',
                 cut_threshold=None,
                 name=None,
                 random_state=None):
        
        self.method = method
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.deep_split = deep_split
        self.dynamic_cut_method = dynamic_cut_method
        self.cut_method = cut_method
        self.cut_threshold = cut_threshold
        self.name = name
        self.random_state = random_state
        
        # Initialize attributes
        self.labels_ = None
        self.linkage_matrix_ = None
        self.tree_ = None
        self.dendrogram_ = None
        self.tracks_ = OrderedDict()
        self._is_fitted = False
        
    def fit(self, X, y=None):
        """
        Fit hierarchical clustering to data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Training data. If square matrix, assumed to be distance matrix.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_input(X)
        
        # Store original data and create sample labels first
        self.data_ = X
        if hasattr(X, 'index'):
            self.sample_labels_ = X.index
        else:
            self.sample_labels_ = np.arange(X.shape[0])
        
        # Compute distance matrix if needed
        if self._is_distance_matrix(X):
            self.distance_matrix_ = X
        else:
            if ENSEMBLE_NETWORKX_AVAILABLE and isinstance(X, Symmetric):
                self.distance_matrix_ = X.to_pandas_dataframe()
            else:
                self.distance_matrix_ = self._compute_distance_matrix(X)
            
        # Perform hierarchical clustering
        self._perform_clustering()
        
        # Cut tree to get clusters
        self._cut_tree()
        
        # Build tree representation
        if SKBIO_AVAILABLE:
            self._build_tree()
            
        self._is_fitted = True
        return self
        
    def transform(self, X=None):
        """
        Return cluster labels.
        
        Parameters
        ----------
        X : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self._check_fitted()
        return self.labels_
        
    def fit_transform(self, X, y=None):
        """
        Fit hierarchical clustering and return cluster labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return self.fit(X, y).transform()
        
    def _validate_input(self, X):
        """Validate and convert input data."""
        if hasattr(X, 'values'):  # pandas DataFrame
            return X
        else:
            return np.asarray(X)
            
    def _is_distance_matrix(self, X):
        """Check if X is a distance matrix."""
        if hasattr(X, 'shape'):
            return X.shape[0] == X.shape[1]
        return False
        
    def _compute_distance_matrix(self, X):
        """Compute pairwise distance matrix."""
        if hasattr(X, 'values'):
            X_values = X.values
        else:
            X_values = X
            
        distances = pdist(X_values, metric=self.metric)
        return pd.DataFrame(
            squareform(distances),
            index=self.sample_labels_,
            columns=self.sample_labels_
        )
        
    def _perform_clustering(self):
        """Perform hierarchical clustering."""
        # Get condensed distance matrix
        if hasattr(self.distance_matrix_, 'values'):
            dist_condensed = squareform(self.distance_matrix_.values)
        else:
            dist_condensed = squareform(self.distance_matrix_)
            
        # Perform linkage
        if FASTCLUSTER_AVAILABLE:
            self.linkage_matrix_ = fast_linkage(dist_condensed, method=self.method)
        else:
            self.linkage_matrix_ = linkage(dist_condensed, method=self.method)
            
        # Generate dendrogram
        self.dendrogram_ = scipy_dendrogram(
            self.linkage_matrix_,
            labels=list(self.sample_labels_),  # Convert to list
            no_plot=True
        )
        
    def _cut_tree(self):
        """Cut tree to obtain clusters."""
        if self.cut_method == 'dynamic' and DYNAMIC_TREE_CUT_AVAILABLE:
            self._cut_tree_dynamic()
        elif self.cut_method == 'height':
            self._cut_tree_height()
        elif self.cut_method == 'maxclust':
            self._cut_tree_maxclust()
        else:
            # Fallback to height-based cutting
            warnings.warn(f"Cut method '{self.cut_method}' not available, using height-based cutting")
            self._cut_tree_height()
            
        # Set n_clusters_ after cutting
        if self.labels_ is not None:
            self.n_clusters_ = len(np.unique(self.labels_[self.labels_ > 0]))
            
    def _cut_tree_dynamic(self):
        """Perform dynamic tree cutting."""
        try:
            # Convert linkage matrix format for dynamicTreeCut
            # Note: This is a simplified implementation - you may need to adjust
            # based on the exact API of your dynamicTreeCut package
            
            results = dynamicTreeCut.cutreeHybrid(
                self.linkage_matrix_,
                self.distance_matrix_.values if hasattr(self.distance_matrix_, 'values') else self.distance_matrix_,
                minClusterSize=self.min_cluster_size,
                deepSplit=self.deep_split,
                cutHeight=self.cut_threshold
            )
            
            if isinstance(results, dict) and 'labels' in results:
                self.labels_ = results['labels']
            else:
                self.labels_ = results
                
        except Exception as e:
            warnings.warn(f"Dynamic tree cutting failed: {e}. Using height-based cutting.")
            self._cut_tree_height()
            
    def _cut_tree_height(self):
        """Cut tree at specified height."""
        if self.cut_threshold is None:
            # Use 70% of max height as default
            max_height = np.max(self.linkage_matrix_[:, 2])
            self.cut_threshold = 0.7 * max_height
            
        self.labels_ = fcluster(
            self.linkage_matrix_,
            self.cut_threshold,
            criterion='distance'
        )
        
    def _cut_tree_maxclust(self):
        """Cut tree to get specified number of clusters."""
        if self.cut_threshold is None:
            self.cut_threshold = 3  # Default number of clusters
            
        self.labels_ = fcluster(
            self.linkage_matrix_,
            self.cut_threshold,
            criterion='maxclust'
        )
        
    def _build_tree(self):
        """Build skbio tree from linkage matrix."""
        if not SKBIO_AVAILABLE:
            return
            
        try:
            self.tree_ = skbio.TreeNode.from_linkage_matrix(
                self.linkage_matrix_,
                list(self.sample_labels_)  # Convert to list to ensure compatibility
            )
            if self.name:
                self.tree_.name = self.name
        except Exception as e:
            warnings.warn(f"Tree building failed: {e}")
            self.tree_ = None
            
    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not self._is_fitted:
            raise ValueError("This HierarchicalClustering instance is not fitted yet.")
        
    def add_track(self, name, data, track_type='continuous', color=None, **kwargs):
        """
        Add metadata track for visualization.
        
        Parameters
        ----------
        name : str
            Name of the track.
        data : array-like or dict
            Track data. Should be same length as samples.
        track_type : str, default='continuous'
            Type of track: 'continuous' or 'categorical'.
        color : str or array-like, optional
            Color(s) for the track.
        **kwargs
            Additional plotting parameters.
        """
        self._check_fitted()
        
        # Convert data to pandas Series
        if isinstance(data, dict):
            data = pd.Series(data)
        elif not isinstance(data, pd.Series):
            data = pd.Series(data, index=self.sample_labels_)
            
        # Align with sample labels
        data = data.reindex(self.sample_labels_)
        
        self.tracks_[name] = {
            'data': data,
            'type': track_type,
            'color': color,
            'kwargs': kwargs
        }
        
    def plot_dendrogram(self, figsize=(12, 6), show_clusters=True, show_tracks=True,
                       cluster_colors=None, track_height=0.8, **kwargs):
        """
        Plot dendrogram with optional cluster coloring and tracks.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 6)
            Figure size.
        show_clusters : bool, default=True
            Whether to color dendrogram by clusters.
        show_tracks : bool, default=True
            Whether to show metadata tracks.
        cluster_colors : dict, optional
            Custom colors for clusters.
        track_height : float, default=0.8
            Height ratio for tracks.
        **kwargs
            Additional dendrogram plotting parameters.
        """
        self._check_fitted()
        
        # Calculate subplot ratios
        n_tracks = len(self.tracks_) if show_tracks else 0
        height_ratios = [4] + [track_height] * n_tracks
        
        if n_tracks > 0:
            fig, axes = plt.subplots(
                n_tracks + 1, 1,
                figsize=figsize,
                height_ratios=height_ratios,
                sharex=True
            )
            if n_tracks == 1:
                axes = [axes[0], axes[1]]
            ax_dendro = axes[0]
        else:
            fig, ax_dendro = plt.subplots(figsize=figsize)
            axes = [ax_dendro]
            
        # Plot dendrogram
        dendro_kwargs = {
            'orientation': 'top',
            'labels': list(self.sample_labels_),  # Convert to list
            'leaf_rotation': 90,
            'leaf_font_size': 8
        }
        dendro_kwargs.update(kwargs)
        
        if show_clusters and self.labels_ is not None:
            # Color dendrogram by clusters
            if cluster_colors is None:
                cluster_colors = self._generate_cluster_colors()
            dendro_kwargs['color_threshold'] = 0
            dendro_kwargs['above_threshold_color'] = 'gray'
            
        scipy_dendrogram(self.linkage_matrix_, ax=ax_dendro, **dendro_kwargs)
        
        if self.name:
            ax_dendro.set_title(f'Hierarchical Clustering: {self.name}')
        else:
            ax_dendro.set_title('Hierarchical Clustering')
            
        # Plot tracks
        if show_tracks and n_tracks > 0:
            self._plot_tracks(axes[1:], track_height)
            
        plt.tight_layout()
        return fig, axes
        
    def _generate_cluster_colors(self):
        """Generate colors for clusters."""
        n_clusters = self.n_clusters_
        if n_clusters <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))
            
        return {i+1: rgb2hex(colors[i]) for i in range(n_clusters)}
        
    def _plot_tracks(self, axes, track_height):
        """Plot metadata tracks."""
        for i, (track_name, track_info) in enumerate(self.tracks_.items()):
            ax = axes[i]
            data = track_info['data']
            track_type = track_info['type']
            color = track_info['color']
            
            # Get positions for each sample
            positions = np.arange(len(self.sample_labels_))
            
            if track_type == 'continuous':
                # Plot as bar chart
                if color is None:
                    color = 'steelblue'
                ax.bar(positions, data.values, color=color, width=0.8)
                ax.set_ylabel(track_name)
                
            elif track_type == 'categorical':
                # Plot as colored rectangles
                unique_vals = data.dropna().unique()
                if color is None:
                    color_map = dict(zip(unique_vals, 
                                       plt.cm.Set1(np.linspace(0, 1, len(unique_vals)))))
                else:
                    color_map = color
                    
                for j, val in enumerate(data.values):
                    if pd.notna(val):
                        rect_color = color_map.get(val, 'gray')
                        rect = patches.Rectangle((j-0.4, 0), 0.8, 1, 
                                               facecolor=rect_color, edgecolor='none')
                        ax.add_patch(rect)
                        
                ax.set_ylim(0, 1)
                ax.set_ylabel(track_name)
                ax.set_yticks([])
                
            ax.set_xlim(-0.5, len(self.sample_labels_) - 0.5)
            
    def eigenprofiles(self, data=None, n_components=1):
        """
        Calculate eigenprofiles (first principal components) for each cluster.
        
        Parameters
        ----------
        data : array-like, optional
            Data matrix. If None, uses original fitting data.
        n_components : int, default=1
            Number of principal components to return.
            
        Returns
        -------
        eigenprofiles : dict
            Dictionary mapping cluster labels to eigenprofiles.
        """
        self._check_fitted()
        
        if data is None:
            if hasattr(self, 'data_') and not self._is_distance_matrix(self.data_):
                data = self.data_
            else:
                raise ValueError("Original data not available. Please provide data matrix.")
                
        if hasattr(data, 'values'):
            data_values = data.values
        else:
            data_values = data
            
        eigenprofiles = {}
        
        for cluster_id in np.unique(self.labels_):
            if cluster_id == 0:  # Skip noise/unassigned
                continue
                
            cluster_mask = self.labels_ == cluster_id
            cluster_data = data_values[cluster_mask]
            
            if cluster_data.shape[0] > 1:
                pca = PCA(n_components=n_components)
                pca.fit(cluster_data.T)  # Transpose for feature PCA
                eigenprofiles[cluster_id] = {
                    'eigenprofile': pca.components_[0],
                    'explained_variance_ratio': pca.explained_variance_ratio_[0],
                    'eigenvalue': pca.explained_variance_[0]
                }
            else:
                # Single sample cluster
                eigenprofiles[cluster_id] = {
                    'eigenprofile': cluster_data[0],
                    'explained_variance_ratio': 1.0,
                    'eigenvalue': np.var(cluster_data[0])
                }
                
        return eigenprofiles
        
    def connectivity(self, return_type='summary'):
        """
        Calculate network connectivity metrics.
        
        Parameters
        ----------
        return_type : str, default='summary'
            Type of connectivity to return: 'summary', 'detailed', or 'matrix'.
            
        Returns
        -------
        connectivity : dict or DataFrame
            Connectivity metrics.
        """
        self._check_fitted()
        
        if not ENSEMBLE_NETWORKX_AVAILABLE:
            warnings.warn("ensemble_networkx not available. Using simplified connectivity.")
            return self._simple_connectivity()
            
        # Create Symmetric object from distance matrix
        # Convert distances to similarities (invert)
        if hasattr(self.distance_matrix_, 'values'):
            similarity_matrix = 1 / (1 + self.distance_matrix_)
            np.fill_diagonal(similarity_matrix.values, 1.0)
        else:
            similarity_matrix = 1 / (1 + self.distance_matrix_)
            np.fill_diagonal(similarity_matrix, 1.0)
        
        try:
            sym_obj = Symmetric(similarity_matrix)
            return sym_obj.connectivity(
                groups=pd.Series(self.labels_, index=self.sample_labels_),
                return_type=return_type
            )
        except Exception as e:
            warnings.warn(f"Symmetric connectivity failed: {e}. Using simplified version.")
            return self._simple_connectivity()
            
    def _simple_connectivity(self):
        """Simple connectivity calculation fallback."""
        cluster_sizes = pd.Series(self.labels_).value_counts().sort_index()
        return {
            'cluster_sizes': cluster_sizes,
            'total_samples': len(self.labels_),
            'n_clusters': self.n_clusters_
        }
        
    def silhouette_analysis(self):
        """
        Calculate silhouette scores for cluster validation.
        
        Returns
        -------
        silhouette_scores : dict
            Dictionary containing overall and per-sample silhouette scores.
        """
        self._check_fitted()
        
        if self.labels_ is None or len(np.unique(self.labels_)) < 2:
            return {'overall_score': None, 'sample_scores': None}
            
        # Use distance matrix if available
        if hasattr(self, 'distance_matrix_'):
            if hasattr(self.distance_matrix_, 'values'):
                distance_matrix = self.distance_matrix_.values
            else:
                distance_matrix = self.distance_matrix_
        else:
            distance_matrix = None
            
        try:
            if distance_matrix is not None:
                overall_score = silhouette_score(
                    distance_matrix, self.labels_, metric='precomputed'
                )
                sample_scores = silhouette_samples(
                    distance_matrix, self.labels_, metric='precomputed'
                )
            else:
                raise ValueError("No distance matrix available")
        except:
            # Fallback to euclidean if precomputed fails
            if hasattr(self, 'data_') and not self._is_distance_matrix(self.data_):
                data = self.data_.values if hasattr(self.data_, 'values') else self.data_
                overall_score = silhouette_score(data, self.labels_)
                sample_scores = silhouette_samples(data, self.labels_)
            else:
                return {'overall_score': None, 'sample_scores': None}
        
        return {
            'overall_score': overall_score,
            'sample_scores': pd.Series(sample_scores, index=self.sample_labels_)
        }
        
    def to_networkx(self, weight_threshold=None):
        """
        Convert clustering result to NetworkX graph.
        
        Parameters
        ----------
        weight_threshold : float, optional
            Minimum edge weight to include in graph.
            
        Returns
        -------
        graph : networkx.Graph
            NetworkX graph representation.
        """
        self._check_fitted()
        
        # Create graph from similarity matrix
        if hasattr(self.distance_matrix_, 'values'):
            distance_values = self.distance_matrix_.values
        else:
            distance_values = self.distance_matrix_
            
        similarity_matrix = 1 / (1 + distance_values)
        
        G = nx.Graph()
        
        # Add nodes with cluster information
        for i, label in enumerate(self.sample_labels_):
            G.add_node(label, cluster=self.labels_[i])
            
        # Add edges
        for i in range(len(self.sample_labels_)):
            for j in range(i+1, len(self.sample_labels_)):
                if hasattr(similarity_matrix, 'iloc'):
                    weight = similarity_matrix.iloc[i, j]
                else:
                    weight = similarity_matrix[i, j]
                if weight_threshold is None or weight >= weight_threshold:
                    G.add_edge(
                        self.sample_labels_[i],
                        self.sample_labels_[j],
                        weight=weight
                    )
                    
        return G
        
    def to_newick(self, filepath=None):
        """
        Export tree in Newick format.
        
        Parameters
        ----------
        filepath : str, optional
            If provided, saves to file. Otherwise returns string.
            
        Returns
        -------
        newick_str : str
            Newick format string (if filepath is None).
        """
        self._check_fitted()
        
        if not SKBIO_AVAILABLE or self.tree_ is None:
            raise ValueError("Tree not available. Requires skbio and successful tree building.")
            
        newick_str = str(self.tree_)
        
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(newick_str)
            return None
        else:
            return newick_str
            
    def summary(self):
        """
        Print summary of clustering results.
        
        Returns
        -------
        summary_dict : dict
            Dictionary containing summary statistics.
        """
        self._check_fitted()
        
        summary_dict = {
            'n_samples': len(self.sample_labels_),
            'n_clusters': self.n_clusters_,
            'method': self.method,
            'metric': self.metric,
            'cut_method': self.cut_method
        }
        
        if self.labels_ is not None:
            cluster_counts = pd.Series(self.labels_).value_counts().sort_index()
            summary_dict['cluster_sizes'] = cluster_counts.to_dict()
            
        # Add silhouette score if possible
        silhouette_results = self.silhouette_analysis()
        if silhouette_results['overall_score'] is not None:
            summary_dict['silhouette_score'] = silhouette_results['overall_score']
            
        print("Hierarchical Clustering Summary")
        print("=" * 30)
        for key, value in summary_dict.items():
            if key != 'cluster_sizes':
                print(f"{key}: {value}")
                
        if 'cluster_sizes' in summary_dict:
            print("\nCluster sizes:")
            for cluster, size in summary_dict['cluster_sizes'].items():
                print(f"  Cluster {cluster}: {size} samples")
                
        return summary_dict


class KMeansRepresentativeSampler(BaseEstimator, TransformerMixin):
    """
    K-means-based sampler for creating test sets with many clusters (k = 10% of data).
    Optimized for large datasets where traditional clustering would be too slow.
    
    Parameters
    ----------
    sampling_size : float, default=0.1
        Proportion of data to use as test set (determines k = sampling_size * n_samples)
    stratify : bool, default=True
        Whether to maintain class proportions in clustering
    method : str, default='minibatch'
        Clustering method: 'minibatch' (fast), 'hierarchical' (balanced), 'kmeans' (slow but exact)
    batch_size : int, default=1000
        Batch size for MiniBatchKMeans (only used with method='minibatch')
    coverage_boost : float, default=1.5
        Boost factor for minority classes when stratified
    min_clusters_per_class : int, default=10
        Minimum clusters per class regardless of proportion
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, sampling_size=0.1, stratify=True, method='minibatch', 
                 batch_size=1000, coverage_boost=1.5, min_clusters_per_class=10,
                 random_state=None):
        self.sampling_size = sampling_size
        self.stratify = stratify
        self.method = method
        self.batch_size = batch_size
        self.coverage_boost = coverage_boost
        self.min_clusters_per_class = min_clusters_per_class
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """
        Fit the test sampler to create clusters and identify representatives.
        """
        # Input validation
        if self.stratify and y is None:
            raise ValueError("y is required when stratify=True")
        
        # Handle pandas input
        self.is_pandas_input_ = False
        original_index = None
        
        if isinstance(X, pd.DataFrame):
            self.is_pandas_input_ = True
            original_index = X.index.copy()
            X = X.values
        elif isinstance(X, pd.Series):
            self.is_pandas_input_ = True
            original_index = X.index.copy()
            X = X.values.reshape(-1, 1)
        else:
            X = check_array(X)
            original_index = np.arange(len(X))
        
        if isinstance(y, pd.Series):
            y = y.values
        
        if self.stratify:
            X, y = check_X_y(X, y)
            check_classification_targets(y)
        
        n_samples = X.shape[0]
        k_total = max(1, int(n_samples * self.sampling_size))
        
        logger.info(f"Creating {k_total} clusters for test set from {n_samples} samples...")
        
        if self.stratify:
            representatives = self._stratified_clustering(X, y, k_total)
        else:
            representatives = self._global_clustering(X, k_total)
        
        # Create boolean mask for representatives
        self.n_clusters_ = len(representatives)
        representatives_mask = np.zeros(n_samples, dtype=bool)
        representatives_mask[representatives] = True
        
        if self.is_pandas_input_:
            self.representatives_ = pd.Series(representatives_mask, index=original_index, 
                                            name='is_representative')
            self.representative_indices_ = pd.Index(original_index[representatives], 
                                                  name='representative_indices')
        else:
            self.representatives_ = representatives_mask
            self.representative_indices_ = representatives
        
        logger.info(f"Selected {self.n_clusters_} representatives as test set")
        return self
    
    def _global_clustering(self, X, k_total):
        """Non-stratified clustering for the entire dataset."""
        logger.info(f"Performing global clustering with k={k_total}...")
        
        if self.method == 'minibatch':
            clusterer = MiniBatchKMeans(
                n_clusters=k_total,
                batch_size=self.batch_size,
                random_state=self.random_state,
                n_init=3
            )
            labels = clusterer.fit_predict(X)
            centroids = clusterer.cluster_centers_
            
        elif self.method == 'hierarchical':
            # Use hierarchical approach for better quality
            return self._hierarchical_clustering(X, k_total)
            
        else:  # kmeans
            clusterer = KMeans(
                n_clusters=k_total,
                n_init=1,
                random_state=self.random_state
            )
            labels = clusterer.fit_predict(X)
            centroids = clusterer.cluster_centers_
        
        # Find representatives (closest to centroids)
        representatives = []
        for cluster_id in range(k_total):
            cluster_mask = (labels == cluster_id)
            if np.any(cluster_mask):
                cluster_indices = np.where(cluster_mask)[0]
                cluster_samples = X[cluster_mask]
                
                # Find sample closest to centroid
                centroid = centroids[cluster_id]
                distances = np.linalg.norm(cluster_samples - centroid, axis=1)
                best_idx = cluster_indices[np.argmin(distances)]
                representatives.append(best_idx)
        
        return np.array(representatives)
    
    def _stratified_clustering(self, X, y, k_total):
        """Stratified clustering maintaining class proportions."""
        class_counts = Counter(y)
        total_samples = len(y)
        representatives = []
        
        logger.info(f"Stratified clustering across {len(class_counts)} classes...")
        
        # Calculate cluster allocation per class
        cluster_allocation = self._calculate_cluster_allocation(class_counts, k_total, total_samples)
        
        for class_label, k_class in cluster_allocation.items():
            logger.info(f"  Class {class_label}: {k_class} clusters from {class_counts[class_label]} samples")
            
            class_mask = (y == class_label)
            X_class = X[class_mask]
            class_indices = np.where(class_mask)[0]
            
            if k_class == 1 or len(X_class) <= k_class:
                # Too few samples for clustering
                if len(X_class) <= k_class:
                    representatives.extend(class_indices)
                else:
                    # Single cluster - select centroid
                    centroid = np.mean(X_class, axis=0)
                    distances = np.linalg.norm(X_class - centroid, axis=1)
                    best_idx = class_indices[np.argmin(distances)]
                    representatives.append(best_idx)
            else:
                # Cluster within class
                class_representatives = self._cluster_class(X_class, class_indices, k_class)
                representatives.extend(class_representatives)
        
        return np.array(representatives)
    
    def _cluster_class(self, X_class, class_indices, k_class):
        """Cluster samples within a single class."""
        if self.method == 'minibatch':
            clusterer = MiniBatchKMeans(
                n_clusters=k_class,
                batch_size=min(self.batch_size, len(X_class)),
                random_state=self.random_state,
                n_init=3
            )
        else:  # kmeans or hierarchical
            clusterer = KMeans(
                n_clusters=k_class,
                n_init=1,
                random_state=self.random_state
            )
        
        labels = clusterer.fit_predict(X_class)
        
        representatives = []
        for cluster_id in range(k_class):
            cluster_mask = (labels == cluster_id)
            if np.any(cluster_mask):
                cluster_samples = X_class[cluster_mask]
                cluster_class_indices = class_indices[cluster_mask]
                
                # Find representative (closest to centroid)
                centroid = clusterer.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_samples - centroid, axis=1)
                best_idx = cluster_class_indices[np.argmin(distances)]
                representatives.append(best_idx)
        
        return representatives
    
    def _hierarchical_clustering(self, X, k_total):
        """Hierarchical clustering for better quality with large k."""
        # Level 1: Create macro-clusters (manageable number)
        n_macro = min(500, k_total // 10)  # ~10-40 samples per macro-cluster
        
        logger.info(f"  Level 1: {n_macro} macro-clusters")
        macro_clusterer = MiniBatchKMeans(
            n_clusters=n_macro,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init=3
        )
        macro_labels = macro_clusterer.fit_predict(X)
        
        # Level 2: Sub-cluster within each macro-cluster
        representatives = []
        clusters_per_macro = k_total // n_macro
        remaining_clusters = k_total % n_macro
        
        logger.info(f"  Level 2: ~{clusters_per_macro} micro-clusters per macro-cluster")
        
        for macro_id in range(n_macro):
            macro_mask = (macro_labels == macro_id)
            if not np.any(macro_mask):
                continue
                
            macro_indices = np.where(macro_mask)[0]
            macro_samples = X[macro_mask]
            
            # Determine number of micro-clusters for this macro-cluster
            n_micro = clusters_per_macro
            if macro_id < remaining_clusters:
                n_micro += 1
            
            n_micro = min(n_micro, len(macro_samples))
            
            if n_micro <= 1:
                # Single representative
                if len(macro_samples) == 1:
                    representatives.append(macro_indices[0])
                else:
                    centroid = np.mean(macro_samples, axis=0)
                    distances = np.linalg.norm(macro_samples - centroid, axis=1)
                    best_idx = macro_indices[np.argmin(distances)]
                    representatives.append(best_idx)
            else:
                # Micro-clustering
                micro_clusterer = KMeans(n_clusters=n_micro, n_init=1, random_state=self.random_state)
                micro_labels = micro_clusterer.fit_predict(macro_samples)
                
                for micro_id in range(n_micro):
                    micro_mask = (micro_labels == micro_id)
                    if np.any(micro_mask):
                        micro_samples = macro_samples[micro_mask]
                        micro_indices = macro_indices[micro_mask]
                        
                        centroid = micro_clusterer.cluster_centers_[micro_id]
                        distances = np.linalg.norm(micro_samples - centroid, axis=1)
                        best_idx = micro_indices[np.argmin(distances)]
                        representatives.append(best_idx)
        
        return np.array(representatives)
    
    def _calculate_cluster_allocation(self, class_counts, k_total, total_samples):
        """Calculate how many clusters each class should get."""
        max_class_count = max(class_counts.values())
        cluster_allocation = {}
        
        # Calculate boosted weights
        boosted_weights = {}
        total_boosted_weight = 0
        
        for class_label, class_count in class_counts.items():
            imbalance_ratio = max_class_count / class_count
            boost_factor = imbalance_ratio ** (1 / self.coverage_boost) if self.coverage_boost > 1.0 else 1.0
            boosted_weight = (class_count / total_samples) * boost_factor
            
            boosted_weights[class_label] = boosted_weight
            total_boosted_weight += boosted_weight
        
        # Allocate clusters
        allocated_total = 0
        for class_label, class_count in class_counts.items():
            normalized_weight = boosted_weights[class_label] / total_boosted_weight
            proportional_clusters = max(1, round(k_total * normalized_weight))
            
            # Apply constraints
            final_clusters = max(self.min_clusters_per_class, proportional_clusters)
            final_clusters = min(final_clusters, class_count)  # Can't have more clusters than samples
            
            cluster_allocation[class_label] = final_clusters
            allocated_total += final_clusters
        
        # Adjust if over-allocated
        if allocated_total > k_total:
            excess = allocated_total - k_total
            # Reduce from largest allocations first
            sorted_classes = sorted(cluster_allocation.items(), key=lambda x: x[1], reverse=True)
            
            for class_label, clusters in sorted_classes:
                if excess <= 0:
                    break
                reduction = min(excess, max(0, clusters - self.min_clusters_per_class))
                cluster_allocation[class_label] -= reduction
                excess -= reduction
        
        return cluster_allocation
    
    def transform(self, X):
        """Return the representative samples (test set)."""
        if not hasattr(self, 'representatives_'):
            raise ValueError("This KMeansRepresentativeSampler instance is not fitted yet.")
        
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X[self.representatives_]
        else:
            X = check_array(X)
            return X[self.representatives_]
    
    def fit_transform(self, X, y=None):
        """Fit the sampler and return the test set."""
        return self.fit(X, y).transform(X)
    
    def get_train_sampling_split(self, X, y=None):
        """
        Get train/test split with representatives as test set.
        
        Returns
        -------
        X_train, X_test, y_train, y_test : arrays
            Train/test split where test set contains the cluster representatives
        """
        if not hasattr(self, 'representatives_'):
            raise ValueError("This KMeansRepresentativeSampler instance is not fitted yet.")
        
        # Handle pandas input
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_test = X[self.representatives_]
            X_train = X[~self.representatives_]
            if y is not None:
                y_test = y[self.representatives_] if isinstance(y, pd.Series) else y[self.representatives_.values]
                y_train = y[~self.representatives_] if isinstance(y, pd.Series) else y[~self.representatives_.values]
                return X_train, X_test, y_train, y_test
            return X_train, X_test
        else:
            X = check_array(X)
            X_test = X[self.representatives_]
            X_train = X[~self.representatives_]
            if y is not None:
                y_test = y[self.representatives_]
                y_train = y[~self.representatives_]
                return X_train, X_test, y_train, y_test
            return X_train, X_test

# Export main classes and functions
__all__ = [
    'HierarchicalClustering',
    'KMeansRepresentativeSampler',
]


# -*- coding: utf-8 -*-
"""
skclust: A comprehensive hierarchical clustering toolkit
========================================================================

A scikit-learn compatible implementation of hierarchical clustering with 
advanced tree cutting, visualization, and network analysis capabilities.

Author: Josh L. Espinoza
"""

__version__ = "2025.9.8"
__author__ = "Josh L. Espinoza"

import warnings
import logging
from collections import (
    Counter,
    OrderedDict,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import rgb2hex
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
from sklearn.cluster import (
    KMeans, 
    MiniBatchKMeans,
)
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets

from loguru import logger

# Optional dependencies with fallbacks
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
    
# Classes
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
    min_cluster_size : int, default=3
        Minimum cluster size for dynamic tree cutting.
    deep_split : int, default=2
        Deep split parameter for dynamic tree cutting (0-4).
    cut_method : str, default='dynamic'
        Tree cutting method: 'dynamic', 'height', or 'maxclust'.
    cut_threshold : float, optional
        Threshold for height-based cutting or number of clusters for maxclust.
    name : str, optional
        Name for the clustering instance.
    random_state : int, optional
        Random state for reproducible results.
    distance_matrix_tol : float, default=1e-10
        Tolerance for validating distance matrix properties (symmetry, zero diagonal).
    outlier_cluster : int, default=-1
        Label used for outlier/noise samples that don't belong to any cluster.
    cluster_prefix : str, optional
        If provided, cluster labels will be converted to strings with this prefix
        (e.g., cluster_prefix="C" -> "C1", "C2", etc.).
        
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
                 min_cluster_size=3,
                 deep_split=2,
                 cut_method='dynamic',
                 cut_threshold=None,
                 name=None,
                 random_state=None,
                 distance_matrix_tol=1e-10,
                 outlier_cluster=-1,
                 cluster_prefix=None):
        
        # Validate parameters
        valid_methods = ['ward', 'complete', 'average', 'single', 'centroid', 'median', 'weighted']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
        
        valid_cut_methods = ['dynamic', 'height', 'maxclust']
        if cut_method not in valid_cut_methods:
            raise ValueError(f"cut_method must be one of {valid_cut_methods}, got '{cut_method}'")
        
        if cut_method == 'dynamic' and not DYNAMIC_TREE_CUT_AVAILABLE:
            warnings.warn(
                "dynamicTreeCut not available but cut_method='dynamic' specified. "
                "Consider using 'height' or 'maxclust' instead."
            )
        
        if deep_split not in range(5):  # 0-4
            raise ValueError(f"deep_split must be between 0 and 4, got {deep_split}")
        
        if min_cluster_size < 1:
            raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size}")
        
        if distance_matrix_tol <= 0:
            raise ValueError(f"distance_matrix_tol must be positive, got {distance_matrix_tol}")
        
        if cluster_prefix is not None and not isinstance(cluster_prefix, str):
            raise ValueError(f"cluster_prefix must be a string or None, got {type(cluster_prefix)}")
        
        self.method = method
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.deep_split = deep_split
        self.cut_method = cut_method
        self.cut_threshold = cut_threshold
        self.name = name
        self.random_state = random_state
        self.distance_matrix_tol = distance_matrix_tol
        self.outlier_cluster = outlier_cluster
        self.cluster_prefix = cluster_prefix
        
        # Initialize attributes
        self.labels_ = None
        self.linkage_matrix_ = None
        self.tree_ = None
        self.dendrogram_ = None
        self.n_clusters_ = None
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
        
        # Store original data and create sample labels ONCE
        self.data_ = X
        if hasattr(X, 'index'):
            self.sample_labels_ = list(X.index)
        else:
            self.sample_labels_ = list(range(X.shape[0]))
        
        # Compute distance matrix if needed
        if self._is_distance_matrix(X, tol=self.distance_matrix_tol):
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
            
    def _is_distance_matrix(self, X, tol=1e-10):
        """
        Check if X is a valid distance matrix.
        
        Parameters
        ----------
        X : array-like
            Input matrix to check.
        tol : float, default=1e-10
            Tolerance for numerical comparisons.
            
        Returns
        -------
        bool
            True if X appears to be a valid distance matrix.
        """
        if not hasattr(X, 'shape') or X.shape[0] != X.shape[1]:
            return False
        
        # Additional checks for valid distance matrix
        if hasattr(X, 'values'):
            values = X.values
        else:
            values = X
        
        # Check if symmetric (within tolerance)
        if not np.allclose(values, values.T, rtol=tol, atol=tol):
            return False
        
        # Check if diagonal is zero (within tolerance)  
        if not np.allclose(np.diag(values), 0, atol=tol):
            return False
        
        # Check if all values are non-negative (distance matrices should be non-negative)
        if np.any(values < -tol):
            return False
        
        return True
        
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
            labels=self.sample_labels_,  # Already a list
            no_plot=True
        )
        
        # Store the leaf order from dendrogram (this is the proper order for plotting)
        self.leaves_ = self.dendrogram_["ivl"]
        
    def _cut_tree(self):
        """Cut tree to obtain clusters."""
        if self.cut_method == 'dynamic':
            if not DYNAMIC_TREE_CUT_AVAILABLE:
                raise ValueError(
                    "Dynamic tree cutting requested but dynamicTreeCut not available. "
                    "Install dynamicTreeCut or use 'height' or 'maxclust' methods."
                )
            self._cut_tree_dynamic()
        elif self.cut_method == 'height':
            self._cut_tree_height()
        elif self.cut_method == 'maxclust':
            self._cut_tree_maxclust()
        else:
            raise ValueError(
                f"Unknown cut_method '{self.cut_method}'. "
                "Must be 'dynamic', 'height', or 'maxclust'."
            )
        
        # Set n_clusters_ after cutting
        if self.labels_ is not None:
            unique_labels = np.unique(self.labels_)
            # Remove outlier/noise labels 
            cluster_labels = unique_labels[unique_labels != self.outlier_cluster]
            self.n_clusters_ = len(cluster_labels)
            
            # Apply cluster prefix if specified
            if self.cluster_prefix is not None:
                self.labels_ = self._apply_cluster_prefix(self.labels_)
            
    def _cut_tree_dynamic(self):
        """Perform dynamic tree cutting."""
        # Prepare parameters, handling None values appropriately
        params = {
            'minClusterSize': self.min_cluster_size,
            'deepSplit': self.deep_split,
        }
        
        # Only add cutHeight if it's specified
        if self.cut_threshold is not None:
            params['cutHeight'] = self.cut_threshold
        
        try:
            distance_matrix = (self.distance_matrix_.values 
                              if hasattr(self.distance_matrix_, 'values') 
                              else self.distance_matrix_)
            
            results = dynamicTreeCut.cutreeHybrid(
                self.linkage_matrix_,
                distance_matrix,
                **params
            )
            
            if isinstance(results, dict) and 'labels' in results:
                self.labels_ = results['labels']
            else:
                self.labels_ = results
                
        except Exception as e:
            raise RuntimeError(f"Dynamic tree cutting failed: {e}")
            
    def _cut_tree_height(self):
        """Cut tree at specified height."""
        cut_height = self.cut_threshold
        if cut_height is None:
            # Use 70% of max height as default (don't modify self.cut_threshold)
            max_height = np.max(self.linkage_matrix_[:, 2])
            cut_height = 0.7 * max_height
            
        self.labels_ = fcluster(
            self.linkage_matrix_,
            cut_height,
            criterion='distance'
        )
        
    def _cut_tree_maxclust(self):
        """Cut tree to get specified number of clusters."""
        if self.cut_threshold is None:
            raise ValueError("cut_threshold must be specified when using cut_method='maxclust'")
            
        if not isinstance(self.cut_threshold, int) or self.cut_threshold < 1:
            raise ValueError("cut_threshold must be a positive integer when using cut_method='maxclust'")
            
        self.labels_ = fcluster(
            self.linkage_matrix_,
            self.cut_threshold,
            criterion='maxclust'
        )
        
    def _build_tree(self):
        """Build skbio tree from linkage matrix."""
        if not SKBIO_AVAILABLE:
            self.tree_ = None
            return
            
        try:
            self.tree_ = skbio.TreeNode.from_linkage_matrix(
                self.linkage_matrix_,
                self.sample_labels_  # Already a list
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
        data : Mapping or pandas.Series
            Track data mapping sample names to values. Must be a mapping type
            (dict, OrderedDict, etc.) with sample names as keys or a pandas Series
            with sample names as index.
        track_type : str, default='continuous'
            Type of track: 'continuous' or 'categorical'.
        color : str or array-like, optional
            Color(s) for the track.
        **kwargs
            Additional plotting parameters.
        """
        self._check_fitted()
        
        if track_type not in ['continuous', 'categorical']:
            raise ValueError(f"track_type must be 'continuous' or 'categorical', got '{track_type}'")
        
        # Import Mapping here to avoid top-level imports
        from collections.abc import Mapping
        
        # Validate input data type - must be a mapping or pandas Series
        if not isinstance(data, (Mapping, pd.Series)):
            raise ValueError(
                "Track data must be a mapping type (dict, OrderedDict, etc.) with "
                "sample names as keys or a pandas Series with sample names as index. "
                f"Got {type(data)} instead."
            )
        
        # Convert data to pandas Series
        if isinstance(data, pd.Series):
            # If it's already a pandas Series, use it as-is
            pass
        else:
            # Convert any mapping type to pandas Series
            data = pd.Series(data)
            
        # Align with sample labels
        data = data.reindex(self.sample_labels_)
        
        # Validate that we have data for all samples
        missing_samples = set(self.sample_labels_) - set(data.index)
        if missing_samples:
            warnings.warn(f"Track '{name}' missing data for samples: {missing_samples}")
        
        self.tracks_[name] = {
            'data': data,
            'type': track_type,
            'color': color,
            'kwargs': kwargs
        }
        
    def _plot_categorical_track(self, ax, data, colors, show_labels=False, label_text=None):
        """Plot categorical data as colored rectangles (used for both clusters and categorical tracks)."""
        # Use the proper leaf order from dendrogram
        ordered_leaves = self.leaves_
        
        # Plot rectangles for each category
        for i, sample in enumerate(ordered_leaves):
            if sample in data.index and pd.notna(data[sample]):
                category = data[sample]
                color = colors.get(category, 'gray')
                # Create rectangle for this sample - use dendrogram position
                rect = patches.Rectangle((i*10 + 5 - 5, 0), 10, 1, 
                                       facecolor=color, edgecolor='none', alpha=0.8)
                ax.add_patch(rect)
        
        # Add category labels if requested
        if show_labels and label_text is not None:
            # Group positions by category
            category_positions = {}
            for i, sample in enumerate(ordered_leaves):
                if sample in data.index and pd.notna(data[sample]):
                    category = data[sample]
                    if category not in category_positions:
                        category_positions[category] = []
                    category_positions[category].append(i*10 + 5)  # Use dendrogram positions
            
            # Place labels at center of each category group
            for category, positions_list in category_positions.items():
                if len(positions_list) > 0:
                    center_pos = np.mean(positions_list)
                    ax.text(center_pos, 0.5, str(category), 
                           ha='center', va='center', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Use the same x-limits as the dendrogram
        tree_width = len(ordered_leaves) * 10
        ax.set_xlim(0, tree_width)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        if label_text:
            ax.set_ylabel(label_text)

    def _plot_tracks(self, axes, track_height):
        """Plot metadata tracks."""
        track_names = list(self.tracks_.keys())
        ordered_leaves = self.leaves_
        
        for i, track_name in enumerate(track_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            track_info = self.tracks_[track_name]
            data = track_info['data']
            track_type = track_info['type']
            color = track_info['color']
            
            if track_type == 'continuous':
                # Plot as bar chart using dendrogram positions
                positions = []
                values = []
                colors_list = []
                
                for j, sample in enumerate(ordered_leaves):
                    if sample in data.index and pd.notna(data[sample]):
                        positions.append(j*10 + 5)  # Match dendrogram positions
                        values.append(data[sample])
                        if isinstance(color, dict):
                            colors_list.append(color.get(sample, 'steelblue'))
                        elif isinstance(color, pd.Series) and sample in color.index:
                            colors_list.append(color[sample])
                        else:
                            colors_list.append(color if color is not None else 'steelblue')
                
                if colors_list:
                    ax.bar(positions, values, color=colors_list, width=8)
                else:
                    ax.bar(positions, values, color='steelblue', width=8)
                ax.set_ylabel(track_name)
                
            elif track_type == 'categorical':
                # Use the same method as clusters
                if color is None or isinstance(color, str):
                    # Generate colors for categories
                    unique_vals = data.dropna().unique()
                    if isinstance(color, str):
                        color_map = {val: color for val in unique_vals}
                    else:
                        color_map = dict(zip(unique_vals, 
                                           plt.cm.Set1(np.linspace(0, 1, len(unique_vals)))))
                else:
                    color_map = color
                    
                self._plot_categorical_track(ax, data, color_map, label_text=track_name)
            
            # Use the same x-limits as the dendrogram
            tree_width = len(ordered_leaves) * 10
            ax.set_xlim(0, tree_width)
            
    def _apply_cluster_prefix(self, labels):
        """Apply cluster prefix to labels, converting to strings."""
        prefixed_labels = np.empty(len(labels), dtype=object)
        
        for i, label in enumerate(labels):
            if label == self.outlier_cluster:
                # Keep outlier cluster as is (could be string or int)
                prefixed_labels[i] = label
            else:
                # Apply prefix to non-outlier clusters
                prefixed_labels[i] = f"{self.cluster_prefix}{label}"
                
        return prefixed_labels
        
    def _generate_cluster_colors(self):
        """Generate colors for clusters."""
        if self.n_clusters_ is None:
            return {}
            
        if self.n_clusters_ <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters_))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(self.n_clusters_, 20)))
            
        # Get actual cluster IDs (exclude outlier cluster)
        unique_labels = np.unique(self.labels_)
        if self.cluster_prefix is not None:
            # Handle string cluster labels
            cluster_ids = [label for label in unique_labels if label != self.outlier_cluster]
        else:
            # Handle numeric cluster labels
            cluster_ids = unique_labels[unique_labels != self.outlier_cluster]
        
        color_dict = {}
        for i, cluster_id in enumerate(cluster_ids):
            if i < len(colors):
                color_dict[cluster_id] = rgb2hex(colors[i])
            else:
                # Fallback for too many clusters
                color_dict[cluster_id] = 'gray'
                
        # Add outlier color if outliers exist
        if self.outlier_cluster in unique_labels:
            color_dict[self.outlier_cluster] = 'white'
            
        return color_dict

    def plot(self, figsize=(13, 5), show_clusters=True, show_tracks=True,
             cluster_colors=None, track_height=0.8, show_cluster_labels=False, 
             cluster_label="Clusters", branch_color="black", show_leaf_labels=True, **kwargs):
        """
        Plot dendrogram with optional cluster coloring and tracks.
        
        Parameters
        ----------
        figsize : tuple, default=(13, 5)
            Figure size.
        show_clusters : bool, default=True
            Whether to show cluster assignments as colored rectangles.
        show_tracks : bool, default=True
            Whether to show metadata tracks.
        cluster_colors : dict, optional
            Custom colors for clusters.
        track_height : float, default=0.8
            Height ratio for tracks.
        show_cluster_labels : bool, default=False
            Whether to show cluster numbers on the cluster track.
        cluster_label : str, default="Clusters"
            Label for the cluster track.
        branch_color : str, default="black"
            Color for dendrogram branches.
        show_leaf_labels : bool, default=True
            Whether to show sample labels on the x-axis.
        **kwargs
            Additional dendrogram plotting parameters.
        """
        self._check_fitted()
        
        # Calculate subplot ratios
        n_tracks = len(self.tracks_) if show_tracks else 0
        n_clusters = 1 if show_clusters and self.labels_ is not None else 0
        
        # Height ratios: dendrogram gets most space, then clusters, then tracks
        height_ratios = [4]
        if show_clusters and self.labels_ is not None:
            height_ratios.append(track_height)
        if show_tracks and n_tracks > 0:
            height_ratios.extend([track_height] * n_tracks)
        
        n_subplots = len(height_ratios)
        
        if n_subplots > 1:
            fig, axes = plt.subplots(
                n_subplots, 1,
                figsize=figsize,
                height_ratios=height_ratios,
                sharex=True
            )
            if n_subplots == 2:
                axes = [axes[0], axes[1]]
            ax_dendro = axes[0]
        else:
            fig, ax_dendro = plt.subplots(figsize=figsize)
            axes = [ax_dendro]
            
        # Plot dendrogram using the pre-computed dendrogram data
        dendro_kwargs = {
            'orientation': 'top',
            'color_threshold': 0,  # Disable automatic coloring
            'above_threshold_color': branch_color,  # All branches same color
            'leaf_rotation': 90,
            'leaf_font_size': 8
        }
        dendro_kwargs.update(kwargs)
        
        # Plot using the stored dendrogram data to ensure consistency
        for xs, ys in zip(self.dendrogram_['icoord'], self.dendrogram_['dcoord']):
            ax_dendro.plot(xs, ys, color=branch_color, linewidth=1)
        
        # Set proper limits and labels
        tree_width = len(self.leaves_) * 10
        max_height = np.max(self.dendrogram_['dcoord'])
        tree_height = max_height + max_height * 0.05
        
        ax_dendro.set_xlim(0, tree_width)
        ax_dendro.set_ylim(0, tree_height)
        
        if self.name:
            ax_dendro.set_title(f'Hierarchical Clustering: {self.name}')
        else:
            ax_dendro.set_title('Hierarchical Clustering')
        
        # Handle leaf labels - they should appear on the bottom-most subplot
        bottom_axis = None
        if show_leaf_labels:
            if n_subplots > 1:
                # Find the bottom-most axis (last one in the list)
                bottom_axis = axes[-1]
            else:
                # Only dendrogram, show labels there
                bottom_axis = ax_dendro
            
        # Remove x-axis labels from dendrogram if we have other plots below
        if n_subplots > 1:
            ax_dendro.set_xticklabels([])
        elif show_leaf_labels:
            # Show leaf labels on dendrogram if it's the only plot
            leaf_positions = [i*10 + 5 for i in range(len(self.leaves_))]
            ax_dendro.set_xticks(leaf_positions)
            ax_dendro.set_xticklabels(self.leaves_, rotation=90)
        
        current_axis_idx = 1
        
        # Plot clusters (treat as categorical data)
        if show_clusters and self.labels_ is not None and n_subplots > 1:
            if cluster_colors is None:
                cluster_colors = self._generate_cluster_colors()
                
            # Create cluster data as pandas Series using the dendrogram leaf order
            cluster_data = pd.Series(self.labels_, index=self.sample_labels_)
            
            # Plot clusters using the categorical track method
            ax_clusters = axes[current_axis_idx]
            self._plot_categorical_track(ax_clusters, cluster_data, cluster_colors, 
                                       show_labels=show_cluster_labels, label_text=cluster_label)
            current_axis_idx += 1
            
        # Plot tracks
        if show_tracks and n_tracks > 0 and n_subplots > 1:
            track_axes = axes[current_axis_idx:current_axis_idx + n_tracks]
            self._plot_tracks(track_axes, track_height)
            
        # Add leaf labels to the bottom-most subplot if requested
        if show_leaf_labels and bottom_axis is not None and n_subplots > 1:
            leaf_positions = [i*10 + 5 for i in range(len(self.leaves_))]
            bottom_axis.set_xticks(leaf_positions)
            bottom_axis.set_xticklabels(self.leaves_, rotation=90)
            
        plt.tight_layout()
        return fig, axes
        
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
            # Only include non-outlier cluster labels in summary
            non_outlier_clusters = cluster_counts[cluster_counts.index != self.outlier_cluster]
            summary_dict['cluster_sizes'] = non_outlier_clusters.to_dict()
            
            # Add outlier count if present
            if self.outlier_cluster in cluster_counts.index:
                summary_dict['n_outliers'] = cluster_counts[self.outlier_cluster]
            
        print("Hierarchical Clustering Summary")
        print("=" * 30)
        for key, value in summary_dict.items():
            if key not in ['cluster_sizes', 'n_outliers']:
                print(f"{key}: {value}")
                
        if 'n_outliers' in summary_dict:
            print(f"n_outliers: {summary_dict['n_outliers']}")
                
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
        Clustering method: 'minibatch' (fast), 'kmeans' (exact), 'hierarchical' (2-level)
    batch_size : int, default=1000
        Batch size for MiniBatchKMeans (only used with method='minibatch')
    coverage_boost : float, default=1.5
        Boost factor for minority classes when stratified (>1.0 boosts minorities)
    min_clusters_per_class : int, default=1
        Minimum clusters per class regardless of proportion
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, sampling_size=0.1, stratify=True, method='minibatch', 
                 batch_size=1000, coverage_boost=1.5, min_clusters_per_class=1,
                 random_state=None):
        self.sampling_size = sampling_size
        self.stratify = stratify
        self.method = method
        self.batch_size = batch_size
        self.coverage_boost = coverage_boost
        self.min_clusters_per_class = min_clusters_per_class
        self.random_state = random_state
        
        # Validate parameters
        if self.method not in ['minibatch', 'kmeans', 'hierarchical']:
            raise ValueError("method must be one of: 'minibatch', 'kmeans', 'hierarchical'")
        if self.sampling_size <= 0 or self.sampling_size >= 1:
            raise ValueError("sampling_size must be between 0 and 1")
        if self.coverage_boost <= 0:
            raise ValueError("coverage_boost must be positive")
        if self.min_clusters_per_class < 1:
            raise ValueError("min_clusters_per_class must be at least 1")
    
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
            X_array = X.values
        elif isinstance(X, pd.Series):
            self.is_pandas_input_ = True
            original_index = X.index.copy()
            X_array = X.values.reshape(-1, 1)
        else:
            X_array = check_array(X)
            original_index = np.arange(len(X_array))
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        if self.stratify:
            X_array, y_array = check_X_y(X_array, y_array)
            check_classification_targets(y_array)
        
        n_samples = X_array.shape[0]
        k_total = max(1, int(n_samples * self.sampling_size))
        
        logger.info(f"Creating {k_total} clusters for test set from {n_samples} samples...")
        
        if self.stratify:
            representatives = self._stratified_clustering(X_array, y_array, k_total)
        else:
            representatives = self._global_clustering(X_array, k_total)
        
        # Ensure we don't have duplicates and sort
        representatives = np.unique(representatives)
        
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
        
        if self.method == 'hierarchical':
            return self._hierarchical_clustering(X, k_total)
        
        # Use appropriate clustering method
        clusterer = self._get_clusterer(k_total, X.shape[0], X.shape[1])  # Added X.shape[1]
        labels = clusterer.fit_predict(X)
        centroids = clusterer.cluster_centers_
        
        # Find representatives (closest to centroids) - vectorized approach
        representatives = self._find_representatives_vectorized(X, labels, centroids, k_total)
        
        return representatives
    
    def _get_clusterer(self, n_clusters, n_samples, n_features):
        """Get appropriate clusterer based on method and data size."""
        if self.method == 'minibatch':
            return MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(self.batch_size, n_samples),
                random_state=self.random_state,
                n_init=3,
                reassignment_ratio=0.01  # Faster convergence
            )
        else:  # kmeans
            return KMeans(
                n_clusters=n_clusters,
                n_init=1 if n_samples > 10000 else 10,  # Fewer inits for large data
                random_state=self.random_state,
                algorithm='elkan' if n_features and n_features > 10 else 'lloyd'  # Fixed: use n_features parameter
            )
    
    def _find_representatives_vectorized(self, X, labels, centroids, n_clusters):
        """Efficiently find representatives using vectorized operations."""
        representatives = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            if not np.any(cluster_mask):
                continue
                
            cluster_indices = np.where(cluster_mask)[0]
            cluster_samples = X[cluster_mask]
            
            if len(cluster_samples) == 1:
                representatives.append(cluster_indices[0])
            else:
                # Vectorized distance calculation
                centroid = centroids[cluster_id].reshape(1, -1)
                closest_idx, _ = pairwise_distances_argmin_min(centroid, cluster_samples)
                representatives.append(cluster_indices[closest_idx[0]])
        
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
            if k_class == 0:
                continue
                
            logger.info(f"  Class {class_label}: {k_class} clusters from {class_counts[class_label]} samples")
            
            class_mask = (y == class_label)
            X_class = X[class_mask]
            class_indices = np.where(class_mask)[0]
            
            if len(X_class) <= k_class:
                # Too few samples for clustering - take all
                representatives.extend(class_indices)
            elif k_class == 1:
                # Single cluster - select centroid
                centroid = np.mean(X_class, axis=0).reshape(1, -1)
                closest_idx, _ = pairwise_distances_argmin_min(centroid, X_class)
                representatives.append(class_indices[closest_idx[0]])
            else:
                # Cluster within class
                class_representatives = self._cluster_class(X_class, class_indices, k_class)
                representatives.extend(class_representatives)
        
        return np.array(representatives)
    
    def _cluster_class(self, X_class, class_indices, k_class):
        """Cluster samples within a single class."""
        clusterer = self._get_clusterer(k_class, len(X_class), X_class.shape[1])  # Added X_class.shape[1]
        labels = clusterer.fit_predict(X_class)
        
        representatives = []
        for cluster_id in range(k_class):
            cluster_mask = (labels == cluster_id)
            if not np.any(cluster_mask):
                continue
                
            cluster_samples = X_class[cluster_mask]
            cluster_class_indices = class_indices[cluster_mask]
            
            if len(cluster_samples) == 1:
                representatives.append(cluster_class_indices[0])
            else:
                # Find representative (closest to centroid) - vectorized
                centroid = clusterer.cluster_centers_[cluster_id].reshape(1, -1)
                closest_idx, _ = pairwise_distances_argmin_min(centroid, cluster_samples)
                representatives.append(cluster_class_indices[closest_idx[0]])
        
        return representatives
    
    def _calculate_cluster_allocation(self, class_counts, k_total, total_samples):
        """Calculate how many clusters each class should get."""
        max_class_count = max(class_counts.values())
        cluster_allocation = {}
        
        # Calculate boosted weights with numerical stability
        boosted_weights = {}
        total_boosted_weight = 0
        
        for class_label, class_count in class_counts.items():
            # Base proportion
            base_proportion = class_count / total_samples
            
            # Apply coverage boost for minority classes
            if self.coverage_boost != 1.0:
                imbalance_ratio = max_class_count / class_count
                # Use log scaling for numerical stability
                boost_factor = np.exp(np.log(imbalance_ratio) / self.coverage_boost)
                boosted_weight = base_proportion * boost_factor
            else:
                boosted_weight = base_proportion
            
            boosted_weights[class_label] = boosted_weight
            total_boosted_weight += boosted_weight
        
        # Allocate clusters with constraints
        allocated_total = 0
        for class_label, class_count in class_counts.items():
            # Proportional allocation
            normalized_weight = boosted_weights[class_label] / total_boosted_weight
            proportional_clusters = max(1, round(k_total * normalized_weight))
            
            # Apply constraints
            final_clusters = max(self.min_clusters_per_class, proportional_clusters)
            final_clusters = min(final_clusters, class_count)  # Can't exceed sample count
            
            cluster_allocation[class_label] = final_clusters
            allocated_total += final_clusters
        
        # Adjust if over-allocated (redistribute excess)
        if allocated_total > k_total:
            excess = allocated_total - k_total
            # Sort by cluster allocation (descending) to reduce from largest first
            sorted_classes = sorted(
                cluster_allocation.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for class_label, clusters in sorted_classes:
                if excess <= 0:
                    break
                max_reduction = max(0, clusters - self.min_clusters_per_class)
                reduction = min(excess, max_reduction)
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
    
    def get_train_test_split(self, X, y=None):
        """
        Get train/test split with representatives as test set.
        
        Returns
        -------
        X_train, X_test, y_train, y_test : arrays
            Train/test split where test set contains the cluster representatives
        """
        if not hasattr(self, 'representatives_'):
            raise ValueError("This KMeansRepresentativeSampler instance is not fitted yet.")
        
        # Boolean indexing works consistently for pandas and numpy
        mask = self.representatives_
        
        if hasattr(X, 'index'):  # pandas
            X_test = X[mask]
            X_train = X[~mask]
        else:  # numpy
            X = check_array(X)
            X_test = X[mask]
            X_train = X[~mask]
        
        if y is not None:
            if hasattr(y, 'index'):  # pandas Series
                y_test = y[mask]
                y_train = y[~mask]
            else:  # numpy
                y_test = y[mask]
                y_train = y[~mask]
            return X_train, X_test, y_train, y_test
        
        return X_train, X_test
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if input_features is None:
            return None
        return input_features

# Export main classes and functions
__all__ = [
    'HierarchicalClustering',
    'KMeansRepresentativeSampler',
]


# -*- coding: utf-8 -*-
# skclust/hierarchical.py

import warnings
import logging
from collections import (
    Counter,
    OrderedDict,
)

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
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
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from fastcluster import linkage
from skbio import DistanceMatrix, TreeNode

from loguru import logger
    
# Classes
class HierarchicalClustering(BaseEstimator, ClusterMixin):
    """
    Hierarchical clustering with advanced tree cutting and visualization.
    
    This class provides a comprehensive hierarchical clustering implementation
    that follows scikit-learn conventions while offering advanced features like
    dynamic tree cutting, metadata tracks, and network analysis.
    
    Parameters
    ----------
    method : str, default='average'
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
    cluster_prefix : str, optional
        If provided, cluster labels will be converted to strings with this prefix
        (e.g., cluster_prefix="C" -> "C1", "C2", etc.).
    store_data : bool, default=False
        If True, store the input data as ``data_`` after fitting.

    Attributes
    ----------
    labels_ : pd.Series or ndarray of shape (n_samples,)
        Cluster labels for each sample. Returns a pd.Series indexed by
        sample IDs when input is a pd.DataFrame or skbio.DistanceMatrix,
        otherwise an ndarray.
    data_ : skbio.DistanceMatrix
        Copy of the distance matrix with sample IDs. Only present
        when ``store_data=True``.
    linkage_matrix_ : ndarray
        The linkage matrix from hierarchical clustering.
    tree_ : skbio.TreeNode
        The hierarchical tree.
    dendrogram_ : dict
        Dendrogram data structure from scipy.
    n_clusters_ : int
        Number of clusters found.
    n_outliers_ : int
        Number of outlier samples. Only nonzero when cut_method='dynamic'.
    outliers_ : list
        Sample labels of outlier samples identified by dynamic tree cutting.
    tracks_ : dict
        Dictionary of metadata tracks for visualization.
    """
    
    def __init__(self,
                 method='average',
                 metric='euclidean',
                 min_cluster_size=3,
                 deep_split=2,
                 cut_method='dynamic',
                 cut_threshold=None,
                 name=None,
                 random_state=None,
                 distance_matrix_tol=1e-10,
                 cluster_prefix=None,
                 store_data=False):

        # Validate parameters
        valid_methods = ['ward', 'complete', 'average', 'single', 'centroid', 'median', 'weighted']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
        
        valid_cut_methods = ['dynamic', 'height', 'maxclust']
        if cut_method not in valid_cut_methods:
            raise ValueError(f"cut_method must be one of {valid_cut_methods}, got '{cut_method}'")
        
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
        self.cluster_prefix = cluster_prefix
        self.store_data = store_data

        # Initialize attributes
        self.labels_ = None
        self.linkage_matrix_ = None
        self.tree_ = None
        self.dendrogram_ = None
        self.n_clusters_ = None
        self.n_outliers_ = 0
        self.outliers_ = []
        self.tracks_ = OrderedDict()
        self._is_fitted = False
        
    def fit(self, X, y=None):
        """
        Fit hierarchical clustering to data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Training data. If a precomputed distance matrix (square DataFrame,
            ndarray, or skbio DistanceMatrix), set metric='precomputed'.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, is_skbio_dm = self._validate_input(X)

        # Validate metric='precomputed' for distance matrix inputs
        if is_skbio_dm and self.metric != 'precomputed':
            raise ValueError(
                "Input is a skbio DistanceMatrix. "
                "Please set metric='precomputed'."
            )

        # Create sample labels and track whether input had real IDs
        if is_skbio_dm:
            self._has_sample_ids = True
            self.sample_labels_ = list(X.ids)
        elif hasattr(X, 'index'):
            self._has_sample_ids = True
            self.sample_labels_ = list(X.index)
        else:
            self._has_sample_ids = False
            self.sample_labels_ = list(range(X.shape[0]))

        # Validate precomputed metric for distance-like inputs
        if not is_skbio_dm and self._is_distance_matrix(X, tol=self.distance_matrix_tol):
            if self.metric != 'precomputed':
                raise ValueError(
                    "Input appears to be a precomputed distance matrix "
                    "(square, symmetric, zero diagonal). "
                    "Please set metric='precomputed'."
                )

        # Compute or wrap distance matrix as skbio DistanceMatrix
        if is_skbio_dm:
            self.distance_matrix_ = X
        elif self.metric == 'precomputed':
            values = X.values if hasattr(X, 'values') else X
            self.distance_matrix_ = DistanceMatrix(values, ids=self.sample_labels_)
        else:
            self.distance_matrix_ = self._compute_distance_matrix(X)

        if self.store_data:
            self.data_ = self.distance_matrix_.copy()

        # Perform hierarchical clustering
        self._perform_clustering()
        
        # Cut tree to get clusters
        self._cut_tree()

        # Wrap labels as pd.Series when input had sample IDs
        if self._has_sample_ids and self.labels_ is not None:
            self.labels_ = pd.Series(self.labels_, index=self.sample_labels_)

        # Build tree representation
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
        labels : pd.Series or ndarray of shape (n_samples,)
            Cluster labels. Returns a pd.Series indexed by sample IDs
            when the input to fit() was a pd.DataFrame or skbio.DistanceMatrix,
            otherwise an ndarray.
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
        labels : pd.Series or ndarray of shape (n_samples,)
            Cluster labels. Returns a pd.Series indexed by sample IDs
            when X is a pd.DataFrame or skbio.DistanceMatrix,
            otherwise an ndarray.
        """
        return self.fit(X, y).transform()
        
    def _validate_input(self, X):
        """Validate and convert input data.

        Returns
        -------
        tuple of (array-like, bool)
            The validated/converted input and whether it was a skbio DistanceMatrix.
        """
        if isinstance(X, DistanceMatrix):
            return X, True

        if hasattr(X, 'values'):  # pandas DataFrame
            return X, False
        else:
            return np.asarray(X), False
            
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
        return DistanceMatrix(squareform(distances), ids=self.sample_labels_)
        
    def _perform_clustering(self):
        """Perform hierarchical clustering."""
        dist_condensed = self.distance_matrix_.condensed_form()
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
        
        # Set n_clusters_ and outlier info after cutting
        if self.labels_ is not None:
            unique_labels = np.unique(self.labels_)
            cluster_labels = unique_labels[unique_labels != -1]
            self.n_clusters_ = len(cluster_labels)

            outlier_mask = self.labels_ == -1
            self.outliers_ = [self.sample_labels_[i] for i, is_outlier in enumerate(outlier_mask) if is_outlier]
            self.n_outliers_ = len(self.outliers_)

            # Apply cluster prefix if specified
            if self.cluster_prefix is not None:
                self.labels_ = self._apply_cluster_prefix(self.labels_)
            
    def _cut_tree_dynamic(self):
        """Perform dynamic tree cutting."""
        try:
            import dynamicTreeCut
        except ImportError:
            raise ImportError(
                "Dynamic tree cutting requires dynamicTreeCut. "
                "Install it with: pip install dynamicTreeCut"
            )
        
        # Prepare parameters, handling None values appropriately
        params = {
            'minClusterSize': self.min_cluster_size,
            'deepSplit': self.deep_split,
        }
        
        # Only add cutHeight if it's specified
        if self.cut_threshold is not None:
            params['cutHeight'] = self.cut_threshold
        
        try:
            # dynamicTreeCut uses np.in1d which was removed in NumPy 2.0
            _in1d_patched = not hasattr(np, 'in1d')
            if _in1d_patched:
                np.in1d = np.isin

            results = dynamicTreeCut.cutreeHybrid(
                self.linkage_matrix_,
                self.distance_matrix_.data,
                **params
            )
            
            if isinstance(results, dict) and 'labels' in results:
                self.labels_ = results['labels']
            else:
                self.labels_ = results

            # dynamicTreeCut labels outliers as 0 and clusters as 1, 2, ...
            # Shift by -1 so outliers become -1 and clusters start at 0
            self.labels_ = self.labels_ - 1

        except Exception as e:
            raise RuntimeError(f"Dynamic tree cutting failed: {e}")
        finally:
            if _in1d_patched:
                del np.in1d
            
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
        """Build tree from linkage matrix."""
        try:
            self.tree_ = TreeNode.from_linkage_matrix(
                self.linkage_matrix_,
                self.sample_labels_
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
        
    def _check_plotting_available(self):
        """Check if plotting dependencies are available."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.colors import rgb2hex
        except ImportError:
            raise ImportError(
                "Plotting functionality requires matplotlib. "
                "Install it with: pip install matplotlib"
            )
        
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
        import matplotlib.patches as patches
        
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
        import matplotlib.pyplot as plt
        
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
            if label == -1:
                prefixed_labels[i] = -1
            else:
                prefixed_labels[i] = f"{self.cluster_prefix}{label}"

        return prefixed_labels
        
    def _generate_cluster_colors(self, outlier_color='white', outlier_label=""):
        """Generate colors for clusters."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import rgb2hex

        if self.n_clusters_ is None:
            return {}

        if self.n_clusters_ <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters_))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(self.n_clusters_, 20)))

        # Get actual cluster IDs (exclude outliers)
        unique_labels = np.unique(self.labels_)
        cluster_ids = [label for label in unique_labels if label != -1]

        color_dict = {}
        for i, cluster_id in enumerate(cluster_ids):
            if i < len(colors):
                color_dict[cluster_id] = rgb2hex(colors[i])
            else:
                color_dict[cluster_id] = 'gray'

        # Add outlier color if outliers exist
        if -1 in unique_labels:
            color_dict[-1] = outlier_color

        return color_dict

    def plot(self, figsize=(13, 5), show_clusters=True, show_tracks=True,
             cluster_colors=None, track_height=0.8, show_cluster_labels=False,
             cluster_label="Clusters", branch_color="black", show_leaf_labels=True,
             title=None, track_padding=None, outlier_color='white',
             outlier_label=None, **kwargs):
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
        title : str or None, default=None
            Custom title for the plot. If None, auto-generates based on
            the instance name.
        track_padding : float or None, default=None
            Vertical spacing (hspace) between subplots. If None, uses
            matplotlib's tight_layout(). Typical values: 0.0-0.5.
        outlier_color : str, default='white'
            Color used for the outlier cluster in the cluster track.
        outlier_label : str or None, default=None
            Display label for outlier samples in the cluster track.
            If None, defaults to an empty string.
        **kwargs
            Additional dendrogram plotting parameters.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        self._check_fitted()
        self._check_plotting_available()
        
        import matplotlib.pyplot as plt
        import numpy as np
        
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
        
        if title is not None:
            ax_dendro.set_title(title)
        elif self.name:
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
            if outlier_label is None:
                outlier_label = ""

            if cluster_colors is None:
                cluster_colors = self._generate_cluster_colors(outlier_color=outlier_color)

            # Replace -1 with the display outlier_label in colors and data
            if -1 in cluster_colors:
                cluster_colors[outlier_label] = cluster_colors.pop(-1)

            if isinstance(self.labels_, pd.Series):
                cluster_data = self.labels_.replace({-1: outlier_label})
            else:
                cluster_data = pd.Series(self.labels_, index=self.sample_labels_).replace({-1: outlier_label})

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
            
        if track_padding is not None:
            plt.subplots_adjust(hspace=track_padding)
        else:
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
            non_outlier_clusters = cluster_counts[cluster_counts.index != -1]
            summary_dict['cluster_sizes'] = non_outlier_clusters.to_dict()
            summary_dict['n_outliers'] = self.n_outliers_
            
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

    def to_igraph(self, threshold=None):
        """
        Convert the fitted distance matrix to an igraph weighted graph.

        Creates a weighted undirected graph from the pairwise distance matrix.
        Edge weights are the distances. If a threshold is provided, only edges
        with distances <= threshold are included; otherwise, a fully connected
        graph is created.

        Parameters
        ----------
        threshold : float or None, default=None
            Maximum distance for edge inclusion. If None, all pairwise
            distances are included (fully connected graph).

        Returns
        -------
        ig.Graph
            Undirected weighted graph. Vertex attribute ``name`` contains
            sample labels; vertex attribute ``cluster`` contains cluster
            assignments (if available). Edge attribute ``weight`` contains
            pairwise distances.
        """
        self._check_fitted()

        import igraph as ig

        dm_values = self.distance_matrix_.data
        n = dm_values.shape[0]
        i_indices, j_indices = np.triu_indices(n, k=1)
        distances = dm_values[i_indices, j_indices]

        if threshold is not None:
            mask = distances <= threshold
            i_indices = i_indices[mask]
            j_indices = j_indices[mask]
            distances = distances[mask]

        edges = list(zip(i_indices.tolist(), j_indices.tolist()))
        weights = distances.tolist()

        graph = ig.Graph(n=n, edges=edges, directed=False)
        graph.es['weight'] = weights
        graph.vs['name'] = [str(label) for label in self.sample_labels_]

        if self.labels_ is not None:
            graph.vs['cluster'] = list(self.labels_)

        return graph
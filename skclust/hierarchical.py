# -*- coding: utf-8 -*-
# skclust/hierarchical.py

import warnings
from abc import (
    ABC,
    abstractmethod,
)
from collections import (
    Counter,
    OrderedDict,
)

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
)
from scipy.spatial.distance import (
    squareform,
    pdist,
)
from sklearn.base import (
    BaseEstimator,
    ClusterMixin,
)
from skbio import (
    DistanceMatrix,
    TreeNode,
)
from fastcluster import linkage

from loguru import logger


# =============================================================================
# Base class
# =============================================================================
class BaseHierarchicalClustering(ABC, BaseEstimator, ClusterMixin):
    """
    Abstract base for hierarchical clustering with tree cutting and visualization.

    Subclasses must implement ``_prepare_input`` (validate input, set
    ``sample_labels_`` and ``_has_sample_ids``) and ``_perform_clustering``
    (produce ``linkage_matrix_``).

    Parameters
    ----------
    method : str
        Linkage method.
    metric : str
        Distance metric.
    min_cluster_size : int
        Minimum cluster size for dynamic/hybrid tree cutting.
    deep_split : int or bool
        Cluster-splitting sensitivity (interpretation depends on ``cut_method``).
    cut_method : str
        Tree cutting strategy.
    cut_threshold : float or None
        Threshold for tree cutting (interpretation depends on ``cut_method``).
    name : str or None
        Instance name (used in plot titles and tree root).
    random_state : int or None
        Random state for reproducibility.
    cluster_prefix : str or None
        If provided, cluster labels become ``"{prefix}{id}"`` strings.
    """

    # Subclasses must override with their supported cut methods
    _valid_cut_methods: list = []

    def __init__(
        self,
        method="average",
        metric="euclidean",
        min_cluster_size=20,
        deep_split=1,
        cut_method="tree",
        cut_threshold=None,
        name=None,
        random_state=None,
        cluster_prefix=None,
    ):
        # --- method validation (common superset; subclasses may restrict) ---
        valid_methods = [
            "ward", "complete", "average", "single",
            "centroid", "median", "weighted",
        ]
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{method}'"
            )

        # --- cut_method validation ---
        if cut_method not in self._valid_cut_methods:
            raise ValueError(
                f"cut_method must be one of {self._valid_cut_methods} "
                f"for {type(self).__name__}, got '{cut_method}'"
            )

        # --- deep_split validation for cut_method='tree' ---
        if cut_method == "tree":
            if isinstance(deep_split, int) and not isinstance(deep_split, bool):
                deep_split = bool(deep_split)
                logger.info(
                    f"deep_split converted from int to bool ({deep_split}) "
                    "for cut_method='tree'"
                )
            if not isinstance(deep_split, bool):
                raise ValueError(
                    f"deep_split must be a boolean for cut_method='tree', "
                    f"got {deep_split}"
                )

        if min_cluster_size < 1:
            raise ValueError(
                f"min_cluster_size must be >= 1, got {min_cluster_size}"
            )

        if cluster_prefix is not None and not isinstance(cluster_prefix, str):
            raise ValueError(
                f"cluster_prefix must be a string or None, "
                f"got {type(cluster_prefix)}"
            )

        self.method = method
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.deep_split = deep_split
        self.cut_method = cut_method
        self.cut_threshold = cut_threshold
        self.name = name
        self.random_state = random_state
        self.cluster_prefix = cluster_prefix

        # Shared fitted attributes
        self.labels_ = None
        self.linkage_matrix_ = None
        self.tree_ = None
        self.dendrogram_ = None
        self.n_clusters_ = None
        self.n_outliers_ = 0
        self.outliers_ = []
        self.tracks_ = OrderedDict()
        self._is_fitted = False

    # --------------------------------------------------------------------- #
    # Template-method fit
    # --------------------------------------------------------------------- #
    def fit(self, X, y=None):
        """
        Fit hierarchical clustering.

        Parameters
        ----------
        X : array-like
            Input data (features or distances, depending on the subclass).
        y : Ignored

        Returns
        -------
        self
        """
        # Subclass hook: validate input, set sample_labels_ / _has_sample_ids
        self._prepare_input(X)

        # Subclass hook: produce linkage_matrix_
        self._perform_clustering()

        # Dendrogram (shared — operates on linkage_matrix_)
        self.dendrogram_ = dendrogram(
            self.linkage_matrix_,
            labels=self.sample_labels_,
            no_plot=True,
        )
        self.leaves_ = self.dendrogram_["ivl"]

        # Cut tree to get cluster labels
        self._cut_tree()

        # Wrap labels as pd.Series when input had sample IDs
        if self._has_sample_ids and self.labels_ is not None:
            self.labels_ = pd.Series(self.labels_, index=self.sample_labels_)

        # Build skbio tree
        self._build_tree()

        self._is_fitted = True
        return self

    @abstractmethod
    def _prepare_input(self, X):
        """Validate *X*, set ``self.sample_labels_`` and ``self._has_sample_ids``."""

    @abstractmethod
    def _perform_clustering(self):
        """Perform linkage; must set ``self.linkage_matrix_``."""

    # --------------------------------------------------------------------- #
    # Transform
    # --------------------------------------------------------------------- #
    def transform(self, X=None):
        """Return cluster labels."""
        self._check_fitted()
        return self.labels_

    def fit_transform(self, X, y=None):
        """Fit and return cluster labels."""
        return self.fit(X, y).transform()

    # --------------------------------------------------------------------- #
    # Tree cutting
    # --------------------------------------------------------------------- #
    def _cut_tree(self):
        """Dispatch to the appropriate tree-cutting method."""
        if self.cut_method == "tree":
            self._cut_tree_dynamic_tree()
        elif self.cut_method == "height":
            self._cut_tree_height()
        elif self.cut_method == "maxclust":
            self._cut_tree_maxclust()
        else:
            raise ValueError(
                f"Unknown cut_method '{self.cut_method}'. "
                f"Must be one of {self._valid_cut_methods}."
            )

        # Post-cut bookkeeping
        if self.labels_ is not None:
            unique_labels = np.unique(self.labels_)
            cluster_labels = unique_labels[unique_labels != -1]
            self.n_clusters_ = len(cluster_labels)

            outlier_mask = self.labels_ == -1
            self.outliers_ = [
                self.sample_labels_[i]
                for i, is_outlier in enumerate(outlier_mask)
                if is_outlier
            ]
            self.n_outliers_ = len(self.outliers_)

            if self.cluster_prefix is not None:
                self.labels_ = self._apply_cluster_prefix(self.labels_)

    def _cut_tree_height(self):
        """Fixed-height cut via ``scipy.cluster.hierarchy.fcluster``."""
        cut_height = self.cut_threshold
        if cut_height is None:
            max_height = np.max(self.linkage_matrix_[:, 2])
            cut_height = 0.7 * max_height

        self.labels_ = fcluster(
            self.linkage_matrix_, cut_height, criterion="distance"
        )

    def _cut_tree_maxclust(self):
        """Fixed-k cut via ``scipy.cluster.hierarchy.fcluster``."""
        if self.cut_threshold is None:
            raise ValueError(
                "cut_threshold must be specified when using "
                "cut_method='maxclust'"
            )
        if not isinstance(self.cut_threshold, int) or self.cut_threshold < 1:
            raise ValueError(
                "cut_threshold must be a positive integer when using "
                "cut_method='maxclust'"
            )
        self.labels_ = fcluster(
            self.linkage_matrix_, self.cut_threshold, criterion="maxclust"
        )

    def _cut_tree_dynamic_tree(self):
        """Dynamic tree cut using dendrogram structure only (no distance matrix).

        A pure-dendrogram method that identifies clusters by variable-height
        branch pruning.  Faster than hybrid but may misassign outlying objects
        since it cannot consult pairwise distances.

        Based on ``cutreeDynamicTree`` from the R ``dynamicTreeCut`` package.
        """
        n = self.linkage_matrix_.shape[0] + 1
        heights = self.linkage_matrix_[:, 2]

        max_tree_height = (
            self.cut_threshold if self.cut_threshold is not None else 1.0
        )
        if max_tree_height > np.max(heights):
            max_tree_height = 0.99 * np.max(heights)

        static_labels = fcluster(
            self.linkage_matrix_, max_tree_height, criterion="distance"
        )

        # Remove clusters smaller than min_cluster_size
        counts = Counter(static_labels)
        for label, count in counts.items():
            if count < self.min_cluster_size:
                static_labels[static_labels == label] = 0
        static_labels[static_labels == 0] = -1

        if not self.deep_split:
            self.labels_ = self._renumber_labels(static_labels)
            return

        # Deep split: iteratively check branches for substructure
        dendro_order = self.dendrogram_["leaves"]

        merge = np.zeros((n - 1, 2), dtype=int)
        for i in range(n - 1):
            merge[i, 0] = int(self.linkage_matrix_[i, 0])
            merge[i, 1] = int(self.linkage_matrix_[i, 1])

        ordered_heights = np.zeros(n)
        for i in range(n):
            idx = i
            for j in range(n - 1):
                if merge[j, 0] == idx or merge[j, 1] == idx:
                    ordered_heights[i] = heights[j]
                    break

        ordered_labels = static_labels[dendro_order]
        ordered_h = ordered_heights[dendro_order]

        unique_clusters = [c for c in np.unique(ordered_labels) if c != -1]
        new_labels = ordered_labels.copy()
        next_label = max(unique_clusters) + 1 if unique_clusters else 1

        changed = True
        while changed:
            changed = False
            current_clusters = [c for c in np.unique(new_labels) if c != -1]
            for cl in current_clusters:
                mask = new_labels == cl
                indices = np.where(mask)[0]
                if len(indices) < 2 * self.min_cluster_size:
                    continue

                cl_heights = ordered_h[indices]
                mean_h = np.mean(cl_heights)

                above = cl_heights >= mean_h
                below = ~above

                runs = []
                current_run_start = 0
                current_is_above = above[0]
                for k in range(1, len(above)):
                    if above[k] != current_is_above:
                        runs.append(
                            (current_run_start, k - 1, current_is_above)
                        )
                        current_run_start = k
                        current_is_above = above[k]
                runs.append(
                    (current_run_start, len(above) - 1, current_is_above)
                )

                split_points = []
                for k in range(1, len(runs)):
                    if not runs[k - 1][2] and runs[k][2]:
                        if (
                            runs[k - 1][1] - runs[k - 1][0] + 1
                            >= self.min_cluster_size // 3
                        ):
                            split_points.append(runs[k][0])

                if len(split_points) == 0:
                    continue

                segments = []
                prev = 0
                for sp in split_points:
                    segments.append(indices[prev:sp])
                    prev = sp
                segments.append(indices[prev:])

                valid_segments = [
                    s for s in segments if len(s) >= self.min_cluster_size
                ]
                if len(valid_segments) <= 1:
                    continue

                changed = True
                for seg in valid_segments:
                    new_labels[seg] = next_label
                    next_label += 1
                leftover = np.setdiff1d(
                    indices, np.concatenate(valid_segments)
                )
                if len(leftover) > 0:
                    new_labels[leftover] = -1

        # Map back from dendrogram order to original order
        result = np.full(n, -1, dtype=int)
        for i, orig_idx in enumerate(dendro_order):
            result[orig_idx] = new_labels[i]

        self.labels_ = self._renumber_labels(result)

    # --------------------------------------------------------------------- #
    # Label helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _renumber_labels(labels):
        """Renumber cluster labels by descending size, keeping -1 for unassigned."""
        counts = Counter(l for l in labels if l != -1)
        if not counts:
            return labels
        sorted_clusters = [c for c, _ in counts.most_common()]
        mapping = {old: new for new, old in enumerate(sorted_clusters)}
        return np.array([mapping[l] if l != -1 else -1 for l in labels])

    def _apply_cluster_prefix(self, labels):
        """Apply cluster prefix to labels, converting to strings."""
        prefixed = np.empty(len(labels), dtype=object)
        for i, label in enumerate(labels):
            prefixed[i] = -1 if label == -1 else f"{self.cluster_prefix}{label}"
        return prefixed

    # --------------------------------------------------------------------- #
    # Tree
    # --------------------------------------------------------------------- #
    def _build_tree(self):
        """Build ``skbio.TreeNode`` from the linkage matrix."""
        try:
            self.tree_ = TreeNode.from_linkage_matrix(
                self.linkage_matrix_, self.sample_labels_
            )
            if self.name:
                self.tree_.name = self.name
        except Exception as e:
            warnings.warn(f"Tree building failed: {e}")
            self.tree_ = None

    # --------------------------------------------------------------------- #
    # Checks
    # --------------------------------------------------------------------- #
    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet."
            )

    @staticmethod
    def _check_plotting_available():
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            import matplotlib.patches as patches  # noqa: F401
            from matplotlib.colors import rgb2hex  # noqa: F401
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib. "
                "Install it with: pip install matplotlib"
            )

    # --------------------------------------------------------------------- #
    # Tracks
    # --------------------------------------------------------------------- #
    def add_track(self, name, data, track_type="continuous", color=None, **kwargs):
        """
        Add a metadata track for visualization.

        Parameters
        ----------
        name : str
            Track name.
        data : Mapping or pd.Series
            Values keyed / indexed by sample label.
        track_type : str, default='continuous'
            ``'continuous'`` or ``'categorical'``.
        color : str, dict, pd.Series, or None
            Color specification.
        **kwargs
            Extra plotting parameters.
        """
        self._check_fitted()

        if track_type not in ("continuous", "categorical"):
            raise ValueError(
                f"track_type must be 'continuous' or 'categorical', "
                f"got '{track_type}'"
            )

        from collections.abc import Mapping

        if not isinstance(data, (Mapping, pd.Series)):
            raise ValueError(
                "Track data must be a mapping type (dict, OrderedDict, etc.) "
                "with sample names as keys or a pandas Series with sample "
                f"names as index. Got {type(data)} instead."
            )

        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        data = data.reindex(self.sample_labels_)

        missing = set(self.sample_labels_) - set(data.index)
        if missing:
            warnings.warn(f"Track '{name}' missing data for samples: {missing}")

        self.tracks_[name] = {
            "data": data,
            "type": track_type,
            "color": color,
            "kwargs": kwargs,
        }

    # --------------------------------------------------------------------- #
    # Plotting
    # --------------------------------------------------------------------- #
    def _generate_cluster_colors(self, outlier_color="white", outlier_label=""):
        """Generate a color dict for cluster ids."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import rgb2hex

        if self.n_clusters_ is None:
            return {}

        if self.n_clusters_ <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters_))
        else:
            colors = plt.cm.tab20(
                np.linspace(0, 1, min(self.n_clusters_, 20))
            )

        unique_labels = np.unique(self.labels_)
        cluster_ids = [l for l in unique_labels if l != -1]

        color_dict = {}
        for i, cid in enumerate(cluster_ids):
            color_dict[cid] = rgb2hex(colors[i]) if i < len(colors) else "gray"

        if -1 in unique_labels:
            color_dict[-1] = outlier_color

        return color_dict

    def _plot_categorical_track(
        self, ax, data, colors, show_labels=False, label_text=None
    ):
        """Plot categorical data as colored rectangles."""
        import matplotlib.patches as patches

        ordered_leaves = self.leaves_

        for i, sample in enumerate(ordered_leaves):
            if sample in data.index and pd.notna(data[sample]):
                category = data[sample]
                color = colors.get(category, "gray")
                rect = patches.Rectangle(
                    (i * 10 + 5 - 5, 0), 10, 1,
                    facecolor=color, edgecolor="none", alpha=0.8,
                )
                ax.add_patch(rect)

        if show_labels and label_text is not None:
            category_positions = {}
            for i, sample in enumerate(ordered_leaves):
                if sample in data.index and pd.notna(data[sample]):
                    category = data[sample]
                    if category not in category_positions:
                        category_positions[category] = []
                    category_positions[category].append(i * 10 + 5)

            for category, positions_list in category_positions.items():
                if positions_list:
                    center = np.mean(positions_list)
                    ax.text(
                        center, 0.5, str(category),
                        ha="center", va="center", fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white", alpha=0.8,
                        ),
                    )

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
            data = track_info["data"]
            track_type = track_info["type"]
            color = track_info["color"]

            if track_type == "continuous":
                positions = []
                values = []
                colors_list = []

                for j, sample in enumerate(ordered_leaves):
                    if sample in data.index and pd.notna(data[sample]):
                        positions.append(j * 10 + 5)
                        values.append(data[sample])
                        if isinstance(color, dict):
                            colors_list.append(color.get(sample, "steelblue"))
                        elif isinstance(color, pd.Series) and sample in color.index:
                            colors_list.append(color[sample])
                        else:
                            colors_list.append(
                                color if color is not None else "steelblue"
                            )

                bar_color = colors_list if colors_list else "steelblue"
                ax.bar(positions, values, color=bar_color, width=8)
                ax.set_ylabel(track_name)

            elif track_type == "categorical":
                if color is None or isinstance(color, str):
                    unique_vals = data.dropna().unique()
                    if isinstance(color, str):
                        color_map = {val: color for val in unique_vals}
                    else:
                        color_map = dict(
                            zip(
                                unique_vals,
                                plt.cm.Set1(
                                    np.linspace(0, 1, len(unique_vals))
                                ),
                            )
                        )
                else:
                    color_map = color

                self._plot_categorical_track(
                    ax, data, color_map, label_text=track_name
                )

            tree_width = len(ordered_leaves) * 10
            ax.set_xlim(0, tree_width)

    def plot(
        self,
        figsize=(13, 5),
        show_clusters=True,
        show_tracks=True,
        cluster_colors=None,
        track_height=0.8,
        show_cluster_labels=False,
        cluster_label="Clusters",
        branch_color="black",
        show_leaf_labels=True,
        title=None,
        track_padding=None,
        outlier_color="white",
        outlier_label=None,
        **kwargs,
    ):
        """
        Plot dendrogram with optional cluster colouring and tracks.

        Parameters
        ----------
        figsize : tuple, default=(13, 5)
            Figure size.
        show_clusters : bool, default=True
            Show cluster assignments as coloured rectangles.
        show_tracks : bool, default=True
            Show metadata tracks.
        cluster_colors : dict or None
            Custom cluster colours.
        track_height : float, default=0.8
            Height ratio for tracks.
        show_cluster_labels : bool, default=False
            Show cluster numbers on the cluster track.
        cluster_label : str, default='Clusters'
            Label for the cluster track.
        branch_color : str, default='black'
            Dendrogram branch colour.
        show_leaf_labels : bool, default=True
            Show sample labels on the x-axis.
        title : str or None
            Plot title (auto-generated if *None*).
        track_padding : float or None
            ``hspace`` between subplots (tight_layout if *None*).
        outlier_color : str, default='white'
            Colour for outlier entries in the cluster track.
        outlier_label : str or None
            Display label for outliers (empty string if *None*).
        **kwargs
            Extra dendrogram plotting parameters.
        """
        self._check_fitted()
        self._check_plotting_available()

        import matplotlib.pyplot as plt

        n_tracks = len(self.tracks_) if show_tracks else 0
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
                sharex=True,
            )
            if n_subplots == 2:
                axes = [axes[0], axes[1]]
            ax_dendro = axes[0]
        else:
            fig, ax_dendro = plt.subplots(figsize=figsize)
            axes = [ax_dendro]

        # Draw dendrogram branches
        for xs, ys in zip(
            self.dendrogram_["icoord"], self.dendrogram_["dcoord"]
        ):
            ax_dendro.plot(xs, ys, color=branch_color, linewidth=1)

        tree_width = len(self.leaves_) * 10
        max_height = np.max(self.dendrogram_["dcoord"])
        tree_height = max_height + max_height * 0.05

        ax_dendro.set_xlim(0, tree_width)
        ax_dendro.set_ylim(0, tree_height)

        if title is not None:
            ax_dendro.set_title(title)
        elif self.name:
            ax_dendro.set_title(f"Hierarchical Clustering: {self.name}")
        else:
            ax_dendro.set_title("Hierarchical Clustering")

        bottom_axis = None
        if show_leaf_labels:
            bottom_axis = axes[-1] if n_subplots > 1 else ax_dendro

        if n_subplots > 1:
            ax_dendro.set_xticklabels([])
        elif show_leaf_labels:
            leaf_positions = [i * 10 + 5 for i in range(len(self.leaves_))]
            ax_dendro.set_xticks(leaf_positions)
            ax_dendro.set_xticklabels(self.leaves_, rotation=90)

        current_axis_idx = 1

        # Cluster track
        if show_clusters and self.labels_ is not None and n_subplots > 1:
            if outlier_label is None:
                outlier_label = ""

            if cluster_colors is None:
                cluster_colors = self._generate_cluster_colors(
                    outlier_color=outlier_color
                )

            if -1 in cluster_colors:
                cluster_colors[outlier_label] = cluster_colors.pop(-1)

            if isinstance(self.labels_, pd.Series):
                cluster_data = self.labels_.replace({-1: outlier_label})
            else:
                cluster_data = pd.Series(
                    self.labels_, index=self.sample_labels_
                ).replace({-1: outlier_label})

            ax_clusters = axes[current_axis_idx]
            self._plot_categorical_track(
                ax_clusters, cluster_data, cluster_colors,
                show_labels=show_cluster_labels, label_text=cluster_label,
            )
            current_axis_idx += 1

        # Metadata tracks
        if show_tracks and n_tracks > 0 and n_subplots > 1:
            track_axes = axes[current_axis_idx: current_axis_idx + n_tracks]
            self._plot_tracks(track_axes, track_height)

        # Leaf labels on the bottom-most axis
        if show_leaf_labels and bottom_axis is not None and n_subplots > 1:
            leaf_positions = [i * 10 + 5 for i in range(len(self.leaves_))]
            bottom_axis.set_xticks(leaf_positions)
            bottom_axis.set_xticklabels(self.leaves_, rotation=90)

        if track_padding is not None:
            plt.subplots_adjust(hspace=track_padding)
        else:
            plt.tight_layout()
        return fig, axes

    # --------------------------------------------------------------------- #
    # Summary
    # --------------------------------------------------------------------- #
    def summary(self):
        """
        Print and return a summary of the clustering results.

        Returns
        -------
        dict
        """
        self._check_fitted()

        summary_dict = {
            "n_samples": len(self.sample_labels_),
            "n_clusters": self.n_clusters_,
            "method": self.method,
            "metric": self.metric,
            "cut_method": self.cut_method,
        }

        if self.labels_ is not None:
            cluster_counts = pd.Series(self.labels_).value_counts().sort_index()
            non_outlier = cluster_counts[cluster_counts.index != -1]
            summary_dict["cluster_sizes"] = non_outlier.to_dict()
            summary_dict["n_outliers"] = self.n_outliers_

        print(f"{type(self).__name__} Summary")
        print("=" * 30)
        for key, value in summary_dict.items():
            if key not in ("cluster_sizes", "n_outliers"):
                print(f"{key}: {value}")

        if "n_outliers" in summary_dict:
            print(f"n_outliers: {summary_dict['n_outliers']}")

        if "cluster_sizes" in summary_dict:
            print("\nCluster sizes:")
            for cluster, size in summary_dict["cluster_sizes"].items():
                print(f"  Cluster {cluster}: {size} samples")

        return summary_dict


# =============================================================================
# Full-distance-matrix path (fastcluster + dynamicTreeCut)
# =============================================================================
class HierarchicalClustering(BaseHierarchicalClustering):
    """
    Hierarchical clustering using full pairwise distances (fastcluster).

    Supports all seven linkage methods and all four tree-cutting strategies
    including ``'hybrid'`` (adaptive branch pruning backed by the distance
    matrix via ``dynamicTreeCut``).

    Parameters
    ----------
    method : str, default='average'
        Linkage method: ``'ward'``, ``'complete'``, ``'average'``,
        ``'single'``, ``'centroid'``, ``'median'``, ``'weighted'``.
    metric : str, default='euclidean'
        Distance metric (or ``'precomputed'`` for pre-computed distances).
    min_cluster_size : int, default=20
        Minimum cluster size for dynamic / hybrid tree cutting.
    deep_split : int or bool, default=1
        Cluster-splitting sensitivity.

        * ``cut_method='hybrid'``: integer 0–4 (higher → more clusters).
        * ``cut_method='tree'``: boolean (``True`` enables iterative
          sub-cluster detection).
    cut_method : str, default='hybrid'
        One of ``'hybrid'``, ``'tree'``, ``'height'``, ``'maxclust'``.
    cut_threshold : float or None
        Interpretation depends on *cut_method* (see class docs of
        ``BaseHierarchicalClustering``).
    pam_stage : bool, default=True
        ``'hybrid'`` only — perform PAM-like refinement stage.
    pam_respects_dendro : bool, default=True
        ``'hybrid'`` only — PAM respects dendrogram branch structure.
    use_medoids : bool, default=False
        ``'hybrid'`` only — use medoid distances in PAM stage.
    max_pam_dist : float or None
        ``'hybrid'`` only — maximum distance for PAM assignment.
    respect_small_clusters : bool, default=True
        ``'hybrid'`` only — keep small branches together in PAM stage.
    distance_matrix_tol : float, default=1e-10
        Tolerance for validating distance matrix properties.
    store_data : bool, default=False
        If ``True``, store the distance matrix as ``data_`` after fitting.
    name : str or None
        Instance name.
    random_state : int or None
        Random state.
    cluster_prefix : str or None
        Prefix for string cluster labels.
    """

    _valid_cut_methods = ["hybrid", "tree", "height", "maxclust"]

    def __init__(
        self,
        method="average",
        metric="euclidean",
        min_cluster_size=20,
        deep_split=1,
        cut_method="hybrid",
        cut_threshold=None,
        pam_stage=True,
        pam_respects_dendro=True,
        use_medoids=False,
        max_pam_dist=None,
        respect_small_clusters=True,
        distance_matrix_tol=1e-10,
        store_data=False,
        name=None,
        random_state=None,
        cluster_prefix=None,
    ):
        super().__init__(
            method=method,
            metric=metric,
            min_cluster_size=min_cluster_size,
            deep_split=deep_split,
            cut_method=cut_method,
            cut_threshold=cut_threshold,
            name=name,
            random_state=random_state,
            cluster_prefix=cluster_prefix,
        )

        # --- hybrid-specific deep_split validation ---
        if self.cut_method == "hybrid":
            if isinstance(self.deep_split, bool):
                self.deep_split = int(self.deep_split)
                logger.info(
                    f"deep_split converted from bool to int "
                    f"({self.deep_split}) for cut_method='hybrid'"
                )
            if (
                not isinstance(self.deep_split, int)
                or self.deep_split not in range(5)
            ):
                raise ValueError(
                    f"deep_split must be an integer between 0 and 4 for "
                    f"cut_method='hybrid', got {self.deep_split}"
                )

        if distance_matrix_tol <= 0:
            raise ValueError(
                f"distance_matrix_tol must be positive, "
                f"got {distance_matrix_tol}"
            )

        self.pam_stage = pam_stage
        self.pam_respects_dendro = pam_respects_dendro
        self.use_medoids = use_medoids
        self.max_pam_dist = max_pam_dist
        self.respect_small_clusters = respect_small_clusters
        self.distance_matrix_tol = distance_matrix_tol
        self.store_data = store_data

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    def _prepare_input(self, X):
        X, is_skbio_dm = self._validate_input(X)

        if is_skbio_dm and self.metric != "precomputed":
            raise ValueError(
                "Input is a skbio DistanceMatrix. "
                "Please set metric='precomputed'."
            )

        # Sample labels
        if is_skbio_dm:
            self._has_sample_ids = True
            self.sample_labels_ = list(X.ids)
        elif hasattr(X, "index"):
            self._has_sample_ids = True
            self.sample_labels_ = list(X.index)
        else:
            self._has_sample_ids = False
            self.sample_labels_ = list(range(X.shape[0]))

        # Auto-detect distance matrix for non-skbio inputs
        if not is_skbio_dm and self._is_distance_matrix(
            X, tol=self.distance_matrix_tol
        ):
            if self.metric != "precomputed":
                raise ValueError(
                    "Input appears to be a precomputed distance matrix "
                    "(square, symmetric, zero diagonal). "
                    "Please set metric='precomputed'."
                )

        # Build / wrap skbio DistanceMatrix
        if is_skbio_dm:
            self.distance_matrix_ = X
        elif self.metric == "precomputed":
            values = X.values if hasattr(X, "values") else X
            self.distance_matrix_ = DistanceMatrix(
                values, ids=self.sample_labels_
            )
        else:
            self.distance_matrix_ = self._compute_distance_matrix(X)

        if self.store_data:
            self.data_ = self.distance_matrix_.copy()

    def _perform_clustering(self):
        dist_condensed = self.distance_matrix_.condensed_form()
        self.linkage_matrix_ = linkage(dist_condensed, method=self.method)

    # ------------------------------------------------------------------ #
    # Tree cutting override (adds hybrid)
    # ------------------------------------------------------------------ #
    def _cut_tree(self):
        if self.cut_method == "hybrid":
            self._cut_tree_hybrid()
            # Post-cut bookkeeping (same logic as base)
            if self.labels_ is not None:
                unique_labels = np.unique(self.labels_)
                cluster_labels = unique_labels[unique_labels != -1]
                self.n_clusters_ = len(cluster_labels)

                outlier_mask = self.labels_ == -1
                self.outliers_ = [
                    self.sample_labels_[i]
                    for i, is_outlier in enumerate(outlier_mask)
                    if is_outlier
                ]
                self.n_outliers_ = len(self.outliers_)

                if self.cluster_prefix is not None:
                    self.labels_ = self._apply_cluster_prefix(self.labels_)
        else:
            super()._cut_tree()

    def _cut_tree_hybrid(self):
        """Hybrid adaptive tree cut using dendrogram + distance matrix.

        Calls ``dynamicTreeCut.cutreeHybrid``.
        """
        try:
            import dynamicTreeCut
        except ImportError:
            raise ImportError(
                "Hybrid tree cutting requires the dynamicTreeCut package. "
                "Install it with: pip install dynamicTreeCut"
            )

        params = {
            "minClusterSize": self.min_cluster_size,
            "deepSplit": self.deep_split,
            "pamStage": self.pam_stage,
            "pamRespectsDendro": self.pam_respects_dendro,
            "useMedoids": self.use_medoids,
            "respectSmallClusters": self.respect_small_clusters,
            "verbose": 0,
        }

        if self.cut_threshold is not None:
            params["cutHeight"] = self.cut_threshold

        if self.max_pam_dist is not None:
            params["maxPamDist"] = self.max_pam_dist
        elif self.cut_threshold is not None:
            params["maxPamDist"] = self.cut_threshold

        try:
            # dynamicTreeCut uses np.in1d which was removed in NumPy 2.0
            _in1d_patched = not hasattr(np, "in1d")
            if _in1d_patched:
                np.in1d = np.isin

            results = dynamicTreeCut.cutreeHybrid(
                self.linkage_matrix_,
                self.distance_matrix_.data,
                **params,
            )

            if isinstance(results, dict) and "labels" in results:
                self.labels_ = results["labels"]
            else:
                self.labels_ = results

            # dynamicTreeCut: 0 = outlier, 1+ = cluster → shift to -1, 0+
            self.labels_ = self.labels_ - 1

        except Exception as e:
            raise RuntimeError(f"Hybrid tree cutting failed: {e}")
        finally:
            if _in1d_patched:
                del np.in1d

    # ------------------------------------------------------------------ #
    # Private helpers (distance-matrix path only)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_input(X):
        """Validate and convert input; return (data, is_skbio_dm)."""
        if isinstance(X, DistanceMatrix):
            return X, True
        if hasattr(X, "values"):
            return X, False
        return np.asarray(X), False

    @staticmethod
    def _is_distance_matrix(X, tol=1e-10):
        """Heuristic check for a valid distance matrix."""
        if not hasattr(X, "shape") or X.shape[0] != X.shape[1]:
            return False
        values = X.values if hasattr(X, "values") else X
        if not np.allclose(values, values.T, rtol=tol, atol=tol):
            return False
        if not np.allclose(np.diag(values), 0, atol=tol):
            return False
        if np.any(values < -tol):
            return False
        return True

    def _compute_distance_matrix(self, X):
        """Compute pairwise distances and wrap as ``skbio.DistanceMatrix``."""
        X_values = X.values if hasattr(X, "values") else X
        distances = pdist(X_values, metric=self.metric)
        return DistanceMatrix(squareform(distances), ids=self.sample_labels_)

    # ------------------------------------------------------------------ #
    # Extras
    # ------------------------------------------------------------------ #
    def to_igraph(self, threshold=None):
        """
        Convert the distance matrix to an igraph weighted graph.

        Parameters
        ----------
        threshold : float or None
            Maximum distance for edge inclusion. If *None*, fully connected.

        Returns
        -------
        ig.Graph
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
        graph.es["weight"] = weights
        graph.vs["name"] = [str(l) for l in self.sample_labels_]

        if self.labels_ is not None:
            graph.vs["cluster"] = list(self.labels_)

        return graph


# =============================================================================
# Connectivity-constrained path (sklearn)
# =============================================================================
class ConnectivityHierarchicalClustering(BaseHierarchicalClustering):
    """
    Hierarchical clustering with a connectivity constraint (sklearn backend).

    Uses ``sklearn.cluster.AgglomerativeClustering`` under the hood, then
    reconstructs a scipy-format linkage matrix so that all shared tree
    cutting, plotting, and track functionality works identically to
    :class:`HierarchicalClustering`.

    Only linkage methods supported by sklearn are allowed: ``'ward'``,
    ``'complete'``, ``'average'``, ``'single'``.

    Parameters
    ----------
    connectivity : sparse matrix, dense array, or callable
        Connectivity constraint passed to sklearn's
        ``AgglomerativeClustering``. A callable receives *X* and should
        return a connectivity matrix.
    method : str, default='ward'
        Linkage method (sklearn name: ``linkage``).
        One of ``'ward'``, ``'complete'``, ``'average'``, ``'single'``.
    metric : str, default='euclidean'
        Distance metric. ``'precomputed'`` is **not** supported (the
        connectivity path expects feature matrices).
    min_cluster_size : int, default=20
        Minimum cluster size for dynamic tree cutting.
    deep_split : bool, default=True
        Whether to enable iterative sub-cluster detection when
        ``cut_method='tree'``.
    cut_method : str, default='tree'
        One of ``'tree'``, ``'height'``, ``'maxclust'``.
        ``'hybrid'`` is not supported because no full distance matrix
        is available.
    cut_threshold : float or None
        Threshold for tree cutting.
    name : str or None
        Instance name.
    random_state : int or None
        Random state.
    cluster_prefix : str or None
        Prefix for string cluster labels.
    """

    _valid_cut_methods = ["tree", "height", "maxclust"]

    # sklearn supports only these linkage methods
    _valid_sklearn_methods = ["ward", "complete", "average", "single"]

    def __init__(
        self,
        connectivity,
        method="ward",
        metric="euclidean",
        min_cluster_size=20,
        deep_split=True,
        cut_method="tree",
        cut_threshold=None,
        name=None,
        random_state=None,
        cluster_prefix=None,
    ):
        if metric == "precomputed":
            raise ValueError(
                "ConnectivityHierarchicalClustering operates on feature "
                "matrices, not precomputed distances. Use "
                "HierarchicalClustering for distance-matrix inputs."
            )

        if method not in self._valid_sklearn_methods:
            raise ValueError(
                f"method must be one of {self._valid_sklearn_methods} for "
                f"ConnectivityHierarchicalClustering, got '{method}'"
            )

        super().__init__(
            method=method,
            metric=metric,
            min_cluster_size=min_cluster_size,
            deep_split=deep_split,
            cut_method=cut_method,
            cut_threshold=cut_threshold,
            name=name,
            random_state=random_state,
            cluster_prefix=cluster_prefix,
        )

        self.connectivity = connectivity

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    def _prepare_input(self, X):
        # Reject distance matrices
        if isinstance(X, DistanceMatrix):
            raise ValueError(
                "ConnectivityHierarchicalClustering expects a feature "
                "matrix, not a skbio DistanceMatrix."
            )

        if hasattr(X, "values"):
            self._has_sample_ids = True
            self.sample_labels_ = list(X.index)
            self._X_values = X.values
        else:
            self._has_sample_ids = False
            self.sample_labels_ = list(range(X.shape[0]))
            self._X_values = np.asarray(X)

        # Resolve callable connectivity
        if callable(self.connectivity):
            self.connectivity_ = self.connectivity(self._X_values)
        else:
            self.connectivity_ = self.connectivity

        # Check shapes
        if self.connectivity_.shape[0] != self._X_values.shape[0]:
            raise ValueError(
                f"connectivity has {self.connectivity_.shape[0]} samples, "
                f"X has {self._X_values.shape[0]}"
            )

    def _perform_clustering(self):
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            compute_full_tree=True,
            connectivity=self.connectivity_,
            metric=self.metric,
            linkage=self.method,
        )
        model.fit(self._X_values)

        n_samples = self._X_values.shape[0]
        self.linkage_matrix_ = self._linkage_from_sklearn(model, n_samples)

        # Free the feature cache — no longer needed
        del self._X_values

    # ------------------------------------------------------------------ #
    # Linkage reconstruction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _linkage_from_sklearn(model, n_samples):
        """Reconstruct a scipy-format linkage matrix from sklearn's model.

        Parameters
        ----------
        model : AgglomerativeClustering
            Fitted sklearn model (must have ``children_`` and ``distances_``).
        n_samples : int
            Number of leaf samples.

        Returns
        -------
        np.ndarray of shape (n_samples - 1, 4)
            Columns: ``[idx1, idx2, distance, count]``.
        """
        children = model.children_
        distances = model.distances_

        counts = np.zeros(len(children))
        for i, (left, right) in enumerate(children):
            left_count = (
                1 if left < n_samples else counts[left - n_samples]
            )
            right_count = (
                1 if right < n_samples else counts[right - n_samples]
            )
            counts[i] = left_count + right_count

        return np.column_stack(
            [children, distances, counts]
        ).astype(float)
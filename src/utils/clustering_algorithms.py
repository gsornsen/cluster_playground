import time
import logging
import numpy as np
from cuml.cluster import AgglomerativeClustering as CumlAgglomerativeClustering
from cuml.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import cudf
from utils.clustering_base import ClusteringBase
from utils.hdbscan_cpu import HDBSCANCPUClusterer
from utils.hdbscan_gpu import HDBSCANGPUClusterer

logger = logging.getLogger(__name__)


class AgglomerativeClusteringAlgorithm(ClusteringBase):
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

    def perform_clustering(self, embeddings_cudf: cudf.DataFrame, n_clusters: int = 5):
        """
        Performs Agglomerative Clustering on the embeddings using cuML (GPU) or scikit-learn (CPU).
        """
        if self.use_gpu:
            clustering_model = CumlAgglomerativeClustering(n_clusters=n_clusters)
            clustering_model.fit(embeddings_cudf)
            return clustering_model.labels_
        else:
            embeddings_pd = embeddings_cudf.to_pandas()
            clustering_model = SklearnAgglomerativeClustering(n_clusters=n_clusters)
            clustering_model.fit(embeddings_pd)
            return cudf.Series(clustering_model.labels_)

    def find_optimal_clusters(
        self, embeddings_cudf: cudf.DataFrame, min_clusters: int, max_clusters: int
    ):
        start_time = time.time()
        best_score = float("-inf")
        best_n_clusters = min_clusters

        for n_clusters in range(min_clusters, max_clusters + 1):
            labels = self.perform_clustering(embeddings_cudf, n_clusters)

            # Convert to numpy for sklearn metrics
            embeddings_np = embeddings_cudf.to_numpy()
            labels_np = labels.to_numpy()

            score = silhouette_score(embeddings_np, labels_np)

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters

        end_time = time.time()
        optimization_time = end_time - start_time
        return best_n_clusters, optimization_time


class KMeansClusteringAlgorithm(ClusteringBase):
    def perform_clustering(self, embeddings_cudf: cudf.DataFrame, n_clusters: int = 5):
        """
        Performs K-Means Clustering on the embeddings using cuML.
        """
        clustering_model = KMeans(n_clusters=n_clusters)
        clustering_model.fit(embeddings_cudf)
        return clustering_model.labels_

    def find_optimal_clusters(
        self, embeddings_cudf: cudf.DataFrame, min_clusters: int, max_clusters: int
    ):
        start_time = time.time()
        best_score = float("-inf")
        best_n_clusters = min_clusters

        for n_clusters in range(min_clusters, max_clusters + 1):
            labels = self.perform_clustering(embeddings_cudf, n_clusters)

            # Convert to numpy for sklearn metrics
            embeddings_np = embeddings_cudf.to_numpy()
            labels_np = labels.to_numpy()

            score = silhouette_score(embeddings_np, labels_np)

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters

        end_time = time.time()
        optimization_time = end_time - start_time
        return best_n_clusters, optimization_time


class HDBSCANClusteringAlgorithm(ClusteringBase):
    """
    HDBSCAN clustering algorithm that can use either CPU or GPU implementation.

    This class follows the same interface as other clustering algorithms in the framework,
    but adapts HDBSCAN's density-based approach which doesn't require pre-specifying
    the number of clusters.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        min_cluster_size: int = 5,
        min_samples: int = None,
        merge_clusters: bool = False,
        standardize_features: bool = True,
        skip_grid_search: bool = False,
    ):
        """
        Initialize HDBSCAN algorithm.

        Args:
            use_gpu: Whether to use GPU implementation
            min_cluster_size: Minimum size for clusters
            min_samples: Minimum samples for core points
            merge_clusters: Whether to merge similar clusters
            standardize_features: Whether to standardize input features
            skip_grid_search: Skip grid search optimization
        """
        self.use_gpu = use_gpu
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.merge_clusters = merge_clusters
        self.standardize_features = standardize_features
        self.skip_grid_search = skip_grid_search

        # Initialize the appropriate clusterer
        if self.use_gpu:
            self.clusterer = HDBSCANGPUClusterer(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                merge_clusters=merge_clusters,
                standardize_features=standardize_features,
            )
        else:
            self.clusterer = HDBSCANCPUClusterer(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                merge_clusters=merge_clusters,
                standardize_features=standardize_features,
                skip_grid_search=skip_grid_search,
            )

        logger.info(
            f"Initialized HDBSCAN with {'GPU' if use_gpu else 'CPU'} implementation"
        )

    def perform_clustering(
        self, embeddings_cudf: cudf.DataFrame, n_clusters: int = None
    ):
        """
        Perform HDBSCAN clustering.

        Note: n_clusters parameter is ignored for HDBSCAN as it's a density-based
        algorithm that automatically determines the number of clusters.

        Args:
            embeddings_cudf: Input embeddings as cuDF DataFrame
            n_clusters: Ignored for HDBSCAN (kept for interface consistency)

        Returns:
            Cluster labels as cuDF Series
        """
        if n_clusters is not None:
            logger.warning(
                f"HDBSCAN ignores n_clusters parameter ({n_clusters}). "
                "Number of clusters determined automatically."
            )

        start_time = time.time()

        if self.use_gpu:
            # GPU implementation expects cuDF DataFrame
            self.clusterer.fit(embeddings_cudf)
            labels = self.clusterer.labels_
        else:
            # CPU implementation expects numpy array
            embeddings_np = embeddings_cudf.to_numpy()
            self.clusterer.fit(embeddings_np)
            # Convert numpy labels back to cuDF Series
            labels = cudf.Series(self.clusterer.labels_)

        clustering_time = time.time() - start_time
        n_clusters_found = len(labels.unique())

        logger.info(
            f"HDBSCAN clustering completed in {clustering_time:.2f}s, "
            f"found {n_clusters_found} clusters"
        )

        return labels

    def find_optimal_clusters(
        self, embeddings_cudf: cudf.DataFrame, min_clusters: int, max_clusters: int
    ):
        """
        Find optimal clustering for HDBSCAN.

        Since HDBSCAN automatically determines the number of clusters, this method
        performs clustering once and evaluates the quality. The min_clusters and
        max_clusters parameters are used only for validation.

        Args:
            embeddings_cudf: Input embeddings
            min_clusters: Minimum expected clusters (for validation)
            max_clusters: Maximum expected clusters (for validation)

        Returns:
            Tuple of (n_clusters_found, optimization_time)
        """
        logger.info("Running HDBSCAN optimization (automatic cluster detection)")

        start_time = time.time()

        # Perform clustering
        labels = self.perform_clustering(embeddings_cudf)

        # Get number of clusters found
        n_clusters_found = len(labels.unique())

        optimization_time = time.time() - start_time

        logger.info(
            f"HDBSCAN found {n_clusters_found} clusters "
            f"(expected range: {min_clusters}-{max_clusters})"
        )

        # Warn if clusters found is outside expected range
        if n_clusters_found < min_clusters:
            logger.warning(
                f"Found fewer clusters ({n_clusters_found}) than minimum expected ({min_clusters})"
            )
        elif n_clusters_found > max_clusters:
            logger.warning(
                f"Found more clusters ({n_clusters_found}) than maximum expected ({max_clusters})"
            )

        return n_clusters_found, optimization_time

    def get_probabilities(self):
        """Get cluster membership probabilities if available."""
        if (
            hasattr(self.clusterer, "probabilities_")
            and self.clusterer.probabilities_ is not None
        ):
            if self.use_gpu:
                return self.clusterer.probabilities_.to_numpy()
            else:
                return self.clusterer.probabilities_
        return None

    def get_outlier_scores(self):
        """Get outlier scores if available."""
        if (
            hasattr(self.clusterer, "outlier_scores_")
            and self.clusterer.outlier_scores_ is not None
        ):
            if self.use_gpu:
                return self.clusterer.outlier_scores_.to_numpy()
            else:
                return self.clusterer.outlier_scores_
        return None


class ClusteringAlgorithmFactory:
    @staticmethod
    def get_algorithm(algorithm_name: str, use_gpu: bool = False, **kwargs):
        algorithm_name = algorithm_name.lower()

        if algorithm_name == "agglomerative":
            return AgglomerativeClusteringAlgorithm(use_gpu=use_gpu)
        elif algorithm_name == "kmeans":
            return KMeansClusteringAlgorithm()
        elif algorithm_name == "hdbscan":
            return HDBSCANClusteringAlgorithm(use_gpu=use_gpu, **kwargs)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm_name}")

    @staticmethod
    def get_supported_algorithms():
        """Get list of supported algorithm names."""
        return ["agglomerative", "kmeans", "hdbscan"]

    @staticmethod
    def get_algorithm_info(algorithm_name: str):
        """Get information about a specific algorithm."""
        algorithm_name = algorithm_name.lower()

        info = {
            "agglomerative": {
                "name": "Agglomerative Clustering",
                "type": "Hierarchical",
                "requires_n_clusters": True,
                "supports_gpu": True,
                "density_based": False,
            },
            "kmeans": {
                "name": "K-Means Clustering",
                "type": "Centroid-based",
                "requires_n_clusters": True,
                "supports_gpu": True,
                "density_based": False,
            },
            "hdbscan": {
                "name": "HDBSCAN",
                "type": "Density-based",
                "requires_n_clusters": False,
                "supports_gpu": True,
                "density_based": True,
                "automatic_clusters": True,
                "handles_noise": True,
                "provides_probabilities": True,
            },
        }

        return info.get(algorithm_name, {})

"""
CPU-based HDBSCAN implementation using scikit-learn.

This module implements HDBSCAN clustering optimized for CPU processing
"""

import time
import logging
import numpy as np
import pandas as pd
import platform
import psutil
from typing import Dict, Optional, Tuple, Union
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HDBSCANCPUClusterer:
    """
    CPU-optimized HDBSCAN clustering implementation with grid search and quality metrics.
    
    Includes platform-specific optimizations for ARM (Apple Silicon) and AMD systems.
    """

    # Parameter ranges for grid search
    MIN_SAMPLES_RANGE = [1, 3, 6, 9, 11]
    MIN_CLUSTER_SIZE_RANGE = [3, 5, 10, 15, 20]
    
    @staticmethod
    def _detect_system_info():
        """Detect system architecture and CPU details for optimization."""
        system_info = {
            'architecture': platform.machine().lower(),
            'processor': platform.processor().lower(),
            'system': platform.system().lower(),
            'cpu_count': psutil.cpu_count(logical=False),
            'logical_cpu_count': psutil.cpu_count(logical=True),
        }
        
        # Detect specific CPU types
        processor_lower = system_info['processor'].lower()
        if 'arm' in system_info['architecture'] or 'apple' in processor_lower:
            system_info['cpu_type'] = 'arm'
        elif 'amd' in processor_lower or 'ryzen' in processor_lower or 'epyc' in processor_lower:
            system_info['cpu_type'] = 'amd'
        elif 'intel' in processor_lower or 'xeon' in processor_lower:
            system_info['cpu_type'] = 'intel'
        else:
            system_info['cpu_type'] = 'unknown'
            
        # Try alternative detection methods if processor string is empty
        if system_info['cpu_type'] == 'unknown':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'amd' in cpuinfo or 'ryzen' in cpuinfo:
                        system_info['cpu_type'] = 'amd'
                    elif 'intel' in cpuinfo:
                        system_info['cpu_type'] = 'intel'
            except:
                pass  # Ignore if /proc/cpuinfo doesn't exist
            
        return system_info

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_method: str = "eom",
        metric: str = "euclidean",
        merge_clusters: bool = False,
        merge_threshold: float = 0.95,
        standardize_features: bool = True,
    ):
        """
        Initialize HDBSCAN clusterer.

        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Number of samples in neighborhood for core point
            cluster_selection_method: Method for selecting clusters ('eom' or 'leaf')
            metric: Distance metric ('euclidean', 'cosine', etc.)
            merge_clusters: Whether to merge similar clusters
            merge_threshold: Cosine similarity threshold for merging
            standardize_features: Whether to standardize features before clustering
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.merge_clusters = merge_clusters
        self.merge_threshold = merge_threshold
        self.standardize_features = standardize_features

        # Results storage
        self.labels_ = None
        self.probabilities_ = None
        self.outlier_scores_ = None
        self.centroids_ = {}
        self.performance_metrics_ = {}

        # For feature standardization
        self.scaler_ = StandardScaler() if standardize_features else None

    def fit(self, X: np.ndarray) -> "HDBSCANCPUClusterer":
        """
        Fit HDBSCAN to the data with grid search optimization.

        Args:
            X: Input data matrix of shape (n_samples, n_features)

        Returns:
            self: Fitted clusterer instance
        """
        start_time = time.time()
        
        # Detect system for platform-specific optimizations
        system_info = self._detect_system_info()
        self._system_info = system_info  # Store for fallback use
        logger.info(f"Detected system: {system_info['cpu_type']} ({system_info['architecture']})")
        logger.info(
            f"Starting HDBSCAN clustering on {X.shape[0]} samples with {X.shape[1]} features"
        )

        # Standardize features if requested
        if self.standardize_features:
            X = self.scaler_.fit_transform(X)

        # Handle very small datasets
        if X.shape[0] <= 2:
            logger.warning("Dataset too small for HDBSCAN clustering")
            self.labels_ = np.zeros(X.shape[0], dtype=int)
            self.probabilities_ = np.ones(X.shape[0])
            self.outlier_scores_ = np.zeros(X.shape[0])
            return self

        # For very large datasets, warn about potential memory issues
        if X.shape[0] > 10000:
            logger.warning(
                f"Large dataset detected ({X.shape[0]} samples). This may require significant memory and time."
            )
            logger.warning(
                "Consider using a smaller sample size if you encounter memory issues."
            )

        # Perform grid search for optimal parameters with system-specific settings
        best_params, best_labels, best_probabilities = self._grid_search(X, system_info)

        if best_params is None:
            logger.warning(
                "Grid search failed, attempting fallback clustering with default parameters"
            )
            try:
                # Try basic HDBSCAN with conservative parameters
                fallback_params = {
                    'min_cluster_size': min(10, X.shape[0] // 20),
                    'min_samples': 3,
                    'metric': "euclidean",
                    'n_jobs': 1,
                }
                
                # Platform-specific fallback parameters
                if hasattr(self, '_system_info'):
                    cpu_type = self._system_info.get('cpu_type')
                    if cpu_type == 'arm':
                        fallback_params.update({
                            'algorithm': 'ball_tree',
                            'leaf_size': 40,
                            'min_cluster_size': min(5, X.shape[0] // 30),
                        })
                        logger.info("Using ARM-optimized fallback parameters")
                    elif cpu_type == 'amd':
                        fallback_params.update({
                            'algorithm': 'brute',  # Most conservative for AMD
                            'leaf_size': 20,  # Small leaf size for stability
                            'min_cluster_size': min(3, X.shape[0] // 50),  # Very conservative
                        })
                        logger.info("Using AMD-optimized fallback parameters")
                
                fallback_clusterer = HDBSCAN(**fallback_params)
                fallback_clusterer.fit(X)
                self.labels_ = fallback_clusterer.labels_.copy()
                self.probabilities_ = fallback_clusterer.probabilities_.copy()
                self.outlier_scores_ = 1 - fallback_clusterer.probabilities_
                logger.info("Fallback clustering succeeded")
            except Exception as e:
                logger.error(f"Fallback clustering also failed: {e}")
                self.labels_ = np.zeros(X.shape[0], dtype=int)
                self.probabilities_ = np.ones(X.shape[0])
                self.outlier_scores_ = np.zeros(X.shape[0])
        else:
            logger.info(f"Best parameters: {best_params}")
            self.labels_ = best_labels
            self.probabilities_ = best_probabilities
            self.outlier_scores_ = 1 - best_probabilities

            # Convert noise points (-1) to separate cluster
            if -1 in self.labels_:
                next_cluster = self.labels_.max() + 1 if self.labels_.max() >= 0 else 0
                self.labels_[self.labels_ == -1] = next_cluster

            # Compute centroids
            self._compute_centroids(X)

            # Merge similar clusters if requested
            if self.merge_clusters:
                self._merge_similar_clusters(X)

        # Record performance metrics
        total_time = time.time() - start_time
        self.performance_metrics_ = {
            "total_time": total_time,
            "n_clusters": len(np.unique(self.labels_)),
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }

        logger.info(
            f"Clustering complete in {total_time:.2f}s, found {self.performance_metrics_['n_clusters']} clusters"
        )

        return self

    def _grid_search(
        self, X: np.ndarray, system_info: Dict = None
    ) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Perform grid search to find optimal HDBSCAN parameters.

        Args:
            X: Input data matrix

        Returns:
            Tuple of (best_params, best_labels, best_probabilities)
        """
        logger.info("Starting grid search for optimal parameters")

        best_score = float("-inf")
        best_params = None
        best_labels = None
        best_probabilities = None

        # Early stopping parameters
        early_stop_threshold = 0.85
        no_improvement_limit = 3
        no_improvement_count = 0

        # Determine parameter ranges based on dataset size with memory safety
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Platform-specific optimizations
        cpu_type = system_info.get('cpu_type') if system_info else 'unknown'
        
        if cpu_type == 'arm':
            # ARM/Apple Silicon specific optimizations - more conservative
            logger.info("Applying ARM/Apple Silicon optimizations")
            max_reasonable_cluster_size = min(n_samples // 15, 25)
            size_reduction_factor = 20 if n_samples > 3000 else 10
            search_threshold = 3000
        elif cpu_type == 'amd':
            # AMD-specific optimizations for known segfault issues
            logger.info("Applying AMD CPU optimizations")
            max_reasonable_cluster_size = min(n_samples // 12, 30)  # Conservative for AMD
            size_reduction_factor = 15  # More conservative than Intel
            search_threshold = 4000  # Lower threshold for AMD
        else:
            # Default Intel/unknown optimizations
            max_reasonable_cluster_size = min(n_samples // 10, 50)
            size_reduction_factor = 10
            search_threshold = 5000

        min_cluster_size_range = [
            mcs
            for mcs in self.MIN_CLUSTER_SIZE_RANGE
            if mcs <= max_reasonable_cluster_size and mcs < n_samples
        ]
        if not min_cluster_size_range:
            min_cluster_size_range = [min(5, n_samples // size_reduction_factor, n_samples - 1)]

        # For very large datasets, reduce the search space based on CPU type
        if n_samples > search_threshold:
            min_cluster_size_range = min_cluster_size_range[:2]  # Even smaller search space for ARM
            min_samples_range = self.MIN_SAMPLES_RANGE[:2]  # Use only first 2 values
        elif n_samples > 1000 and system_info and system_info.get('cpu_type') == 'arm':
            # Additional reduction for ARM at medium sizes
            min_cluster_size_range = min_cluster_size_range[:3]
            min_samples_range = self.MIN_SAMPLES_RANGE[:3]
        else:
            min_samples_range = self.MIN_SAMPLES_RANGE

        logger.info(
            f"Grid search parameters: cluster_sizes={min_cluster_size_range}, samples={min_samples_range}"
        )
        logger.info(f"Dataset: {n_samples} samples, {n_features} features")

        total_iterations = len(min_cluster_size_range) * len(min_samples_range)
        current_iteration = 0

        for min_cluster_size in min_cluster_size_range:
            for min_samples in min_samples_range:
                current_iteration += 1

                # Skip invalid parameter combinations
                if min_samples >= n_samples:
                    continue

                iteration_start = time.time()

                try:
                    # Validate parameters before creating clusterer
                    if min_cluster_size >= n_samples or min_samples >= n_samples:
                        logger.warning(
                            f"Skipping invalid parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}"
                        )
                        continue

                    # Memory check for large datasets
                    memory_estimate = (n_samples * n_features * 8) / (
                        1024**3
                    )  # GB estimate
                    if memory_estimate > 8:  # More than 8GB
                        logger.warning(
                            f"Large memory requirement estimated: {memory_estimate:.2f}GB"
                        )

                    # Create HDBSCAN with platform-specific optimizations
                    hdbscan_params = {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'metric': self.metric,
                        'cluster_selection_method': self.cluster_selection_method,
                        'n_jobs': 1,  # Single threaded for consistency
                    }
                    
                    # Platform-specific parameters
                    cpu_type = system_info.get('cpu_type') if system_info else 'unknown'
                    
                    if cpu_type == 'arm':
                        hdbscan_params.update({
                            'algorithm': 'ball_tree',  # More stable on ARM
                            'leaf_size': 40,  # Larger leaf size for ARM efficiency
                        })
                    elif cpu_type == 'amd':
                        hdbscan_params.update({
                            'algorithm': 'ball_tree',  # Often more stable than 'auto' on AMD
                            'leaf_size': 30,  # Conservative leaf size for AMD
                        })
                    else:
                        hdbscan_params.update({
                            'algorithm': 'auto',  # Let HDBSCAN choose for Intel/unknown
                        })
                    
                    clusterer = HDBSCAN(**hdbscan_params)

                    logger.info(
                        f"Fitting HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}"
                    )
                    clusterer.fit(X)

                    # Calculate quality score
                    score = self._calculate_quality_score(
                        X, clusterer.labels_, clusterer.probabilities_
                    )

                    if score is not None and score > best_score:
                        best_score = score
                        best_params = {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                        }
                        best_labels = clusterer.labels_.copy()
                        best_probabilities = clusterer.probabilities_.copy()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    iteration_time = time.time() - iteration_start
                    logger.info(
                        f"Grid search [{current_iteration}/{total_iterations}] - "
                        f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, "
                        f"score: {score:.3f}, time: {iteration_time:.2f}s"
                    )

                    # Early stopping checks
                    if best_score >= early_stop_threshold:
                        logger.info(
                            f"Early stopping: score {best_score:.3f} >= threshold"
                        )
                        break
                    if no_improvement_count >= no_improvement_limit:
                        logger.info(
                            f"Early stopping: no improvement for {no_improvement_limit} iterations"
                        )
                        break

                except Exception as e:
                    logger.warning(
                        f"Failed for min_cluster_size={min_cluster_size}, min_samples={min_samples}: {e}"
                    )
                    continue

            # Break outer loop if early stopping triggered
            if (
                best_score >= early_stop_threshold
                or no_improvement_count >= no_improvement_limit
            ):
                break

        return best_params, best_labels, best_probabilities

    def _calculate_quality_score(
        self, X: np.ndarray, labels: np.ndarray, probabilities: np.ndarray
    ) -> Optional[float]:
        """
        Calculate composite quality score for clustering results.

        Uses the same scoring methodology as the production implementation:
        - Calinski-Harabasz score (normalized)
        - Intra-cluster cosine similarity
        - Cluster size ratio score

        Args:
            X: Input data matrix
            labels: Cluster labels
            probabilities: Cluster membership probabilities

        Returns:
            Composite quality score or None if calculation fails
        """
        try:
            unique_labels = np.unique(labels[labels != -1])
            n_labels = len(unique_labels)
            n_samples = X.shape[0]

            if n_labels <= 1 or n_samples <= n_labels:
                return 0.0

            # 1. Calinski-Harabasz score (normalized)
            ch_score = calinski_harabasz_score(X, labels)
            ch_score_norm = 2 / (1 + np.exp(-ch_score / 1000)) - 1

            # 2. Intra-cluster cosine similarity
            cluster_similarities = []
            for label in unique_labels:
                if label == -1:
                    continue
                cluster_mask = labels == label
                cluster_points = X[cluster_mask]
                if len(cluster_points) < 2:
                    continue

                # Vectorized cosine similarity calculation
                cosine_distances = pairwise_distances(cluster_points, metric="cosine")
                n = cosine_distances.shape[0]
                if n > 1:
                    upper_indices = np.triu_indices(n, k=1)
                    upper_triangle = cosine_distances[upper_indices]
                    cluster_similarities.append(np.mean(upper_triangle))

            cos_score = np.mean(cluster_similarities) if cluster_similarities else 1.0

            # 3. Cluster size ratio score
            n_clusters = len(unique_labels)
            ratio = n_clusters / n_samples if n_samples > 0 else 0
            optimal_ratio = np.sqrt(n_samples) / n_samples if n_samples > 0 else 0
            cluster_ratio_score = (
                1 / (1 + np.exp(-10 * (ratio - optimal_ratio))) if n_samples > 0 else 0
            )

            # Composite score (same weighting as production)
            cos_norm = 1 - cos_score
            final_score = (
                0.25 * ch_score_norm + 0.25 * cos_norm + 0.5 * cluster_ratio_score
            )

            return final_score

        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return None

    def _compute_centroids(self, X: np.ndarray):
        """Compute centroids for each cluster."""
        self.centroids_ = {}
        unique_clusters = np.unique(self.labels_)

        for cluster_id in unique_clusters:
            cluster_mask = self.labels_ == cluster_id
            cluster_points = X[cluster_mask]
            if len(cluster_points) > 0:
                self.centroids_[cluster_id] = np.mean(cluster_points, axis=0)

    def _merge_similar_clusters(self, X: np.ndarray):
        """
        Merge clusters with high centroid similarity.

        Args:
            X: Input data matrix
        """
        if len(self.centroids_) <= 1:
            return

        logger.info(f"Merging similar clusters (threshold: {self.merge_threshold})")

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            centroid_ids = list(self.centroids_.keys())
            centroid_matrix = np.vstack([self.centroids_[cid] for cid in centroid_ids])

            # Compute pairwise similarities
            similarities = cosine_similarity(centroid_matrix)
            np.fill_diagonal(similarities, 0)

            while len(similarities) > 1:
                # Find maximum similarity pair
                i, j = np.unravel_index(similarities.argmax(), similarities.shape)
                max_similarity = similarities[i, j]

                if max_similarity < self.merge_threshold:
                    break

                # Merge cluster j into cluster i
                old_cluster = centroid_ids[j]
                new_cluster = centroid_ids[i]

                # Update labels
                self.labels_[self.labels_ == old_cluster] = new_cluster

                # Update centroids
                new_cluster_mask = self.labels_ == new_cluster
                merged_points = X[new_cluster_mask]
                self.centroids_[new_cluster] = np.mean(merged_points, axis=0)
                del self.centroids_[old_cluster]

                # Update similarity matrix
                centroid_ids.pop(j)
                similarities = np.delete(similarities, j, axis=0)
                similarities = np.delete(similarities, j, axis=1)

            logger.info(f"After merging: {len(self.centroids_)} clusters")

        except Exception as e:
            logger.error(f"Error merging clusters: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X: New data to predict

        Returns:
            Predicted cluster labels
        """
        if self.labels_ is None:
            raise ValueError("Clusterer has not been fitted yet")

        if self.standardize_features and self.scaler_ is not None:
            X = self.scaler_.transform(X)

        # Simple nearest centroid prediction
        if not self.centroids_:
            return np.zeros(X.shape[0], dtype=int)

        labels = np.zeros(X.shape[0], dtype=int)
        centroid_ids = list(self.centroids_.keys())
        centroid_matrix = np.vstack([self.centroids_[cid] for cid in centroid_ids])

        for i, point in enumerate(X):
            distances = np.linalg.norm(centroid_matrix - point, axis=1)
            closest_idx = np.argmin(distances)
            labels[i] = centroid_ids[closest_idx]

        return labels

    def get_cluster_info(self) -> Dict:
        """
        Get comprehensive information about the clustering results.

        Returns:
            Dictionary containing cluster statistics and performance metrics
        """
        if self.labels_ is None:
            return {}

        unique_labels, counts = np.unique(self.labels_, return_counts=True)

        cluster_info = {
            "n_clusters": len(unique_labels),
            "cluster_sizes": dict(zip(unique_labels.tolist(), counts.tolist())),
            "total_samples": len(self.labels_),
            "performance_metrics": self.performance_metrics_,
            "parameters": {
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "metric": self.metric,
                "merge_clusters": self.merge_clusters,
                "standardize_features": self.standardize_features,
            },
        }

        return cluster_info

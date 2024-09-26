import time
from cuml.cluster import AgglomerativeClustering as CumlAgglomerativeClustering
from cuml.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerativeClustering
from sklearn.metrics import silhouette_score
import cudf
from utils.clustering_base import ClusteringBase

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

    def find_optimal_clusters(self, embeddings_cudf: cudf.DataFrame, min_clusters: int, max_clusters: int):
        start_time = time.time()
        best_score = float('-inf')
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

    def find_optimal_clusters(self, embeddings_cudf: cudf.DataFrame, min_clusters: int, max_clusters: int):
        start_time = time.time()
        best_score = float('-inf')
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

class ClusteringAlgorithmFactory:
    @staticmethod
    def get_algorithm(algorithm_name: str, use_gpu: bool = False):
        if algorithm_name.lower() == "agglomerative":
            return AgglomerativeClusteringAlgorithm(use_gpu=use_gpu)
        elif algorithm_name.lower() == "kmeans":
            return KMeansClusteringAlgorithm()
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm_name}")

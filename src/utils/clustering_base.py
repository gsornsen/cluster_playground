from abc import ABC, abstractmethod
import cudf

class ClusteringBase(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings_cudf: cudf.DataFrame, n_clusters: int):
        """
        Abstract method to perform clustering.
        Subclasses must implement this method.
        """
        pass

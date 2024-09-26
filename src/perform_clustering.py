import cudf
import asyncio
import time
import argparse
from utils.embedding_utils import (
    calculate_text_hash, 
    save_embeddings_to_cache, 
    load_embeddings_from_cache, 
    get_embeddings_in_parallel
)
from utils.preprocessing_utils import fetch_dataset, preprocess_data
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from utils.clustering_algorithms import ClusteringAlgorithmFactory
from tabulate import tabulate

async def main(sample_size: int, use_gpu: bool, iterations: int, min_clusters: int, max_clusters: int, algorithm: str):
    print(f"Using sample size: {sample_size}")
    print(f"Clustering algorithm: {algorithm}")
    print("Fetching the dataset...")
    dataset = fetch_dataset()

    print("Preprocessing the dataset...")
    texts = preprocess_data(dataset)

    text_hash = calculate_text_hash(texts[:sample_size])
    
    cached_embeddings = load_embeddings_from_cache(text_hash)
    if cached_embeddings:
        print("Loaded embeddings from cache.")
        embeddings = cached_embeddings
    else:
        print("Fetching OpenAI embeddings in parallel...")
        embeddings = await get_embeddings_in_parallel(texts[:sample_size])
        
        if embeddings:
            save_embeddings_to_cache(embeddings, text_hash)
        else:
            print("Failed to fetch embeddings. Exiting.")
            return
    
    embeddings_cudf = cudf.DataFrame(embeddings, dtype=float)

    print(f"Performing {algorithm} Clustering using {'GPU' if use_gpu else 'CPU'} for {iterations} iterations...")
    clustering_algorithm = ClusteringAlgorithmFactory.get_algorithm(algorithm, use_gpu=use_gpu)

    print(f"Finding optimal number of clusters between {min_clusters} and {max_clusters}...")
    optimal_clusters, optimization_time = clustering_algorithm.find_optimal_clusters(embeddings_cudf, min_clusters, max_clusters)
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Time taken to find optimal clusters: {optimization_time:.4f} seconds")

    table_data = []
    cluster_data = ["GPU" if use_gpu else "CPU", sample_size, algorithm, optimal_clusters, f"{optimization_time:.4f}"]
    total_execution_time = 0
    total_silhouette_score = 0
    total_davies_bouldin_score = 0
    total_calinski_harabasz_score = 0
    max_silhouette_score = float('-inf')
    max_davies_bouldin_score = float('-inf')
    max_calinski_harabasz_score = float('-inf')

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        start_time = time.time()
        labels_cudf = clustering_algorithm.perform_clustering(embeddings_cudf, n_clusters=optimal_clusters)
        end_time = time.time()

        execution_time = end_time - start_time
        total_execution_time += execution_time

        # Convert cuDF DataFrame to numpy array for sklearn metrics
        embeddings_np = embeddings_cudf.to_numpy()
        labels_np = labels_cudf.to_numpy()

        # Calculate metrics
        silhouette = silhouette_score(embeddings_np, labels_np)
        davies_bouldin = davies_bouldin_score(embeddings_np, labels_np)
        calinski_harabasz = calinski_harabasz_score(embeddings_np, labels_np)

        total_silhouette_score += silhouette
        total_davies_bouldin_score += davies_bouldin
        total_calinski_harabasz_score += calinski_harabasz

        max_silhouette_score = max(max_silhouette_score, silhouette)
        max_davies_bouldin_score = max(max_davies_bouldin_score, davies_bouldin)
        max_calinski_harabasz_score = max(max_calinski_harabasz_score, calinski_harabasz)

        print(f"  Iteration completed in {execution_time:.4f} seconds")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz:.4f}")

    average_execution_time = total_execution_time / iterations
    average_silhouette_score = total_silhouette_score / iterations
    average_davies_bouldin_score = total_davies_bouldin_score / iterations
    average_calinski_harabasz_score = total_calinski_harabasz_score / iterations

    print(f"\nResults for {optimal_clusters} clusters:")
    print(f"Average clustering time over {iterations} iterations: {average_execution_time:.4f} seconds")
    print(f"Average Silhouette Score: {average_silhouette_score:.4f}")
    print(f"Max Silhouette Score: {max_silhouette_score:.4f}")
    print(f"Average Davies-Bouldin Score: {average_davies_bouldin_score:.4f}")
    print(f"Max Davies-Bouldin Score: {max_davies_bouldin_score:.4f}")
    print(f"Average Calinski-Harabasz Score: {average_calinski_harabasz_score:.4f}")
    print(f"Max Calinski-Harabasz Score: {max_calinski_harabasz_score:.4f}")

    cluster_data.extend([
        f"{average_execution_time:.4f}",
        f"{average_silhouette_score:.4f}",
        f"{max_silhouette_score:.4f}",
        f"{average_davies_bouldin_score:.4f}",
        f"{max_davies_bouldin_score:.4f}",
        f"{average_calinski_harabasz_score:.4f}",
        f"{max_calinski_harabasz_score:.4f}"
    ])

    table_data.append(cluster_data)

    headers = [
        "Hardware",
        "Sample Size",
        "Algorithm",
        "Clusters",
        "Optimization Time (s)",
        "Avg Time (s)",
        "Avg Silhouette",
        "Max Silhouette",
        "Avg Davies-Bouldin",
        "Max Davies-Bouldin",
        "Avg Calinski-Harabasz",
        "Max Calinski-Harabasz"
    ]

    print("\nClustering Results Summary:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform clustering on text data.")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of samples to use for clustering (default: 500)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for clustering (default: False)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of clustering iterations to run (default: 1)")
    parser.add_argument("--min_clusters", type=int, default=2, help="Minimum number of clusters (default: 2)")
    parser.add_argument("--max_clusters", type=int, default=10, help="Maximum number of clusters (default: 10)")
    parser.add_argument("--algorithm", type=str, default="agglomerative", choices=["agglomerative", "kmeans"], help="Clustering algorithm to use (default: agglomerative)")
    args = parser.parse_args()

    asyncio.run(main(args.sample_size, args.use_gpu, args.iterations, args.min_clusters, args.max_clusters, args.algorithm))

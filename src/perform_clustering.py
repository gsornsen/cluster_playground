import cudf
import asyncio
import time
import argparse
from utils.embedding_utils import (
    calculate_text_hash,
    save_embeddings_to_cache,
    load_embeddings_from_cache,
    get_embeddings_in_parallel,
)
from utils.preprocessing_utils import fetch_dataset, preprocess_data
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import numpy as np
from utils.clustering_algorithms import ClusteringAlgorithmFactory
from tabulate import tabulate
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main(
    sample_size: int,
    use_gpu: bool,
    iterations: int,
    min_clusters: int,
    max_clusters: int,
    algorithm: str,
):
    print(f"Using sample size: {sample_size}")
    print(f"Clustering algorithm: {algorithm}")

    # Get algorithm info to check if it's density-based
    algorithm_info = ClusteringAlgorithmFactory.get_algorithm_info(algorithm)
    is_density_based = algorithm_info.get("density_based", False)

    if is_density_based:
        print(
            f"Note: {algorithm} is a density-based algorithm that automatically determines the number of clusters."
        )
        print(
            f"The min_clusters ({min_clusters}) and max_clusters ({max_clusters}) parameters will be used for validation only."
        )

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

    print(
        f"Performing {algorithm} Clustering using {'GPU' if use_gpu else 'CPU'} for {iterations} iterations..."
    )

    # Pass additional parameters for HDBSCAN
    if algorithm.lower() == "hdbscan":
        # Calculate reasonable min_cluster_size based on sample size with safety caps
        hdbscan_min_cluster_size = max(
            3, min(sample_size // 100, 25)
        )  # 1% of sample size, minimum 3, maximum 25 for memory safety
        clustering_algorithm = ClusteringAlgorithmFactory.get_algorithm(
            algorithm,
            use_gpu=use_gpu,
            min_cluster_size=hdbscan_min_cluster_size,
            merge_clusters=True,
            standardize_features=True,
            skip_grid_search=True,  # Bypass grid search for testing
        )
        print(
            f"HDBSCAN parameters: min_cluster_size={hdbscan_min_cluster_size}, merge_clusters=True"
        )
    else:
        clustering_algorithm = ClusteringAlgorithmFactory.get_algorithm(
            algorithm, use_gpu=use_gpu
        )

    if is_density_based:
        print("Running density-based clustering (automatic cluster detection)...")
    else:
        print(
            f"Finding optimal number of clusters between {min_clusters} and {max_clusters}..."
        )

    optimal_clusters, optimization_time = clustering_algorithm.find_optimal_clusters(
        embeddings_cudf, min_clusters, max_clusters
    )
    print(
        f"{'Clusters found' if is_density_based else 'Optimal number of clusters'}: {optimal_clusters}"
    )
    print(
        f"Time taken to {'find clusters' if is_density_based else 'find optimal clusters'}: {optimization_time:.4f} seconds"
    )

    table_data = []
    cluster_data = [
        "GPU" if use_gpu else "CPU",
        sample_size,
        algorithm,
        optimal_clusters,
        f"{optimization_time:.4f}",
    ]
    total_execution_time = 0
    total_silhouette_score = 0
    total_davies_bouldin_score = 0
    total_calinski_harabasz_score = 0
    max_silhouette_score = float("-inf")
    max_davies_bouldin_score = float("-inf")
    max_calinski_harabasz_score = float("-inf")

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")

        start_time = time.time()
        labels_cudf = clustering_algorithm.perform_clustering(
            embeddings_cudf, n_clusters=optimal_clusters
        )
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
        max_calinski_harabasz_score = max(
            max_calinski_harabasz_score, calinski_harabasz
        )

        print(f"  Iteration completed in {execution_time:.4f} seconds")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz:.4f}")

        # Print HDBSCAN-specific metrics if available
        if algorithm.lower() == "hdbscan":
            probabilities = clustering_algorithm.get_probabilities()
            outlier_scores = clustering_algorithm.get_outlier_scores()

            if probabilities is not None:
                avg_probability = np.mean(probabilities)
                min_probability = np.min(probabilities)
                print(f"  Average Cluster Probability: {avg_probability:.4f}")
                print(f"  Minimum Cluster Probability: {min_probability:.4f}")

            if outlier_scores is not None:
                avg_outlier_score = np.mean(outlier_scores)
                max_outlier_score = np.max(outlier_scores)
                print(f"  Average Outlier Score: {avg_outlier_score:.4f}")
                print(f"  Maximum Outlier Score: {max_outlier_score:.4f}")

    average_execution_time = total_execution_time / iterations
    average_silhouette_score = total_silhouette_score / iterations
    average_davies_bouldin_score = total_davies_bouldin_score / iterations
    average_calinski_harabasz_score = total_calinski_harabasz_score / iterations

    print(f"\nResults for {optimal_clusters} clusters:")
    print(
        f"Average clustering time over {iterations} iterations: {average_execution_time:.4f} seconds"
    )
    print(f"Average Silhouette Score: {average_silhouette_score:.4f}")
    print(f"Max Silhouette Score: {max_silhouette_score:.4f}")
    print(f"Average Davies-Bouldin Score: {average_davies_bouldin_score:.4f}")
    print(f"Max Davies-Bouldin Score: {max_davies_bouldin_score:.4f}")
    print(f"Average Calinski-Harabasz Score: {average_calinski_harabasz_score:.4f}")
    print(f"Max Calinski-Harabasz Score: {max_calinski_harabasz_score:.4f}")

    cluster_data.extend(
        [
            f"{average_execution_time:.4f}",
            f"{average_silhouette_score:.4f}",
            f"{max_silhouette_score:.4f}",
            f"{average_davies_bouldin_score:.4f}",
            f"{max_davies_bouldin_score:.4f}",
            f"{average_calinski_harabasz_score:.4f}",
            f"{max_calinski_harabasz_score:.4f}",
        ]
    )

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
        "Max Calinski-Harabasz",
    ]

    print("\nClustering Results Summary:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def run_multiple_algorithms(
    algorithms: list,
    sample_size: int,
    use_gpu: bool,
    iterations: int,
    min_clusters: int,
    max_clusters: int,
):
    """Run clustering with multiple algorithms for comparison."""
    print(f"Running comparison with algorithms: {', '.join(algorithms)}")
    print("=" * 60)

    for algorithm in algorithms:
        print(f"\n{'=' * 20} Running {algorithm.upper()} {'=' * 20}")
        try:
            asyncio.run(
                main(
                    sample_size,
                    use_gpu,
                    iterations,
                    min_clusters,
                    max_clusters,
                    algorithm,
                )
            )
        except Exception as e:
            logger.error(f"Failed to run {algorithm}: {e}")
        print("=" * 60)


if __name__ == "__main__":
    # Get supported algorithms dynamically
    supported_algorithms = ClusteringAlgorithmFactory.get_supported_algorithms()

    parser = argparse.ArgumentParser(
        description="Perform clustering on text data with CPU/GPU comparison capabilities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available algorithms: {", ".join(supported_algorithms)}

Algorithm types:
  - agglomerative: Hierarchical clustering (requires n_clusters)
  - kmeans: Centroid-based clustering (requires n_clusters) 
  - hdbscan: Density-based clustering (automatic cluster detection)

Examples:
  # Run single algorithm
  python src/perform_clustering.py --algorithm hdbscan --use_gpu --sample_size 1000
  
  # Run all algorithms for comparison  
  python src/perform_clustering.py --algorithms all --sample_size 500
  
  # Run specific algorithms
  python src/perform_clustering.py --algorithms hdbscan kmeans --use_gpu
        """,
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="Number of samples to use for clustering (default: 500)",
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for clustering (default: False)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of clustering iterations to run (default: 1)",
    )
    parser.add_argument(
        "--min_clusters",
        type=int,
        default=2,
        help="Minimum number of clusters (default: 2)",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=10,
        help="Maximum number of clusters (default: 10)",
    )

    # Algorithm selection - either single algorithm or multiple
    algorithm_group = parser.add_mutually_exclusive_group()
    algorithm_group.add_argument(
        "--algorithm",
        type=str,
        choices=supported_algorithms,
        help="Single clustering algorithm to use",
    )
    algorithm_group.add_argument(
        "--algorithms",
        nargs="+",
        help=f"Multiple algorithms to run. Use 'all' for all algorithms, or specify: {', '.join(supported_algorithms)}",
    )

    parser.add_argument(
        "--list_algorithms",
        action="store_true",
        help="List all supported algorithms with their properties",
    )

    args = parser.parse_args()

    # Handle list algorithms request
    if args.list_algorithms:
        print("Supported Clustering Algorithms:")
        print("=" * 50)
        for alg_name in supported_algorithms:
            info = ClusteringAlgorithmFactory.get_algorithm_info(alg_name)
            print(f"\n{alg_name.upper()}:")
            print(f"  Name: {info.get('name', 'N/A')}")
            print(f"  Type: {info.get('type', 'N/A')}")
            print(f"  Requires n_clusters: {info.get('requires_n_clusters', 'N/A')}")
            print(f"  Supports GPU: {info.get('supports_gpu', 'N/A')}")
            print(f"  Density-based: {info.get('density_based', False)}")
            if info.get("automatic_clusters"):
                print(f"  Automatic cluster detection: Yes")
            if info.get("handles_noise"):
                print(f"  Handles noise/outliers: Yes")
            if info.get("provides_probabilities"):
                print(f"  Provides probabilities: Yes")
        exit(0)

    # Determine which algorithms to run
    if args.algorithms:
        if "all" in args.algorithms:
            algorithms_to_run = supported_algorithms
        else:
            # Validate algorithm names
            invalid_algorithms = [
                alg for alg in args.algorithms if alg not in supported_algorithms
            ]
            if invalid_algorithms:
                parser.error(
                    f"Invalid algorithms: {invalid_algorithms}. Supported: {supported_algorithms}"
                )
            algorithms_to_run = args.algorithms

        # Run multiple algorithms
        run_multiple_algorithms(
            algorithms_to_run,
            args.sample_size,
            args.use_gpu,
            args.iterations,
            args.min_clusters,
            args.max_clusters,
        )

    elif args.algorithm:
        # Run single algorithm
        asyncio.run(
            main(
                args.sample_size,
                args.use_gpu,
                args.iterations,
                args.min_clusters,
                args.max_clusters,
                args.algorithm,
            )
        )

    else:
        # Default to agglomerative if nothing specified
        print("No algorithm specified, defaulting to 'agglomerative'")
        print("Use --list_algorithms to see all available algorithms")
        asyncio.run(
            main(
                args.sample_size,
                args.use_gpu,
                args.iterations,
                args.min_clusters,
                args.max_clusters,
                "agglomerative",
            )
        )

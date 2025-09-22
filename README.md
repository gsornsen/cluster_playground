## Project Overview

This repository is designed to test the efficacy and efficiency of different clustering algorithms using both CPU and GPU implementations. The main goals are:

1. Compare the performance of clustering algorithms between CPU and GPU usage.
2. Evaluate the quality of clustering results using various metrics.
3. Demonstrate the benefits of parallelization and GPU acceleration in data processing and machine learning tasks.

### 🚀 Quick Start

```bash
# List available algorithms
python src/perform_clustering.py --list_algorithms

# Run HDBSCAN with automatic cluster detection on GPU
python src/perform_clustering.py --algorithm hdbscan --use_gpu --sample_size 1000

# Compare all algorithms
python src/perform_clustering.py --algorithms all --sample_size 500
```

### Benefits of Parallelization and GPU Usage

1. **Faster Processing**: GPU-accelerated clustering can significantly reduce computation time, especially for large datasets.
2. **Scalability**: The parallel nature of GPUs allows for efficient processing of high-dimensional data and large sample sizes.
3. **Real-time Analysis**: Faster clustering enables near real-time analysis of streaming data or rapid iteration in exploratory data analysis.
4. **Energy Efficiency**: GPUs can often perform clustering tasks more energy-efficiently than CPUs for large-scale problems.


## Setup

1. Install Miniconda and create the conda environment:
   ```
   make conda
   ```
   After installation, restart your terminal or run `source $HOME/.bashrc`.

2. Create the conda environment:
   ```
   make env
   ```

3. Activate the environment:
   ```
   conda activate rapids-24.08
   ```

4. To update the environment (if needed):
   ```
   make update
   ```

5. To remove the environment:
   ```
   make clean
   ```

## Usage

To run the clustering algorithm, use the `perform_clustering.py` script:

```bash
python src/perform_clustering.py [options]
```

### Basic Options
- `--sample_size`: Number of samples to use for clustering (default: 500)
- `--use_gpu`: Use GPU for clustering (default: False)
- `--iterations`: Number of clustering iterations to run (default: 1)
- `--min_clusters`: Minimum number of clusters (default: 2)
- `--max_clusters`: Maximum number of clusters (default: 10)

### Algorithm Selection
- `--algorithm`: Single clustering algorithm to use (choices: "agglomerative", "kmeans", "hdbscan")
- `--algorithms`: Multiple algorithms to run for comparison (e.g., "hdbscan kmeans" or "all")
- `--list_algorithms`: List all supported algorithms with their properties

### Examples

#### Single Algorithm Examples

```bash
# Run K-means clustering on GPU
python src/perform_clustering.py --algorithm kmeans --use_gpu --sample_size 1000 --iterations 5

# Run HDBSCAN (density-based clustering with automatic cluster detection)
python src/perform_clustering.py --algorithm hdbscan --use_gpu --sample_size 1000

# Run Agglomerative clustering on CPU
python src/perform_clustering.py --algorithm agglomerative --sample_size 500
```

#### Multiple Algorithm Comparison

```bash
# Compare all algorithms on GPU
python src/perform_clustering.py --algorithms all --use_gpu --sample_size 500

# Compare specific algorithms
python src/perform_clustering.py --algorithms hdbscan kmeans --use_gpu --sample_size 1000

# Compare CPU vs GPU performance (run separately)
python src/perform_clustering.py --algorithms hdbscan --sample_size 1000
python src/perform_clustering.py --algorithms hdbscan --use_gpu --sample_size 1000
```

#### List Available Algorithms

```bash
# See all supported algorithms and their properties
python src/perform_clustering.py --list_algorithms
```

### Currently Supported Algorithms:

1. **Agglomerative Clustering**
   - Type: Hierarchical clustering
   - Requires: Pre-specified number of clusters
   - Best for: Small to medium datasets, when cluster hierarchy is important

2. **K-Means Clustering** 
   - Type: Centroid-based clustering
   - Requires: Pre-specified number of clusters  
   - Best for: Large datasets, spherical clusters

3. **HDBSCAN** ⭐ *New!*
   - Type: Density-based clustering
   - **Automatic cluster detection** - no need to specify cluster count
   - Handles noise and outliers
   - Provides cluster membership probabilities
   - Based on production implementation from Lattice text analysis pipeline
   - Best for: Complex cluster shapes, unknown cluster count, noisy data

### Algorithm-Specific Features

#### HDBSCAN Special Features
- **Automatic Parameter Optimization**: Uses grid search to find optimal parameters
- **Cluster Quality Scoring**: Composite scoring using Calinski-Harabasz, cosine similarity, and cluster ratio
- **Cluster Merging**: Optional merging of similar clusters based on centroid similarity
- **Noise Handling**: Automatically identifies and handles outliers
- **Probability Scores**: Provides membership probabilities and outlier scores
- **Feature Standardization**: Optional z-score normalization of input features

When using HDBSCAN, the `min_clusters` and `max_clusters` parameters are used only for validation warnings, as HDBSCAN automatically determines the optimal number of clusters.

### Memory Management and Large Datasets

The HDBSCAN implementation includes several optimizations for handling large datasets:

- **Parameter Optimization**: For datasets > 5,000 samples, the grid search space is automatically reduced to prevent memory issues
- **Memory Safety**: Cluster size parameters are capped at reasonable values to prevent segmentation faults
- **Fallback Clustering**: If grid search fails, the system attempts basic HDBSCAN with conservative parameters
- **Memory Estimation**: Warns when dataset size may require significant memory (>6-8GB)

#### Recommended Sample Sizes:
- **Small datasets**: < 1,000 samples - Full grid search with all parameter combinations
- **Medium datasets**: 1,000 - 5,000 samples - Standard optimization with reduced search space
- **Large datasets**: > 5,000 samples - Conservative parameters with memory monitoring

#### For Very Large Datasets:
If you encounter memory issues or segmentation faults with large datasets:
1. Reduce sample size: `--sample_size 2000` 
2. Use CPU implementation first: Remove `--use_gpu` flag
3. Monitor system memory usage during clustering

#### ARM/Apple Silicon Compatibility:
The implementation includes specific optimizations for ARM-based systems (Apple Silicon):
- **Automatic Detection**: System architecture is detected automatically
- **Conservative Parameters**: More conservative clustering parameters for ARM stability
- **Algorithm Selection**: Uses `generic` algorithm instead of `best` for better compatibility
- **Reduced Search Space**: Smaller grid search space to prevent memory issues
- **Optimized Leaf Size**: Larger leaf size (40) for ARM efficiency

**Recommended ARM Settings:**
- Start with smaller sample sizes (< 3,000 samples)
- CPU implementation is often more stable than GPU on Apple Silicon
- Monitor memory usage as ARM systems may have different memory patterns


## Example Clustering Results Summary

> [!NOTE]
> These were run on an Intel i7-9700k CPU with an Nvidia RTX 4090 GPU


## Clustering Metrics Overview

### 1. **Silhouette Score**
The Silhouette score measures how similar an object is to its own cluster compared to other clusters. It ranges from `-1` to `1`:
- **Good:** A score close to `1` indicates that samples are well-matched to their own cluster and poorly matched to neighboring clusters.
- **Neutral:** A score of `0` indicates that the sample is on or very close to the decision boundary between two neighboring clusters.
- **Bad:** A score close to `-1` means that samples have been misclassified and are assigned to the wrong cluster.

### 2. **Davies-Bouldin Index**
The Davies-Bouldin Index is a measure of cluster quality, where a lower value indicates better clustering. It compares the ratio of within-cluster scatter with between-cluster separation.
- **Good:** A lower Davies-Bouldin index signifies more distinct clusters with better separation.
- **Bad:** Higher values indicate overlapping or poorly separated clusters.

### 3. **Calinski-Harabasz Index**
The Calinski-Harabasz Index, also known as the Variance Ratio Criterion, evaluates how well the clusters are separated. It is the ratio of the sum of between-cluster dispersion and within-cluster dispersion.
- **Good:** Higher values indicate well-separated clusters with compact members.
- **Bad:** Lower values indicate clusters that are not well-separated.

### 4. **HDBSCAN-Specific Metrics**
When using HDBSCAN, additional metrics are provided:

#### **Cluster Membership Probabilities**
- Ranges from `0` to `1` indicating confidence of cluster assignment
- **Good:** Higher average probabilities indicate confident cluster assignments
- **Concerning:** Low probabilities suggest points near cluster boundaries or noise

#### **Outlier Scores** 
- Calculated as `1 - membership_probability`
- Ranges from `0` to `1` where higher values indicate more likely outliers
- **Good:** Low average outlier scores indicate cohesive clusters
- **Useful:** High outlier scores help identify anomalies or noise points

#### **Composite Quality Score** (Used in Grid Search)
HDBSCAN uses a weighted composite score for parameter optimization:
- 25% Normalized Calinski-Harabasz score
- 25% Inverted intra-cluster cosine similarity 
- 50% Cluster size ratio score (penalizes too many/few clusters)
- **Range:** 0 to 1, where higher scores indicate better clustering quality

---

## Clustering Performance Results

> [!NOTE]
> The [Reddit Comments Uwaterloo](https://huggingface.co/datasets/alvanlii/reddit-comments-uwaterloo) dataset from Hugging Face was used for the tests below.

### Agglomerative Clustering

| Hardware | Sample Size | Algorithm     | Clusters | Optimization Time (s) | Avg Time (s) | Avg Silhouette | Max Silhouette | Avg Davies-Bouldin | Max Davies-Bouldin | Avg Calinski-Harabasz | Max Calinski-Harabasz |
|----------|-------------|---------------|----------|-----------------------|--------------|----------------|----------------|--------------------|--------------------|-----------------------|-----------------------|
| GPU      | 5000        | Agglomerative | 2        | 7.574                 | 0.233        | 0.0434         | 0.0434         | 0.9183             | 0.9183             | 1.1843                | 1.1843                |
| CPU      | 5000        | Agglomerative | 2        | 85.4279               | 10.0897      | 0.0181         | 0.0181         | 6.1139             | 6.1139             | 113.896               | 113.896               |

### KMeans Clustering

| Hardware | Sample Size | Algorithm | Clusters | Optimization Time (s) | Avg Time (s) | Avg Silhouette | Max Silhouette | Avg Davies-Bouldin | Max Davies-Bouldin | Avg Calinski-Harabasz | Max Calinski-Harabasz |
|----------|-------------|-----------|----------|-----------------------|--------------|----------------|----------------|--------------------|--------------------|-----------------------|-----------------------|
| GPU      | 5000        | KMeans    | 2        | 8.2121                | 0.2369       | 0.0295         | 0.0295         | 5.1668             | 5.1668             | 171.257               | 171.257               |
| CPU      | 5000        | KMeans    | 2        | 7.7188                | 0.2356       | 0.0295         | 0.0295         | 5.1668             | 5.1668             | 171.257               | 171.257               |

---

## Key Insights

1. **Silhouette Scores:**
   - For both algorithms, the **Silhouette scores** are low, with the highest value being `0.0434` for Agglomerative clustering on the GPU. This indicates that the clusters are not very well-defined, and samples may be close to the boundary between clusters.
   - The KMeans algorithm had slightly lower Silhouette scores (max `0.0295`), which suggests that neither algorithm performed particularly well at forming distinct clusters in this dataset.

2. **Davies-Bouldin Index:**
   - The **Davies-Bouldin Index** is significantly better (lower) for Agglomerative clustering on the GPU (`0.9183`), meaning that the clusters are more distinct in comparison to KMeans clustering, which has a higher index (`5.1668`).
   - The CPU implementation of Agglomerative clustering shows poor cluster separation, as indicated by the much higher Davies-Bouldin value of `6.1139`.

3. **Calinski-Harabasz Index:**
   - The **Calinski-Harabasz Index** is notably higher for KMeans clustering (`171.257`), which suggests better cluster separation compared to Agglomerative clustering (max value `113.896`).
   - For Agglomerative clustering on the GPU, the Calinski-Harabasz score is particularly low (`1.1843`), indicating poor between-cluster separation.

4. **Hardware Performance:**
   - The **GPU significantly outperforms the CPU** in optimization time for Agglomerative clustering (7.574s vs. 85.4279s) and achieves the best Silhouette score.
   - The difference in time between the CPU and GPU for KMeans clustering is minimal, with both achieving nearly identical optimization times (`~7.7s` for CPU and `~8.2s` for GPU).

### Summary

- Agglomerative clustering performed better on the GPU in terms of Silhouette and Davies-Bouldin scores, but KMeans clustering achieved a higher Calinski-Harabasz score, indicating better cluster separation.
- The **GPU is generally faster and produces slightly better cluster quality**, especially with Agglomerative clustering.
- Both algorithms show low Silhouette scores, indicating that further tuning or different clustering methods may be required to achieve clearer separation of clusters.


## Implementation Details

### Core Framework
- **GPU Acceleration**: RAPIDS cuML for GPU-accelerated clustering algorithms
- **CPU Processing**: scikit-learn for CPU-based clustering implementations
- **Data Processing**: cuDF for GPU data manipulation, pandas for CPU operations
- **Embeddings**: OpenAI API integration with intelligent caching for efficiency
- **Metrics**: Comprehensive clustering quality evaluation using multiple metrics

### HDBSCAN Implementation

The HDBSCAN implementation is based on the production clustering system used in Lattice's text analysis pipeline, featuring:

#### **Architecture**
- **Dual Implementation**: Separate optimized versions for CPU (`HDBSCANCPUClusterer`) and GPU (`HDBSCANGPUClusterer`)
- **Unified Interface**: Common API through `HDBSCANClusteringAlgorithm` class
- **Grid Search Optimization**: Automated parameter tuning using composite quality scoring
- **Production-Grade**: Based on real-world clustering system handling large-scale text analysis

#### **Key Features**
- **Parameter Optimization**: Grid search over `min_cluster_size` and `min_samples` ranges
- **Quality Scoring**: Composite metric combining Calinski-Harabasz, cosine similarity, and cluster ratio
- **Cluster Merging**: Optional post-processing to merge similar clusters based on centroid similarity  
- **Early Stopping**: Intelligent grid search termination to reduce computation time
- **Feature Standardization**: Optional z-score normalization with sklearn/cuML scalers

#### **Performance Optimizations**
- **Vectorized Operations**: Efficient numpy/cupy operations for distance calculations
- **Memory Management**: Optimized GPU memory usage with cuDF operations
- **Cached Computations**: Embedding matrices computed once and reused
- **Parallel-Safe**: Single-threaded clustering for consistent benchmarking

#### **GPU-Specific Optimizations**
- **cuML Integration**: Native RAPIDS HDBSCAN implementation
- **GPU Memory**: Efficient cuDF DataFrame operations
- **Fallback Support**: Automatic CPU fallback for unsupported GPU operations
- **Memory Transfers**: Minimized CPU-GPU data movement

### Algorithm Comparison Framework

The clustering comparison system provides:
- **Multi-Algorithm Support**: Easy comparison between agglomerative, k-means, and HDBSCAN
- **Performance Metrics**: Comprehensive timing and quality measurements
- **Flexible CLI**: Support for single algorithm runs or batch comparisons
- **Result Visualization**: Tabulated results with detailed metrics breakdown

### Data Pipeline
1. **Dataset Loading**: Automated fetching from HuggingFace datasets
2. **Text Preprocessing**: Cleaning and normalization of text data  
3. **Embedding Generation**: Parallel OpenAI API calls with caching
4. **Clustering Execution**: Algorithm-specific clustering with optimization
5. **Results Analysis**: Comprehensive metrics calculation and reporting

By comparing CPU and GPU implementations across multiple algorithms, this project provides insights into the trade-offs between processing speed, clustering quality, and hardware requirements for different clustering scenarios.

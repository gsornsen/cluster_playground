## Project Overview

This repository is designed to test the efficacy and efficiency of different clustering algorithms using both CPU and GPU implementations. The main goals are:

1. Compare the performance of clustering algorithms between CPU and GPU usage.
2. Evaluate the quality of clustering results using various metrics.
3. Demonstrate the benefits of parallelization and GPU acceleration in data processing and machine learning tasks.

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

```
python src/perform_clustering.py [options]
```
### Options
- `--sample_size`: Number of samples to use for clustering (default: 500)
- `--use_gpu`: Use GPU for clustering (default: False)
- `--iterations`: Number of clustering iterations to run (default: 1)
- `--min_clusters`: Minimum number of clusters (default: 2)
- `--max_clusters`: Maximum number of clusters (default: 10)
- `--algorithm`: Clustering algorithm to use (choices: "agglomerative", "kmeans", default: "agglomerative")

### Example

```
python src/perform_clustering.py --sample_size 1000 --use_gpu --iterations 5 --min_clusters 2 --max_clusters 15 --algorithm kmeans
```

This command will run the K-means clustering algorithm on 1000 samples using GPU acceleration, performing 5 iterations for each cluster count from 2 to 15.

### Currently Supported Algorithms:

1. Agglomerative Clustering
2. K-Means Clustering

More algorithms will be added in the future.


## Example Clustering Results Summary

>
[!]These were run on an Intel i7-9700k CPU with an Nvidia RTX 4090 GPU


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

---

## Clustering Performance Results

>
[!] The [Reddit Comments Uwaterloo](https://huggingface.co/datasets/alvanlii/reddit-comments-uwaterloo) dataset from Hugging Face was used for the tests below.

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

- The project uses RAPIDS cuML for GPU-accelerated clustering and scikit-learn for CPU-based clustering.
- Embeddings are generated using OpenAI's API and cached for efficiency.
- Multiple clustering iterations are performed to ensure robust results.
- Various clustering quality metrics (silhouette score, Davies-Bouldin index, and Calinski-Harabasz index) are calculated to evaluate the clustering performance.

By comparing CPU and GPU implementations, this project aims to provide insights into the trade-offs between processing speed, clustering quality, and hardware requirements for different clustering scenarios.

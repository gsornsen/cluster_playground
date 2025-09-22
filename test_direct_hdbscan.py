#!/usr/bin/env python3
"""
Test HDBSCAN directly with your actual embeddings data, bypassing the grid search.
"""

import cudf
import asyncio
import numpy as np
from utils.embedding_utils import (
    calculate_text_hash,
    save_embeddings_to_cache,
    load_embeddings_from_cache,
    get_embeddings_in_parallel,
)
from utils.preprocessing_utils import fetch_dataset, preprocess_data
from sklearn.cluster import HDBSCAN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_direct_hdbscan(sample_size=1000):
    """Test HDBSCAN directly without grid search."""
    
    logger.info(f"Testing direct HDBSCAN with sample_size={sample_size}")
    
    # Load the same data your main script uses
    logger.info("Fetching dataset...")
    dataset = fetch_dataset()
    
    logger.info("Preprocessing...")
    texts = preprocess_data(dataset)
    
    text_hash = calculate_text_hash(texts[:sample_size])
    
    cached_embeddings = load_embeddings_from_cache(text_hash)
    if cached_embeddings:
        logger.info("Loaded embeddings from cache.")
        embeddings = cached_embeddings
    else:
        logger.info("Fetching OpenAI embeddings...")
        embeddings = await get_embeddings_in_parallel(texts[:sample_size])
        if embeddings:
            save_embeddings_to_cache(embeddings, text_hash)
        else:
            logger.error("Failed to fetch embeddings")
            return False
    
    logger.info(f"Embeddings shape: {np.array(embeddings).shape}")
    
    # Convert to numpy (like your pipeline does)
    X = np.array(embeddings, dtype=np.float32)
    logger.info(f"Numpy array shape: {X.shape}")
    
    # Test 1: Very conservative single HDBSCAN call
    logger.info("\n=== Test 1: Conservative Single Call ===")
    try:
        clusterer = HDBSCAN(
            min_cluster_size=10,
            min_samples=3,
            algorithm='brute',
            metric='euclidean',
            n_jobs=1,
            leaf_size=50
        )
        
        logger.info("Fitting direct HDBSCAN...")
        clusterer.fit(X)
        
        n_clusters = len(set(clusterer.labels_))
        n_noise = sum(1 for x in clusterer.labels_ if x == -1)
        logger.info(f"✓ Direct HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
        
    except Exception as e:
        logger.error(f"✗ Direct HDBSCAN failed: {e}")
        return False
    
    # Test 2: Parameters similar to your grid search
    logger.info("\n=== Test 2: Grid Search Parameters ===")
    test_params = [
        {'min_cluster_size': 3, 'min_samples': 1},
        {'min_cluster_size': 5, 'min_samples': 3},
        {'min_cluster_size': 10, 'min_samples': 6},
    ]
    
    for i, params in enumerate(test_params):
        try:
            logger.info(f"Testing params {i+1}: {params}")
            
            clusterer = HDBSCAN(
                algorithm='ball_tree',
                metric='euclidean', 
                n_jobs=1,
                **params
            )
            
            clusterer.fit(X)
            n_clusters = len(set(clusterer.labels_))
            logger.info(f"✓ Params {i+1}: {n_clusters} clusters")
            
        except Exception as e:
            logger.error(f"✗ Params {i+1} failed: {e}")
            return False
    
    # Test 3: With standardization (like your pipeline)
    logger.info("\n=== Test 3: With Standardization ===")
    try:
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Applied standardization")
        
        clusterer = HDBSCAN(
            min_cluster_size=10,
            min_samples=3,
            algorithm='ball_tree',
            metric='euclidean',
            n_jobs=1
        )
        
        clusterer.fit(X_scaled)
        n_clusters = len(set(clusterer.labels_))
        logger.info(f"✓ Standardized HDBSCAN: {n_clusters} clusters")
        
    except Exception as e:
        logger.error(f"✗ Standardized HDBSCAN failed: {e}")
        return False
    
    logger.info("\n🎉 All direct HDBSCAN tests passed!")
    logger.info("The issue is in the grid search implementation, not basic HDBSCAN.")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_direct_hdbscan(1000))
    
    if success:
        print("\n✅ Direct HDBSCAN works! The problem is in the grid search.")
        print("The segfault occurs during parameter optimization, not basic clustering.")
    else:
        print("\n❌ Direct HDBSCAN also fails. Issue might be in data loading/preprocessing.")
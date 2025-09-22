#!/usr/bin/env python3
"""
Debug script to isolate HDBSCAN segfault causes.
Run this to test different HDBSCAN configurations and identify the problematic component.
"""

import numpy as np
import logging
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_hdbscan():
    """Test basic HDBSCAN functionality with known-good parameters."""
    logger.info("=== Testing Basic HDBSCAN ===")
    
    # Create simple synthetic data
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    logger.info(f"Created synthetic data: {X.shape}")
    
    try:
        clusterer = HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            algorithm='ball_tree',
            metric='euclidean',
            n_jobs=1
        )
        clusterer.fit(X)
        logger.info(f"✓ Basic HDBSCAN works: {len(set(clusterer.labels_))} clusters")
        return True
    except Exception as e:
        logger.error(f"✗ Basic HDBSCAN failed: {e}")
        return False

def test_memory_patterns():
    """Test different memory allocation patterns."""
    logger.info("=== Testing Memory Patterns ===")
    
    sizes_to_test = [100, 500, 1000, 2000, 3000]
    
    for size in sizes_to_test:
        logger.info(f"Testing size: {size}")
        
        # Monitor memory before
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Create data similar to your embeddings (1536 features)
            X = np.random.randn(size, 100)  # Start with smaller feature space
            
            clusterer = HDBSCAN(
                min_cluster_size=max(3, size // 50),
                min_samples=3,
                algorithm='ball_tree',
                metric='euclidean',
                n_jobs=1
            )
            clusterer.fit(X)
            
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            logger.info(f"✓ Size {size}: {len(set(clusterer.labels_))} clusters, {mem_used:.1f}MB used")
            
        except Exception as e:
            logger.error(f"✗ Size {size} failed: {e}")
            break
    
def test_algorithm_backends():
    """Test different HDBSCAN algorithm backends."""
    logger.info("=== Testing Algorithm Backends ===")
    
    X, _ = make_blobs(n_samples=500, centers=5, random_state=42)
    
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    
    for algo in algorithms:
        try:
            logger.info(f"Testing algorithm: {algo}")
            clusterer = HDBSCAN(
                min_cluster_size=10,
                min_samples=3,
                algorithm=algo,
                metric='euclidean',
                n_jobs=1
            )
            clusterer.fit(X)
            logger.info(f"✓ Algorithm {algo}: {len(set(clusterer.labels_))} clusters")
            
        except Exception as e:
            logger.error(f"✗ Algorithm {algo} failed: {e}")

def test_parameter_combinations():
    """Test problematic parameter combinations."""
    logger.info("=== Testing Parameter Combinations ===")
    
    X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)
    
    # Test combinations that might cause issues
    test_params = [
        {'min_cluster_size': 3, 'min_samples': 1},
        {'min_cluster_size': 5, 'min_samples': 3},
        {'min_cluster_size': 10, 'min_samples': 5},
        {'min_cluster_size': 20, 'min_samples': 10},
        {'min_cluster_size': 50, 'min_samples': 25},  # This might be problematic
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
            logger.info(f"✓ Params {i+1}: {len(set(clusterer.labels_))} clusters")
            
        except Exception as e:
            logger.error(f"✗ Params {i+1} failed: {e}")

def test_large_feature_space():
    """Test with large feature spaces like your embeddings."""
    logger.info("=== Testing Large Feature Space ===")
    
    feature_sizes = [100, 500, 1000, 1536]  # 1536 is your embedding size
    n_samples = 500  # Keep samples small, vary features
    
    for n_features in feature_sizes:
        try:
            logger.info(f"Testing {n_samples} samples x {n_features} features")
            
            # Create random data similar to embeddings
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            
            clusterer = HDBSCAN(
                min_cluster_size=10,
                min_samples=3,
                algorithm='ball_tree',
                metric='euclidean',
                n_jobs=1
            )
            
            clusterer.fit(X)
            logger.info(f"✓ Features {n_features}: {len(set(clusterer.labels_))} clusters")
            
        except Exception as e:
            logger.error(f"✗ Features {n_features} failed: {e}")
            break

if __name__ == "__main__":
    logger.info("Starting HDBSCAN segfault debugging...")
    logger.info(f"System: {os.uname()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB total")
    
    # Run tests in order of complexity
    tests = [
        test_basic_hdbscan,
        test_algorithm_backends, 
        test_parameter_combinations,
        test_memory_patterns,
        test_large_feature_space,
    ]
    
    for test_func in tests:
        try:
            test_func()
            print()  # Add spacing
        except Exception as e:
            logger.error(f"Test {test_func.__name__} crashed: {e}")
            break
    
    logger.info("Debugging complete. Check logs above for failure points.")
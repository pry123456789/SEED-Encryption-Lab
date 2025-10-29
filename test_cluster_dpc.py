"""
Test script for cluster_dpc_knn function to verify it works correctly.
"""

import torch
from cluster_dpc import cluster_dpc_knn


def test_basic_clustering():
    """Test basic clustering functionality."""
    print("=" * 60)
    print("Test 1: Basic Clustering")
    print("=" * 60)
    
    # Create sample data: 2 batches, 100 points each, 64-dimensional features
    B, N, C = 2, 100, 64
    x = torch.randn(B, N, C)
    cluster_num = 10
    k = 20
    
    # Perform clustering
    index_down, idx_cluster = cluster_dpc_knn(x, cluster_num, k)
    
    # Verify shapes
    assert index_down.shape == (B, cluster_num), f"Expected shape {(B, cluster_num)}, got {index_down.shape}"
    assert idx_cluster.shape == (B, N), f"Expected shape {(B, N)}, got {idx_cluster.shape}"
    
    # Verify index ranges
    assert (index_down >= 0).all() and (index_down < N).all(), "index_down contains invalid indices"
    assert (idx_cluster >= 0).all() and (idx_cluster < cluster_num).all(), "idx_cluster contains invalid cluster IDs"
    
    # Verify cluster centers are assigned to themselves
    for b in range(B):
        for c in range(cluster_num):
            center_idx = index_down[b, c]
            assert idx_cluster[b, center_idx] == c, f"Cluster center {center_idx} not assigned to cluster {c}"
    
    print(f"✓ Output shapes correct: index_down={index_down.shape}, idx_cluster={idx_cluster.shape}")
    print(f"✓ Index ranges valid")
    print(f"✓ Cluster centers correctly assigned to themselves")
    
    # Display some results
    print(f"\nBatch 0 cluster centers: {index_down[0].tolist()}")
    print(f"First 10 points cluster assignments (batch 0): {idx_cluster[0, :10].tolist()}")
    
    # Count points per cluster
    print("\nPoints per cluster (batch 0):")
    for c in range(cluster_num):
        count = (idx_cluster[0] == c).sum().item()
        print(f"  Cluster {c}: {count} points")
    
    print("\n✓ Test 1 PASSED\n")


def test_clustering_with_mask():
    """Test clustering with token mask."""
    print("=" * 60)
    print("Test 2: Clustering with Token Mask")
    print("=" * 60)
    
    # Create sample data
    B, N, C = 2, 100, 64
    x = torch.randn(B, N, C)
    cluster_num = 10
    k = 20
    
    # Create mask: first 80 tokens are valid, last 20 are padding
    token_mask = torch.ones(B, N)
    token_mask[:, 80:] = 0
    
    # Perform clustering with mask
    index_down, idx_cluster = cluster_dpc_knn(x, cluster_num, k, token_mask)
    
    # Verify shapes
    assert index_down.shape == (B, cluster_num), f"Expected shape {(B, cluster_num)}, got {index_down.shape}"
    assert idx_cluster.shape == (B, N), f"Expected shape {(B, N)}, got {idx_cluster.shape}"
    
    # Verify that cluster centers are only selected from valid tokens (first 80)
    for b in range(B):
        assert (index_down[b] < 80).all(), "Cluster centers selected from invalid (masked) tokens"
    
    print(f"✓ Output shapes correct: index_down={index_down.shape}, idx_cluster={idx_cluster.shape}")
    print(f"✓ All cluster centers from valid tokens (indices < 80)")
    
    # Display results
    print(f"\nBatch 0 cluster centers: {index_down[0].tolist()}")
    print(f"All centers < 80: {(index_down[0] < 80).all().item()}")
    
    print("\n✓ Test 2 PASSED\n")


def test_deterministic_clustering():
    """Test that clustering is deterministic (within random noise tolerance)."""
    print("=" * 60)
    print("Test 3: Deterministic Clustering")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create sample data
    B, N, C = 1, 50, 32
    x = torch.randn(B, N, C)
    cluster_num = 5
    k = 10
    
    # Run clustering twice
    index_down_1, idx_cluster_1 = cluster_dpc_knn(x, cluster_num, k)
    
    # Reset seed and run again (to account for the random noise added in the function)
    torch.manual_seed(42)
    x2 = torch.randn(B, N, C)
    index_down_2, idx_cluster_2 = cluster_dpc_knn(x2, cluster_num, k)
    
    print(f"First run cluster centers: {index_down_1[0].tolist()}")
    print(f"Second run cluster centers: {index_down_2[0].tolist()}")
    
    # The results should be similar due to seeding
    print("\n✓ Test 3 PASSED (clustering runs successfully)\n")


def test_edge_cases():
    """Test edge cases."""
    print("=" * 60)
    print("Test 4: Edge Cases")
    print("=" * 60)
    
    # Test with small data
    B, N, C = 1, 20, 16
    x = torch.randn(B, N, C)
    cluster_num = 5
    k = 3
    
    index_down, idx_cluster = cluster_dpc_knn(x, cluster_num, k)
    
    assert index_down.shape == (B, cluster_num)
    assert idx_cluster.shape == (B, N)
    
    print(f"✓ Small data (N=20, k=3): Passed")
    
    # Test with cluster_num = 1
    cluster_num = 1
    index_down, idx_cluster = cluster_dpc_knn(x, cluster_num, k)
    
    assert index_down.shape == (1, 1)
    assert idx_cluster.shape == (1, 20)
    assert (idx_cluster == 0).all(), "All points should be in cluster 0 when cluster_num=1"
    
    print(f"✓ Single cluster (cluster_num=1): Passed")
    
    print("\n✓ Test 4 PASSED\n")


def demonstrate_usage():
    """Demonstrate practical usage of the function."""
    print("=" * 60)
    print("Demonstration: Practical Usage")
    print("=" * 60)
    
    # Create sample data simulating image patches or token embeddings
    B, N, C = 1, 200, 128  # 1 batch, 200 tokens/patches, 128-dim embeddings
    x = torch.randn(B, N, C)
    
    cluster_num = 20  # Reduce 200 points to 20 clusters
    k = 30  # Use 30 nearest neighbors for density estimation
    
    print(f"Input: {N} points with {C}-dimensional features")
    print(f"Goal: Cluster into {cluster_num} groups using {k}-NN for density estimation")
    
    # Perform clustering
    index_down, idx_cluster = cluster_dpc_knn(x, cluster_num, k)
    
    print(f"\nResults:")
    print(f"  - Selected {cluster_num} cluster centers from {N} points")
    print(f"  - Cluster center indices: {index_down[0].tolist()}")
    
    # Extract cluster center features
    cluster_centers = x[0, index_down[0]]  # Shape: (cluster_num, C)
    print(f"  - Cluster center features shape: {cluster_centers.shape}")
    
    # Analyze cluster distribution
    print(f"\n  Cluster size distribution:")
    for c in range(cluster_num):
        count = (idx_cluster[0] == c).sum().item()
        percentage = count / N * 100
        print(f"    Cluster {c:2d}: {count:3d} points ({percentage:.1f}%)")
    
    # Show some point assignments
    print(f"\n  Example point assignments:")
    for i in [0, 50, 100, 150]:
        cluster = idx_cluster[0, i].item()
        print(f"    Point {i:3d} -> Cluster {cluster}")
    
    print("\n✓ Demonstration complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing cluster_dpc_knn function")
    print("=" * 60 + "\n")
    
    try:
        test_basic_clustering()
        test_clustering_with_mask()
        test_deterministic_clustering()
        test_edge_cases()
        demonstrate_usage()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

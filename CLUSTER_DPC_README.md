# Cluster DPC KNN Function Documentation

## Overview

This documentation explains the `cluster_dpc_knn` function, which implements Density Peak Clustering (DPC) with K-Nearest Neighbors for density estimation.

## Function Signature

```python
def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    """
    Perform Density Peak Clustering using K-Nearest Neighbors.
    
    Args:
        x (torch.Tensor): Input features of shape (B, N, C)
        cluster_num (int): Number of clusters to create
        k (int): Number of nearest neighbors for density estimation
        token_mask (torch.Tensor, optional): Binary mask of shape (B, N)
    
    Returns:
        tuple: (index_down, idx_cluster)
    """
```

## Return Values Explained

### What Does This Function Return?

The function returns a tuple of two tensors: `(index_down, idx_cluster)`

#### 1. `index_down` - Cluster Center Indices

**Shape**: `(B, cluster_num)`

**Meaning**: The indices of the points selected as cluster centers

**Details**:
- Each element is an integer index in the range [0, N-1]
- These indices point to the original data points that have been selected as cluster centers
- Selected based on highest "score" (distance × density product)
- Points with high scores have:
  - **High density**: Many similar points nearby
  - **Large distance**: Far from other high-density points

**Example**:
```python
# If index_down[0, 0] = 5
# This means the 6th point (index 5) in batch 0 is the first cluster center

# If index_down = [[5, 12, 23, 45, 67],
#                  [3, 15, 28, 39, 56]]
# Batch 0 cluster centers: points at indices 5, 12, 23, 45, 67
# Batch 1 cluster centers: points at indices 3, 15, 28, 39, 56
```

**Use Cases**:
- Extract the most representative points from your dataset
- Downsample data while preserving structure
- Select key/important locations in the data
- Create prototypes for each cluster

#### 2. `idx_cluster` - Cluster Assignments

**Shape**: `(B, N)`

**Meaning**: The cluster label for each data point

**Details**:
- Each element is an integer in the range [0, cluster_num-1]
- `idx_cluster[b, i] = j` means point `i` in batch `b` belongs to cluster `j`
- Cluster centers are assigned to their corresponding cluster indices
- Other points are assigned to the nearest cluster center based on distance

**Example**:
```python
# If idx_cluster[0, 10] = 2
# This means point 10 in batch 0 belongs to cluster 2

# If idx_cluster[0, :] = [0, 0, 1, 1, 1, 0, 2, 2, 1, 0, ...]
# In batch 0:
# - Points 0, 1, 5, 9 belong to cluster 0
# - Points 2, 3, 4, 8 belong to cluster 1
# - Points 6, 7 belong to cluster 2
```

**Use Cases**:
- Group similar data points together
- Aggregate features within each cluster
- Apply attention mechanisms within clusters
- Hierarchical processing (cluster-level then point-level)

## Complete Usage Example

```python
import torch
from cluster_dpc import cluster_dpc_knn

# Create sample data: 2 batches, 100 points each, 64-dimensional features
x = torch.randn(2, 100, 64)

# Set clustering parameters
cluster_num = 10  # Create 10 clusters
k = 20  # Use 20 nearest neighbors for density estimation

# Perform clustering
index_down, idx_cluster = cluster_dpc_knn(x, cluster_num, k)

print(f"Cluster center indices shape: {index_down.shape}")  # (2, 10)
print(f"Cluster assignments shape: {idx_cluster.shape}")      # (2, 100)

# View cluster centers for batch 0
print(f"Batch 0 cluster centers: {index_down[0]}")

# View which cluster point 0 belongs to
print(f"Point 0 in batch 0 belongs to cluster: {idx_cluster[0, 0]}")

# Count points per cluster
for cluster_id in range(cluster_num):
    count = (idx_cluster[0] == cluster_id).sum().item()
    print(f"Cluster {cluster_id} contains {count} points")
```

## Example with Token Mask (Handling Padded Sequences)

```python
# Create data (simulating variable-length sequences)
x = torch.randn(2, 100, 64)

# Create mask: first 80 tokens are valid, last 20 are padding
token_mask = torch.ones(2, 100)
token_mask[:, 80:] = 0  # Last 20 tokens are invalid

# Perform clustering (only considering first 80 valid tokens)
index_down, idx_cluster = cluster_dpc_knn(x, cluster_num=10, k=20, token_mask=token_mask)

# Cluster centers will only be selected from the first 80 valid tokens
# The last 20 padding tokens are effectively excluded from clustering
```

## How the Algorithm Works

1. **Compute Distance Matrix**: Calculate pairwise distances between all points
2. **Estimate Density**: Use K-nearest neighbors to estimate local density at each point
3. **Compute Distances**: For each point, find distance to its nearest higher-density neighbor
4. **Calculate Scores**: score = distance × density
5. **Select Centers**: Choose the top `cluster_num` points with highest scores as cluster centers
6. **Assign Points**: Assign all points to their nearest cluster center

## Practical Applications

### Using `index_down` (Cluster Centers):

```python
# Extract cluster center features
B = x.shape[0]
cluster_centers = x[torch.arange(B)[:, None], index_down]  # Shape: (B, cluster_num, C)

# Use cluster centers for downsampling
downsampled_features = cluster_centers
```

### Using `idx_cluster` (Cluster Assignments):

```python
# Compute average features per cluster
cluster_features = []
for i in range(cluster_num):
    mask = (idx_cluster == i)  # Shape: (B, N)
    cluster_feat = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    cluster_features.append(cluster_feat)
```

## Parameter Recommendations

- **cluster_num**: Typically 10%-30% of the original number of points, depending on your task
- **k**: Recommended range is 15-30
  - Too small: Unstable density estimation
  - Too large: Loss of local structure information
- **token_mask**: Essential when dealing with variable-length sequences to avoid padding tokens affecting clustering

## Summary

**What the return values tell you:**
1. **`index_down`**: WHICH points were selected as cluster centers
2. **`idx_cluster`**: WHICH cluster each point belongs to

Together, these two return values provide a complete clustering solution that can be used for:
- Feature downsampling
- Data compression
- Hierarchical processing
- Attention mechanisms
- Prototype selection
- And many other machine learning tasks

## Testing

Run the included test script to verify the function works correctly:

```bash
python test_cluster_dpc.py
```

## Files in This Documentation

- `cluster_dpc.py`: Main implementation with detailed docstrings
- `test_cluster_dpc.py`: Comprehensive test suite
- `CLUSTER_DPC_README.md`: This file (English documentation)
- `CLUSTER_DPC_DOCUMENTATION_CN.md`: Chinese documentation (中文文档)

## References

Based on the Density Peak Clustering algorithm:
- Rodriguez, A., & Laio, A. (2014). Clustering by fast search and find of density peaks. Science, 344(6191), 1492-1496.

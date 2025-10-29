"""
Density Peak Clustering (DPC) with K-Nearest Neighbors

This module implements a clustering algorithm based on the Density Peak Clustering (DPC)
approach using K-Nearest Neighbors (KNN) for density estimation.

The algorithm clusters data points by:
1. Computing pairwise distances between all points
2. Estimating local density using K-nearest neighbors
3. Computing distances to higher-density neighbors
4. Selecting cluster centers with high density and large distances
5. Assigning remaining points to the nearest cluster center
"""

import torch


def index_points(points, idx):
    """
    Helper function to select points based on indices.
    
    Args:
        points: Input tensor of shape (B, N, C) where B is batch size, 
                N is number of points, C is feature dimension
        idx: Index tensor of shape (B, M) containing indices to select
        
    Returns:
        Tensor of shape (B, M, C) containing the selected points
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    """
    Perform Density Peak Clustering using K-Nearest Neighbors.
    
    This function clusters input data points using a density-based approach. It first
    estimates the local density of each point using its K nearest neighbors, then
    identifies cluster centers as points with high density and large distances to
    higher-density neighbors. Finally, it assigns all points to their nearest cluster center.
    
    Args:
        x (torch.Tensor): Input feature tensor of shape (B, N, C) where:
            - B: batch size
            - N: number of data points/tokens
            - C: feature dimension for each point
            
        cluster_num (int): Number of clusters to create (number of cluster centers to select)
        
        k (int): Number of nearest neighbors to use for density estimation
        
        token_mask (torch.Tensor, optional): Binary mask of shape (B, N) indicating valid tokens.
            - Values > 0 indicate valid tokens to consider for clustering
            - Invalid tokens (mask value = 0) are assigned maximum distance
            - If None, all tokens are considered valid
            - Default: None
    
    Returns:
        tuple: A tuple containing two tensors:
        
        **index_down (torch.Tensor)**: Indices of selected cluster centers
            - Shape: (B, cluster_num)
            - Each element is an integer index in range [0, N-1]
            - These are the indices of the original points selected as cluster centers
            - Selected based on highest "score" (distance * density product)
            - Example: If index_down[0, 0] = 5, then the 6th point in batch 0 is the first cluster center
            
        **idx_cluster (torch.Tensor)**: Cluster assignment for each point
            - Shape: (B, N)
            - Each element is an integer in range [0, cluster_num-1]
            - idx_cluster[b, i] = j means point i in batch b belongs to cluster j
            - Cluster centers are assigned to their own cluster index
            - Other points are assigned to the nearest cluster center based on distance
            - Example: If idx_cluster[0, 10] = 2, then point 10 in batch 0 belongs to cluster 2
    
    Algorithm Steps:
        1. Compute pairwise distance matrix between all points (normalized by sqrt(C))
        2. Apply token mask if provided (invalid tokens get maximum distance)
        3. Find K-nearest neighbors for each point
        4. Estimate local density using exponential of negative mean squared distance to K-NN
        5. Add small random noise to density to break ties
        6. For each point, find distance to nearest point with higher density
        7. Compute clustering score as: score = distance_to_higher_density * density
        8. Select top cluster_num points with highest scores as cluster centers
        9. Assign each point to its nearest cluster center
        10. Ensure cluster centers are assigned to themselves
    
    Example:
        >>> # Create sample data: 2 batches, 100 points each, 64-dimensional features
        >>> x = torch.randn(2, 100, 64)
        >>> cluster_num = 10  # Create 10 clusters
        >>> k = 20  # Use 20 nearest neighbors for density estimation
        >>> 
        >>> # Perform clustering
        >>> index_down, idx_cluster = cluster_dpc_knn(x, cluster_num, k)
        >>> 
        >>> # index_down shape: (2, 10) - indices of 10 cluster centers for each batch
        >>> # idx_cluster shape: (2, 100) - cluster assignment for each of 100 points
        >>> 
        >>> print(f"Cluster centers for batch 0: {index_down[0]}")
        >>> print(f"Point 0 belongs to cluster: {idx_cluster[0, 0]}")
        >>> 
        >>> # With token mask (e.g., for padded sequences)
        >>> token_mask = torch.ones(2, 100)
        >>> token_mask[:, 80:] = 0  # Last 20 tokens are invalid/padding
        >>> index_down, idx_cluster = cluster_dpc_knn(x, cluster_num, k, token_mask)
        >>> # Now only first 80 tokens will be considered for clustering
    
    Notes:
        - Uses cosine-style normalized distance (divided by sqrt(C))
        - Higher density indicates more points nearby (denser region)
        - Cluster centers have high density AND are far from other high-density points
        - The algorithm runs with torch.no_grad() for efficiency (no gradient computation)
        - Small random noise (1e-6) is added to density to ensure unique rankings
        - Invalid tokens (when token_mask is used) are effectively excluded from clustering
    
    References:
        Based on the Density Peak Clustering algorithm:
        Rodriguez, A., & Laio, A. (2014). Clustering by fast search and find of density peaks.
        Science, 344(6191), 1492-1496.
    """
    with torch.no_grad():
        B, N, C = x.shape

        # Compute pairwise distance matrix (normalized by sqrt of feature dimension)
        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        # Apply token mask if provided
        if token_mask is not None:
            token_mask = token_mask > 0
            # Valid tokens keep original distance, invalid tokens get max distance + 1
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # Find K nearest neighbors for each point
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        # Estimate local density: exponential of negative mean squared distance to K-NN
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # Add small random noise to break ties in density
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        # Apply token mask to density (invalid tokens have zero density)
        if token_mask is not None:
            density = density * token_mask

        # Create mask where mask[i,j] = True if density[i] > density[j]
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        
        # Find maximum distance for each batch (used as penalty for same/lower density points)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        
        # For each point, find distance to nearest point with higher density
        # Points with no higher density neighbors get dist_max
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # Compute clustering score: points with high density AND far from higher-density points
        # are good cluster centers
        score = dist * density
        
        # Select top cluster_num points with highest scores as cluster centers
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # Extract distance matrix rows corresponding to cluster centers
        dist_matrix = index_points(dist_matrix, index_down)

        # Assign each point to its nearest cluster center
        idx_cluster = dist_matrix.argmin(dim=1)

        # Ensure cluster centers are assigned to their own cluster index
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster

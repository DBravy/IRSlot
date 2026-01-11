"""
Graph construction utilities for GNN-based encoder.

Provides functions to convert ARC grids to graph representations:
- Edge index generation (4-connectivity and 8-connectivity)
- Node feature extraction (one-hot colors + positions)
- Caching system for edge indices
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def create_grid_edge_index_4conn(H: int, W: int) -> torch.Tensor:
    """
    Create edge_index for 4-connected grid graph.

    Connects each pixel to up, down, left, right neighbors.

    Args:
        H: Grid height
        W: Grid width

    Returns:
        edge_index: [2, num_edges] - source and target node indices
    """
    edges = []

    for i in range(H):
        for j in range(W):
            node_id = i * W + j

            # Up
            if i > 0:
                neighbor_id = (i - 1) * W + j
                edges.append([node_id, neighbor_id])

            # Down
            if i < H - 1:
                neighbor_id = (i + 1) * W + j
                edges.append([node_id, neighbor_id])

            # Left
            if j > 0:
                neighbor_id = i * W + (j - 1)
                edges.append([node_id, neighbor_id])

            # Right
            if j < W - 1:
                neighbor_id = i * W + (j + 1)
                edges.append([node_id, neighbor_id])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def create_grid_edge_index_8conn(H: int, W: int) -> torch.Tensor:
    """
    Create edge_index for 8-connected grid graph.

    Connects each pixel to all 8 neighbors including diagonals.

    Args:
        H: Grid height
        W: Grid width

    Returns:
        edge_index: [2, num_edges] - source and target node indices
    """
    edges = []

    for i in range(H):
        for j in range(W):
            node_id = i * W + j

            # All 8 directions
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue

                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbor_id = ni * W + nj
                        edges.append([node_id, neighbor_id])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def grid_to_node_features(
    grid: torch.Tensor,
    use_position: bool = True
) -> torch.Tensor:
    """
    Convert grid to node feature matrix.

    Args:
        grid: [H, W] tensor with color values 0-9
        use_position: Whether to include position features

    Returns:
        node_features: [H*W, feat_dim] where feat_dim = 10 or 12
                      - First 10 dims: one-hot color encoding
                      - Last 2 dims (optional): normalized (x, y) positions [0, 1]
    """
    H, W = grid.shape
    N = H * W

    # One-hot encode colors
    colors = grid.flatten()  # [H*W]
    node_features = F.one_hot(colors, num_classes=10).float()  # [H*W, 10]

    if use_position:
        # Add normalized position features
        y_coords = torch.arange(H, dtype=torch.float32, device=grid.device) / max(H - 1, 1)
        x_coords = torch.arange(W, dtype=torch.float32, device=grid.device) / max(W - 1, 1)

        # Create meshgrid and flatten
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # [H*W, 2]

        # Concatenate with colors
        node_features = torch.cat([node_features, positions], dim=1)  # [H*W, 12]

    return node_features


class EdgeIndexCache:
    """
    Cache for edge_index tensors to avoid recomputation.

    Caches edge indices for different (H, W, connectivity) combinations
    and moves them to appropriate device on demand.
    """
    def __init__(self):
        self.cache = {}

    def get(self, H: int, W: int, connectivity: int, device: torch.device) -> torch.Tensor:
        """
        Get edge_index for specified grid size and connectivity.

        Args:
            H: Grid height
            W: Grid width
            connectivity: 4 or 8
            device: Target device for edge_index

        Returns:
            edge_index: [2, num_edges] on specified device
        """
        key = (H, W, connectivity, str(device))

        if key not in self.cache:
            if connectivity == 4:
                edge_index = create_grid_edge_index_4conn(H, W)
            elif connectivity == 8:
                edge_index = create_grid_edge_index_8conn(H, W)
            else:
                raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

            edge_index = edge_index.to(device)
            self.cache[key] = edge_index

        return self.cache[key]

    def clear(self):
        """Clear cache to free memory."""
        self.cache.clear()

    def __len__(self):
        """Return number of cached edge indices."""
        return len(self.cache)


# Global cache instance for reuse across encoder instances
_EDGE_INDEX_CACHE = EdgeIndexCache()


def get_cached_edge_index(H: int, W: int, connectivity: int, device: torch.device) -> torch.Tensor:
    """
    Get edge_index from global cache.

    Convenience function to access the global edge index cache.

    Args:
        H: Grid height
        W: Grid width
        connectivity: 4 or 8
        device: Target device

    Returns:
        edge_index: [2, num_edges] on specified device
    """
    return _EDGE_INDEX_CACHE.get(H, W, connectivity, device)


def clear_edge_index_cache():
    """Clear the global edge index cache."""
    _EDGE_INDEX_CACHE.clear()


# Testing utilities
if __name__ == "__main__":
    print("Testing graph_utils.py")
    print("=" * 60)

    # Test 1: 4-connectivity edge index
    print("\nTest 1: 4-connectivity for 3x3 grid")
    edge_index_4 = create_grid_edge_index_4conn(3, 3)
    print(f"Shape: {edge_index_4.shape}")
    print(f"Expected edges: ~2*3*3 = ~18 (accounting for boundaries)")
    print(f"Actual edges: {edge_index_4.shape[1]}")

    # Test 2: 8-connectivity edge index
    print("\nTest 2: 8-connectivity for 3x3 grid")
    edge_index_8 = create_grid_edge_index_8conn(3, 3)
    print(f"Shape: {edge_index_8.shape}")
    print(f"Expected edges: ~4*3*3 = ~36 (accounting for boundaries)")
    print(f"Actual edges: {edge_index_8.shape[1]}")

    # Test 3: Node features without position
    print("\nTest 3: Node features without position")
    grid = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.long)
    features = grid_to_node_features(grid, use_position=False)
    print(f"Grid shape: {grid.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Expected: [9, 10], Got: {list(features.shape)}")

    # Test 4: Node features with position
    print("\nTest 4: Node features with position")
    features_pos = grid_to_node_features(grid, use_position=True)
    print(f"Features shape: {features_pos.shape}")
    print(f"Expected: [9, 12], Got: {list(features_pos.shape)}")
    print(f"Position range: [{features_pos[:, 10:].min():.3f}, {features_pos[:, 10:].max():.3f}]")
    print(f"Expected position range: [0.0, 1.0]")

    # Test 5: Edge index cache
    print("\nTest 5: Edge index caching")
    cache = EdgeIndexCache()
    device = torch.device('cpu')

    # First call (should create)
    edge1 = cache.get(5, 5, 4, device)
    print(f"Cache size after first call: {len(cache)}")

    # Second call (should reuse)
    edge2 = cache.get(5, 5, 4, device)
    print(f"Cache size after second call: {len(cache)}")
    print(f"Same object: {edge1 is edge2}")

    # Different size (should create new)
    edge3 = cache.get(10, 10, 4, device)
    print(f"Cache size after different size: {len(cache)}")

    # Different connectivity (should create new)
    edge4 = cache.get(5, 5, 8, device)
    print(f"Cache size after different connectivity: {len(cache)}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")

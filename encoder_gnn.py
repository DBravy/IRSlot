"""
GNN-based encoder for ARC grids using Graph Convolutional Networks.

Alternative to CNN-based encoder that models grids as graphs where each pixel
is a node connected to its spatial neighbors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: PyTorch Geometric not installed. GNN encoder unavailable.")

from utils.graph_utils import grid_to_node_features, get_cached_edge_index


class ARCGridGNNEncoder(nn.Module):
    """
    GNN-based encoder for ARC grids using Graph Convolutional Networks.

    Takes a grid of shape [B, H, W] with integer color values (0-9)
    and produces feature maps of shape [B, H*W, feature_dim].

    Architecture:
    1. Convert grid to graph: pixels as nodes, spatial neighbors as edges
    2. Node features: one-hot color (10-dim) + optional position (2-dim)
    3. Graph convolutions: 4 GCN layers with residual connections
    4. Output: per-node features matching CNN encoder interface

    Args:
        num_colors: Number of colors in ARC grids (0-9 = 10 colors)
        feature_dim: Output feature dimension (default: 64)
        hidden_dim: Hidden dimension for GCN layers (default: 128)
        num_layers: Number of GCN layers (default: 4)
        edge_connectivity: Edge connectivity - 4 or 8 (default: 4)
        use_position_features: Include normalized position features (default: True)
        dropout: Dropout rate (default: 0.0)
    """
    def __init__(
        self,
        num_colors=10,
        feature_dim=64,
        hidden_dim=128,
        num_layers=4,
        edge_connectivity=4,
        use_position_features=True,
        dropout=0.0
    ):
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "PyTorch Geometric is required for GNN encoder. "
                "Install with: pip install torch-geometric torch-scatter torch-sparse"
            )

        self.num_colors = num_colors
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_connectivity = edge_connectivity
        self.use_position_features = use_position_features
        self.dropout = dropout

        # Input feature dimension: one-hot colors (10) + optional position (2)
        input_dim = num_colors + (2 if use_position_features else 0)

        # Input projection: node features -> hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Output projection: hidden_dim -> feature_dim
        self.output_projection = nn.Linear(hidden_dim, feature_dim)

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, grids):
        """
        Forward pass.

        Args:
            grids: [B, H, W] - Integer tensor with values 0-9

        Returns:
            features: [B, H*W, feature_dim] - Feature maps ready for slot attention
        """
        B, H, W = grids.shape
        N = H * W
        device = grids.device

        # Get edge_index for this grid size (cached)
        edge_index = get_cached_edge_index(H, W, self.edge_connectivity, device)

        # Process each sample in batch
        batch_features = []

        for b in range(B):
            # Convert grid to node features [N, input_dim]
            node_features = grid_to_node_features(
                grids[b],
                use_position=self.use_position_features
            )

            # Input projection: [N, input_dim] -> [N, hidden_dim]
            x = self.input_projection(node_features)
            x = self.relu(x)

            # Apply GCN layers with residual connections
            for i in range(self.num_layers):
                identity = x

                # GCN layer
                x = self.gcn_layers[i](x, edge_index)

                # Layer normalization
                x = self.layer_norms[i](x)

                # ReLU activation
                x = self.relu(x)

                # Residual connection
                x = x + identity

                # Dropout (if enabled)
                if self.dropout_layer is not None:
                    x = self.dropout_layer(x)

            # Output projection: [N, hidden_dim] -> [N, feature_dim]
            x = self.output_projection(x)

            batch_features.append(x)

        # Stack batch: [B, N, feature_dim]
        features = torch.stack(batch_features, dim=0)

        return features

    def get_num_parameters(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing and comparison
if __name__ == "__main__":
    print("Testing ARCGridGNNEncoder")
    print("=" * 60)

    if not HAS_TORCH_GEOMETRIC:
        print("ERROR: PyTorch Geometric not installed. Cannot test GNN encoder.")
        print("Install with: pip install torch-geometric torch-scatter torch-sparse")
        exit(1)

    # Create encoder
    print("\nCreating GNN encoder...")
    encoder = ARCGridGNNEncoder(
        num_colors=10,
        feature_dim=64,
        hidden_dim=128,
        num_layers=4,
        edge_connectivity=4,
        use_position_features=True,
        dropout=0.0
    )
    print(f"Number of parameters: {encoder.get_num_parameters():,}")

    # Test with single grid
    print("\nTest 1: Single grid")
    grid = torch.randint(0, 10, (1, 10, 10))
    print(f"Input shape: {grid.shape}")
    features = encoder(grid)
    print(f"Output shape: {features.shape}")
    print(f"Expected: [1, 100, 64], Got: {list(features.shape)}")

    # Test with batch
    print("\nTest 2: Batch of grids")
    grids_batch = torch.randint(0, 10, (4, 15, 15))
    print(f"Input shape: {grids_batch.shape}")
    features_batch = encoder(grids_batch)
    print(f"Output shape: {features_batch.shape}")
    print(f"Expected: [4, 225, 64], Got: {list(features_batch.shape)}")

    # Test with variable sizes
    print("\nTest 3: Different grid sizes")
    for H, W in [(5, 5), (10, 10), (20, 20), (30, 30)]:
        grid = torch.randint(0, 10, (2, H, W))
        features = encoder(grid)
        print(f"Grid {H}x{W}: Input {grid.shape} -> Output {features.shape}")

    # Test 8-connectivity
    print("\nTest 4: 8-connectivity")
    encoder_8 = ARCGridGNNEncoder(
        num_colors=10,
        feature_dim=64,
        hidden_dim=128,
        num_layers=4,
        edge_connectivity=8,
        use_position_features=True
    )
    grid = torch.randint(0, 10, (2, 10, 10))
    features_8 = encoder_8(grid)
    print(f"8-connectivity output shape: {features_8.shape}")

    # Test without position features
    print("\nTest 5: Without position features")
    encoder_no_pos = ARCGridGNNEncoder(
        num_colors=10,
        feature_dim=64,
        hidden_dim=128,
        num_layers=4,
        edge_connectivity=4,
        use_position_features=False
    )
    grid = torch.randint(0, 10, (2, 10, 10))
    features_no_pos = encoder_no_pos(grid)
    print(f"No position output shape: {features_no_pos.shape}")

    # Compare with CNN encoder (if available)
    print("\nTest 6: Compare with CNN encoder")
    try:
        from encoder import ARCGridEncoder

        cnn_encoder = ARCGridEncoder(
            num_colors=10,
            feature_dim=64,
            hidden_dim=128
        )

        grid = torch.randint(0, 10, (2, 10, 10))
        cnn_features = cnn_encoder(grid)
        gnn_features = encoder(grid)

        print(f"CNN output shape: {cnn_features.shape}")
        print(f"GNN output shape: {gnn_features.shape}")
        print(f"Shapes match: {cnn_features.shape == gnn_features.shape}")

        print(f"CNN parameters: {sum(p.numel() for p in cnn_encoder.parameters()):,}")
        print(f"GNN parameters: {encoder.get_num_parameters():,}")

    except ImportError:
        print("CNN encoder not available for comparison")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")

"""
Encoder for ARC grids that converts them to feature representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ARCGridEncoder(nn.Module):
    """
    CNN-based encoder for ARC grids.

    Takes a grid of shape [B, H, W] with integer color values (0-9)
    and produces feature maps of shape [B, H*W, feature_dim].
    """
    def __init__(self, num_colors=10, feature_dim=64, hidden_dim=128):
        super().__init__()
        self.num_colors = num_colors
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Embedding layer for colors (0-9 -> one-hot or learned embedding)
        self.color_embedding = nn.Embedding(num_colors, hidden_dim)

        # CNN encoder
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        self.conv4 = nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, grids):
        """
        Args:
            grids: [B, H, W] - Integer tensor with values 0-9

        Returns:
            features: [B, H*W, feature_dim] - Feature maps ready for slot attention
        """
        B, H, W = grids.shape

        # Embed colors: [B, H, W] -> [B, H, W, hidden_dim]
        x = self.color_embedding(grids)

        # Permute to [B, hidden_dim, H, W] for convolutions
        x = x.permute(0, 3, 1, 2)

        # Apply conv layers with residual connections
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Residual connection
        if identity.shape[1] == x.shape[1]:
            x = x + identity

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        # Reshape to [B, feature_dim, H*W] then [B, H*W, feature_dim]
        x = x.reshape(B, self.feature_dim, H * W)
        x = x.permute(0, 2, 1)

        return x

"""
Full model combining encoder + slot attention for instance recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import ARCGridEncoder
from slot import SlotAttention


class SlotInstanceModel(nn.Module):
    """
    Full model for instance recognition using Slot Attention.

    Architecture:
    1. Grid Encoder: Converts grid to feature maps
    2. Slot Attention: Decomposes features into object slots
    3. Pooling: Aggregates slots into single embedding
    4. Projection: Projects to embedding space for contrastive learning
    """
    def __init__(
        self,
        num_colors=10,
        encoder_feature_dim=64,
        encoder_hidden_dim=128,
        num_slots=7,
        slot_dim=64,
        num_iterations=3,
        embedding_dim=128,
        max_grid_size=30
    ):
        """
        Args:
            num_colors: Number of colors in ARC grids (0-9 = 10 colors)
            encoder_feature_dim: Feature dimension from encoder
            encoder_hidden_dim: Hidden dimension in encoder
            num_slots: Number of slots for slot attention
            slot_dim: Dimension of each slot
            num_iterations: Number of slot attention iterations
            embedding_dim: Final embedding dimension for contrastive learning
            max_grid_size: Maximum grid size (30 for ARC)
        """
        super().__init__()
        self.num_colors = num_colors
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.embedding_dim = embedding_dim

        # Encoder: Grid -> Features
        self.encoder = ARCGridEncoder(
            num_colors=num_colors,
            feature_dim=encoder_feature_dim,
            hidden_dim=encoder_hidden_dim
        )

        # Slot Attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            feature_dim=encoder_feature_dim,
            num_iterations=num_iterations,
            hidden_dim=128,
            max_spatial_size=max_grid_size
        )

        # Projection head: Slots -> Embedding
        # We'll pool slots and then project
        self.projection_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, embedding_dim)
        )

    def forward(self, grids):
        """
        Forward pass.

        Args:
            grids: [B, H, W] - Input grids with integer values 0-9

        Returns:
            embeddings: [B, embedding_dim] - Final embeddings for contrastive learning
            slots: [B, num_slots, slot_dim] - Intermediate slot representations
        """
        B, H, W = grids.shape

        # Encode grids to features
        # [B, H, W] -> [B, H*W, feature_dim]
        features = self.encoder(grids)

        # Apply slot attention
        # [B, H*W, feature_dim] -> [B, num_slots, slot_dim]
        slots = self.slot_attention(features, spatial_size=(H, W))

        # Pool slots to get single representation
        # Mean pooling across slots: [B, num_slots, slot_dim] -> [B, slot_dim]
        pooled = slots.mean(dim=1)

        # Project to embedding space
        # [B, slot_dim] -> [B, embedding_dim]
        embeddings = self.projection_head(pooled)

        # Normalize embeddings for contrastive learning
        embeddings = F.normalize(embeddings, dim=1)

        return embeddings, slots

    def get_embeddings(self, grids):
        """
        Convenience method to get just embeddings.

        Args:
            grids: [B, H, W]

        Returns:
            embeddings: [B, embedding_dim]
        """
        embeddings, _ = self.forward(grids)
        return embeddings

    def get_slots(self, grids):
        """
        Convenience method to get just slots.

        Args:
            grids: [B, H, W]

        Returns:
            slots: [B, num_slots, slot_dim]
        """
        _, slots = self.forward(grids)
        return slots

"""
Full model combining encoder + slot attention for instance recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import ARCGridEncoder
from slot import SlotAttention, SlotAttentionNoPos


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

    def forward(self, grids, return_attn=False):
        """
        Forward pass.

        Args:
            grids: [B, H, W] - Input grids with integer values 0-9
            return_attn: If True, return attention weights from slot attention

        Returns:
            embeddings: [B, embedding_dim] - Final embeddings for contrastive learning
            slots: [B, num_slots, slot_dim] - Intermediate slot representations
            attn_weights: [B, num_slots, H, W] - Attention masks (only if return_attn=True)
        """
        B, H, W = grids.shape

        # Encode grids to features
        # [B, H, W] -> [B, H*W, feature_dim]
        features = self.encoder(grids)

        # Apply slot attention
        # [B, H*W, feature_dim] -> [B, num_slots, slot_dim]
        if return_attn:
            slots, attn_weights = self.slot_attention(features, spatial_size=(H, W), return_attn=True)
            # Reshape attention weights from [B, num_slots, H*W] to [B, num_slots, H, W]
            attn_weights = attn_weights.reshape(B, self.num_slots, H, W)
        else:
            slots = self.slot_attention(features, spatial_size=(H, W))

        # Pool slots to get single representation
        # Mean pooling across slots: [B, num_slots, slot_dim] -> [B, slot_dim]
        pooled = slots.mean(dim=1)

        # Project to embedding space
        # [B, slot_dim] -> [B, embedding_dim]
        embeddings = self.projection_head(pooled)

        # Normalize embeddings for contrastive learning
        embeddings = F.normalize(embeddings, dim=1)

        if return_attn:
            return embeddings, slots, attn_weights
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


class HierarchicalSlotModel(nn.Module):
    """
    Hierarchical model with hard-coded color segmentation + learned slot attention.

    Architecture:
    1. Grid Encoder: Converts grid to per-pixel feature maps
    2. Color Pooling (HARD-CODED): Groups features by color (10 groups, one per ARC color)
    3. Slot Attention: Reasons over color features to produce object slots
    4. Projection: Projects to embedding space for contrastive learning

    The key insight: color segmentation is trivial for ARC (discrete 10 colors),
    so we hard-code it and let learning focus on higher-level reasoning.
    """
    def __init__(
        self,
        num_colors=10,
        encoder_feature_dim=64,
        encoder_hidden_dim=128,
        num_slots=5,
        slot_dim=64,
        num_iterations=3,
        embedding_dim=128,
        max_grid_size=30,
        color_feature_dim=None,  # If None, uses encoder_feature_dim
    ):
        """
        Args:
            num_colors: Number of colors in ARC grids (0-9 = 10 colors)
            encoder_feature_dim: Feature dimension from encoder
            encoder_hidden_dim: Hidden dimension in encoder
            num_slots: Number of output slots (can be < 10 to group colors)
            slot_dim: Dimension of each slot
            num_iterations: Number of slot attention iterations
            embedding_dim: Final embedding dimension for contrastive learning
            max_grid_size: Maximum grid size (30 for ARC)
            color_feature_dim: Dimension of color features (default: encoder_feature_dim)
        """
        super().__init__()
        self.num_colors = num_colors
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.embedding_dim = embedding_dim
        self.encoder_feature_dim = encoder_feature_dim

        if color_feature_dim is None:
            color_feature_dim = encoder_feature_dim
        self.color_feature_dim = color_feature_dim

        # Encoder: Grid -> Per-pixel features
        self.encoder = ARCGridEncoder(
            num_colors=num_colors,
            feature_dim=encoder_feature_dim,
            hidden_dim=encoder_hidden_dim
        )

        # Color feature projection (optional, if color_feature_dim != encoder_feature_dim)
        if color_feature_dim != encoder_feature_dim:
            self.color_projection = nn.Linear(encoder_feature_dim, color_feature_dim)
        else:
            self.color_projection = None

        # Additional features per color: we can add spatial info (centroid, bbox, count)
        # For now, we add: pixel count (normalized), centroid x, centroid y = 3 extra dims
        extra_color_features = 3
        total_color_feature_dim = color_feature_dim + extra_color_features

        # Slot Attention (without positional encoding - inputs are color features, not spatial)
        self.slot_attention = SlotAttentionNoPos(
            num_slots=num_slots,
            slot_dim=slot_dim,
            feature_dim=total_color_feature_dim,
            num_iterations=num_iterations,
            hidden_dim=128
        )

        # Projection head: Slots -> Embedding
        self.projection_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, embedding_dim)
        )

    def get_color_features(self, grids, encoded_features):
        """
        Extract features for each color via hard-coded pooling.

        Args:
            grids: [B, H, W] - Original grids with color values 0-9
            encoded_features: [B, H*W, feature_dim] - Per-pixel features from encoder

        Returns:
            color_features: [B, num_colors, color_feature_dim + 3]
                One feature vector per color, with spatial statistics
        """
        B, H, W = grids.shape
        N = H * W
        device = grids.device

        # Flatten grids for easier indexing
        grids_flat = grids.reshape(B, N)  # [B, H*W]

        # Apply color projection if needed
        if self.color_projection is not None:
            encoded_features = self.color_projection(encoded_features)

        # Create coordinate grids for centroid computation
        y_coords = torch.arange(H, device=device).float() / max(H - 1, 1)  # Normalized 0-1
        x_coords = torch.arange(W, device=device).float() / max(W - 1, 1)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [H*W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]

        color_features_list = []

        for c in range(self.num_colors):
            # Mask for this color: [B, H*W]
            mask = (grids_flat == c).float()
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1]

            # Weighted mean of encoded features for this color
            # [B, H*W, D] * [B, H*W, 1] -> sum -> [B, D]
            pooled_features = (encoded_features * mask.unsqueeze(-1)).sum(dim=1) / mask_sum

            # Pixel count (normalized by grid size)
            pixel_count = mask.sum(dim=1, keepdim=True) / N  # [B, 1]

            # Centroid (weighted mean of coordinates)
            # [B, H*W, 2] * [B, H*W, 1] -> sum -> [B, 2]
            centroid = (coords * mask.unsqueeze(-1)).sum(dim=1) / mask_sum

            # Combine: [B, color_feature_dim + 3]
            color_feat = torch.cat([pooled_features, pixel_count, centroid], dim=-1)
            color_features_list.append(color_feat)

        # Stack: [B, num_colors, color_feature_dim + 3]
        color_features = torch.stack(color_features_list, dim=1)

        return color_features

    def forward(self, grids, return_attn=False):
        """
        Forward pass.

        Args:
            grids: [B, H, W] - Input grids with integer values 0-9
            return_attn: If True, return attention weights from slot attention

        Returns:
            embeddings: [B, embedding_dim] - Final embeddings for contrastive learning
            slots: [B, num_slots, slot_dim] - Intermediate slot representations
            attn_weights: [B, num_slots, num_colors] - Slot attention over colors (only if return_attn=True)
        """
        B, H, W = grids.shape

        # Step 1: Encode grids to per-pixel features
        # [B, H, W] -> [B, H*W, feature_dim]
        encoded_features = self.encoder(grids)

        # Step 2: Hard-coded color pooling (NO LEARNING)
        # [B, H*W, feature_dim] + [B, H, W] -> [B, num_colors, color_feature_dim + 3]
        color_features = self.get_color_features(grids, encoded_features)

        # Step 3: Slot attention over color features (LEARNED)
        # [B, num_colors, color_feature_dim + 3] -> [B, num_slots, slot_dim]
        if return_attn:
            slots, attn_weights = self.slot_attention(color_features, return_attn=True)
        else:
            slots = self.slot_attention(color_features)

        # Step 4: Pool slots to get single representation
        pooled = slots.mean(dim=1)

        # Step 5: Project to embedding space
        embeddings = self.projection_head(pooled)

        # Normalize embeddings for contrastive learning
        embeddings = F.normalize(embeddings, dim=1)

        if return_attn:
            return embeddings, slots, attn_weights
        return embeddings, slots

    def get_embeddings(self, grids):
        """Convenience method to get just embeddings."""
        embeddings, _ = self.forward(grids)
        return embeddings

    def get_slots(self, grids):
        """Convenience method to get just slots."""
        _, slots = self.forward(grids)
        return slots

    def get_color_attention(self, grids):
        """
        Get attention weights showing how slots attend to colors.

        Returns:
            attn: [B, num_slots, num_colors] - Which colors each slot focuses on
        """
        _, _, attn = self.forward(grids, return_attn=True)
        return attn

"""
SlotAttentionEncoder for converting CNN features to transformer-ready slots.

This module bridges the gap between CNN feature extraction and transformer-based reasoning,
converting spatial features into object-centric slot representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlotAttentionEncoder(nn.Module):
    """
    Converts CNN features to slot representations for transformer reasoning.

    This encoder uses slot attention to extract object-centric representations from
    CNN features, then projects them to the transformer's hidden dimension.

    Architecture:
        CNN Features [batch, num_positions, slot_dim]
            ↓
        Slot Attention (iterative attention mechanism)
            ↓
        Slots [batch, num_slots, slot_dim]
            ↓
        Projection (Linear + LayerNorm)
            ↓
        Transformer-ready Slots [batch, num_slots, hidden_size]

    Args:
        num_slots: Number of object slots to extract
        slot_dim: Dimension of slot features from CNN
        hidden_size: Dimension of transformer hidden states
        num_iterations: Number of slot attention refinement iterations
        mlp_hidden_size: Hidden size for slot update MLP
        max_spatial_size: Maximum spatial dimension (for positional encodings)
        forward_dtype: Data type for forward pass
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        hidden_size: int,
        num_iterations: int = 3,
        mlp_hidden_size: int = 128,
        max_spatial_size: int = 30,
        forward_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.max_spatial_size = max_spatial_size
        self.forward_dtype = forward_dtype
        self.eps = 1e-8

        # Learned 2D spatial positional encodings
        # These provide spatial awareness to the slot attention mechanism
        self.spatial_pos_encoding = nn.Parameter(
            torch.randn(1, max_spatial_size, max_spatial_size, slot_dim) * 0.02
        )

        # Learned slot initialization parameters
        # Slots are initialized from a learned Gaussian distribution
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Layer normalization
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_inputs = nn.LayerNorm(slot_dim)

        # Attention projections (Queries, Keys, Values)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)

        # Slot update mechanism using GRU
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_dim)
        )

        # Project from slot_dim to transformer hidden_size
        if slot_dim != hidden_size:
            self.project_to_hidden = nn.Sequential(
                nn.Linear(slot_dim, hidden_size),
                nn.LayerNorm(hidden_size)
            )
        else:
            self.project_to_hidden = nn.Identity()

    def _infer_spatial_dims(self, num_positions: int):
        """
        Infer spatial dimensions from number of positions.

        Assumes square grid if possible, otherwise finds best factorization.

        Args:
            num_positions: Total number of spatial positions (H * W)

        Returns:
            (H, W): Spatial dimensions
        """
        # Try square first
        sqrt = int(math.sqrt(num_positions))
        if sqrt * sqrt == num_positions:
            return sqrt, sqrt

        # Otherwise, find factors closest to square
        best_h, best_w = 1, num_positions
        for h in range(2, int(math.sqrt(num_positions)) + 1):
            if num_positions % h == 0:
                w = num_positions // h
                if abs(h - w) < abs(best_h - best_w):
                    best_h, best_w = h, w

        return best_h, best_w

    def forward(self, features: torch.Tensor, spatial_size: tuple = None, return_attn: bool = False):
        """
        Apply slot attention to extract object-centric representations.

        Args:
            features: CNN features [batch, num_positions, slot_dim]
            spatial_size: Optional (H, W) tuple specifying spatial dimensions.
                         If None, inferred from num_positions (assumes square grid).
            return_attn: If True, return attention weights from final iteration

        Returns:
            slots: Transformer-ready slots [batch, num_slots, hidden_size]
            attn_weights: (Optional) Attention weights [batch, num_slots, num_positions]
        """
        B, N, D = features.shape
        assert D == self.slot_dim, f"Expected slot_dim={self.slot_dim}, got {D}"

        # Infer or validate spatial dimensions
        if spatial_size is not None:
            H, W = spatial_size
            assert H * W == N, f"Spatial size {H}×{W} doesn't match num_positions={N}"
        else:
            H, W = self._infer_spatial_dims(N)

        # Reshape features to spatial format [B, H, W, D]
        features_spatial = features.reshape(B, H, W, D)

        # Add learned spatial positional encodings
        # Extract relevant portion for actual spatial size
        pos_enc = self.spatial_pos_encoding[:, :H, :W, :]  # [1, H, W, slot_dim]
        features_spatial = features_spatial + pos_enc.to(self.forward_dtype)

        # Reshape back to [B, N, D]
        inputs = features_spatial.reshape(B, N, D)

        # Initialize slots from learned Gaussian distribution
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # Normalize inputs and compute keys/values once
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # [B, N, slot_dim]
        v = self.project_v(inputs)  # [B, N, slot_dim]

        # Iterative slot attention refinement
        attn_weights = None
        for iteration in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Compute queries from slots
            q = self.project_q(slots)  # [B, num_slots, slot_dim]

            # Attention weights: slots compete for input features
            # [B, num_slots, slot_dim] @ [B, slot_dim, N] -> [B, num_slots, N]
            attn_logits = torch.bmm(q, k.transpose(-1, -2)) / (self.slot_dim ** 0.5)
            attn = F.softmax(attn_logits, dim=1)  # Softmax over slots (slots compete for each position)

            # Weighted normalization across slots (ensures proper competition)
            attn = attn + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)

            # Store final iteration attention if requested
            if return_attn and iteration == self.num_iterations - 1:
                attn_weights = attn

            # Weighted aggregation of input features
            # [B, num_slots, N] @ [B, N, slot_dim] -> [B, num_slots, slot_dim]
            updates = torch.bmm(attn, v)

            # Update slots using GRU (recurrent update)
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim)
            )
            slots = slots.reshape(B, self.num_slots, self.slot_dim)

            # Residual MLP for further refinement
            slots = slots + self.mlp(slots)

        # Project slots to transformer hidden dimension
        slots = self.project_to_hidden(slots.to(self.forward_dtype))

        if return_attn:
            return slots, attn_weights
        return slots

    def get_attention_maps(self, features: torch.Tensor, spatial_size: tuple = None):
        """
        Utility method to visualize which spatial positions each slot attends to.

        Args:
            features: CNN features [batch, num_positions, slot_dim]
            spatial_size: Optional (H, W) tuple

        Returns:
            slots: [batch, num_slots, hidden_size]
            attn_maps: [batch, num_slots, H, W] - Spatial attention maps
        """
        slots, attn_weights = self.forward(features, spatial_size, return_attn=True)

        # Reshape attention to spatial format
        B, num_slots, N = attn_weights.shape
        if spatial_size is not None:
            H, W = spatial_size
        else:
            H, W = self._infer_spatial_dims(N)

        attn_maps = attn_weights.reshape(B, num_slots, H, W)

        return slots, attn_maps

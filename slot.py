import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Slot Attention module for object-centric learning.
    
    Based on "Object-Centric Learning with Slot Attention" (Locatello et al., 2020).
    Learns to decompose input features into a fixed number of slots through iterative attention.
    """
    def __init__(self, num_slots, slot_dim, feature_dim, num_iterations=3, hidden_dim=128, eps=1e-8, max_spatial_size=30):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.eps = eps
        self.max_spatial_size = max_spatial_size
        
        # Learned 2D spatial positional encodings
        # These will be added to input features to give spatial awareness
        # Shape: [1, max_spatial_size, max_spatial_size, feature_dim]
        self.spatial_pos_encoding = nn.Parameter(
            torch.randn(1, max_spatial_size, max_spatial_size, feature_dim) * 0.02
        )
        
        # Learned slot initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        
        # Layer norm for slots and inputs
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_inputs = nn.LayerNorm(feature_dim)
        
        # Linear maps for attention (Q, K, V)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(feature_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(feature_dim, slot_dim, bias=False)
        
        # Slot update functions
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        
    def forward(self, inputs, spatial_size=None, return_attn=False):
        """
        Args:
            inputs: Feature maps [B, H*W, feature_dim] or [B, num_features, feature_dim]
            spatial_size: Optional tuple (H, W) indicating spatial dimensions of the features.
                         If None, assumes square grid and computes from input size.
            return_attn: If True, return attention weights from final iteration

        Returns:
            slots: [B, num_slots, slot_dim]
            attn: [B, num_slots, H*W] - Only returned if return_attn=True
        """
        B, N, D = inputs.shape
        
        # Determine spatial dimensions
        if spatial_size is not None:
            H, W = spatial_size
        else:
            # Assume square grid
            H = W = int(N ** 0.5)
            assert H * W == N, f"Cannot infer spatial size from N={N}, provide spatial_size parameter"
        
        # Reshape inputs to spatial format [B, H, W, D]
        inputs_spatial = inputs.reshape(B, H, W, D)
        
        # Add learned spatial positional encodings
        # Extract the relevant portion of positional encodings for the actual spatial size
        pos_enc = self.spatial_pos_encoding[:, :H, :W, :]  # [1, H, W, feature_dim]
        inputs_spatial = inputs_spatial + pos_enc  # Broadcasting across batch dimension
        
        # Reshape back to [B, N, D]
        inputs = inputs_spatial.reshape(B, N, D)
        
        # Initialize slots from learned distribution
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        # Normalize inputs
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # [B, N, slot_dim]
        v = self.project_v(inputs)  # [B, N, slot_dim]
        
        # Iterative attention
        attn_weights = None
        for iteration in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention
            q = self.project_q(slots)  # [B, num_slots, slot_dim]

            # Compute attention weights
            # [B, num_slots, slot_dim] @ [B, slot_dim, N] -> [B, num_slots, N]
            attn_logits = torch.bmm(q, k.transpose(-1, -2)) / (self.slot_dim ** 0.5)
            attn = F.softmax(attn_logits, dim=1)  # Softmax over slots (slots compete for each position)

            # Weighted normalization across slots (ensures proper competition)
            attn = attn + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)

            # Store attention from final iteration if requested
            if return_attn and iteration == self.num_iterations - 1:
                attn_weights = attn

            # Weighted mean
            # [B, num_slots, N] @ [B, N, slot_dim] -> [B, num_slots, slot_dim]
            updates = torch.bmm(attn, v)

            # Update slots with GRU
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim)
            )
            slots = slots.reshape(B, self.num_slots, self.slot_dim)

            # Apply MLP
            slots = slots + self.mlp(slots)

        if return_attn:
            return slots, attn_weights
        return slots


class SlotAttentionNoPos(nn.Module):
    """
    Slot Attention without spatial positional encoding.

    Used for non-spatial inputs like color-pooled features where the input
    is a set of feature vectors without spatial arrangement.
    """
    def __init__(self, num_slots, slot_dim, feature_dim, num_iterations=3, hidden_dim=128, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.eps = eps

        # Learned slot initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Layer norm for slots and inputs
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_inputs = nn.LayerNorm(feature_dim)

        # Linear maps for attention (Q, K, V)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(feature_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(feature_dim, slot_dim, bias=False)

        # Slot update functions
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )

    def forward(self, inputs, return_attn=False):
        """
        Args:
            inputs: [B, num_features, feature_dim] - Non-spatial feature vectors
            return_attn: If True, return attention weights from final iteration

        Returns:
            slots: [B, num_slots, slot_dim]
            attn: [B, num_slots, num_features] - Only returned if return_attn=True
        """
        B, N, D = inputs.shape

        # Initialize slots from learned distribution
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        # Normalize inputs (no positional encoding for non-spatial inputs)
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # [B, N, slot_dim]
        v = self.project_v(inputs)  # [B, N, slot_dim]

        # Iterative attention
        attn_weights = None
        for iteration in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention
            q = self.project_q(slots)  # [B, num_slots, slot_dim]

            # Compute attention weights
            attn_logits = torch.bmm(q, k.transpose(-1, -2)) / (self.slot_dim ** 0.5)
            attn = F.softmax(attn_logits, dim=1)  # Softmax over slots

            # Weighted normalization across slots
            attn = attn + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)

            # Store attention from final iteration if requested
            if return_attn and iteration == self.num_iterations - 1:
                attn_weights = attn

            # Weighted mean
            updates = torch.bmm(attn, v)

            # Update slots with GRU
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim)
            )
            slots = slots.reshape(B, self.num_slots, self.slot_dim)

            # Apply MLP
            slots = slots + self.mlp(slots)

        if return_attn:
            return slots, attn_weights
        return slots


class SlotDecoder(nn.Module):
    """
    Decoder for slot attention that reconstructs the grid from slots.
    Each slot is decoded independently and combined via broadcasting.
    """
    def __init__(self, slot_dim, num_colors, hidden_dim, max_grid_size=30, num_slots=7):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        self.num_slots = num_slots
        
        # Learned positional encodings for each slot
        # These give each slot position-specific information during decoding
        self.slot_pos_encoding = nn.Parameter(torch.randn(num_slots, slot_dim))
        
        # Decode each slot to a spatial feature map
        self.fc_decode = nn.Linear(slot_dim, hidden_dim * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm2d(hidden_dim)
        
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                         kernel_size=4, stride=2, padding=1)
        self.bn_d3 = nn.BatchNorm2d(hidden_dim)
        
        self.conv_out1 = nn.Conv2d(hidden_dim, hidden_dim,
                                   kernel_size=3, padding=1)
        self.bn_out1 = nn.BatchNorm2d(hidden_dim)
        
        # Output: num_colors + 1 for alpha mask
        self.conv_out = nn.Conv2d(hidden_dim, num_colors + 1, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def decode_slot(self, slot):
        """Decode a single slot to spatial features."""
        x = self.relu(self.fc_decode(slot))
        x = x.reshape(-1, self.hidden_dim, 4, 4)
        
        x = self.relu(self.bn_d1(self.deconv1(x)))
        x = self.relu(self.bn_d2(self.deconv2(x)))
        x = self.relu(self.bn_d3(self.deconv3(x)))
        x = self.relu(self.bn_out1(self.conv_out1(x)))
        
        x = self.conv_out(x)  # [num_slots, num_colors+1, H, W]
        
        return x
    
    def forward(self, slots, target_size):
        """
        Args:
            slots: [B, num_slots, slot_dim]
            target_size: (H, W) target output size
            
        Returns:
            logits: [B, num_colors, H, W]
        """
        B, num_slots, slot_dim = slots.shape
        H, W = target_size
        
        # Add learned positional encodings to slots
        # Expand positional encodings for batch: [num_slots, slot_dim] -> [B, num_slots, slot_dim]
        pos_encodings = self.slot_pos_encoding.unsqueeze(0).expand(B, -1, -1)
        slots = slots + pos_encodings  # Add positional information to each slot
        
        # Decode all slots
        slots_flat = slots.reshape(B * num_slots, slot_dim)
        decoded = self.decode_slot(slots_flat)  # [B*num_slots, num_colors+1, 32, 32]
        
        # Reshape to [B, num_slots, num_colors+1, 32, 32]
        decoded = decoded.reshape(B, num_slots, self.num_colors + 1, 
                                 decoded.shape[-2], decoded.shape[-1])
        
        # Resize to target size
        if H != decoded.shape[-2] or W != decoded.shape[-1]:
            decoded = F.interpolate(decoded.reshape(B * num_slots, self.num_colors + 1,
                                                   decoded.shape[-2], decoded.shape[-1]),
                                   size=(H, W), mode='bilinear', align_corners=False)
            decoded = decoded.reshape(B, num_slots, self.num_colors + 1, H, W)
        
        # Split into reconstructions and masks
        recons = decoded[:, :, :self.num_colors, :, :]  # [B, num_slots, num_colors, H, W]
        masks = decoded[:, :, self.num_colors:, :, :]   # [B, num_slots, 1, H, W]
        
        # Normalize masks across slots
        masks = F.softmax(masks, dim=1)
        
        # Combine slots using masks (mixture of slots)
        recon_combined = (recons * masks).sum(dim=1)  # [B, num_colors, H, W]
        
        return recon_combined
"""
Per-puzzle ARC solver.

Approach: Train a small network to memorize one puzzle's transformation rule,
then test if it generalizes to the held-out test input.

The architecture is intentionally compact:
- Slot attention for object perception
- Small transformer for reasoning
- Cross-attention decoder for output generation

Training uses heavy augmentation to prevent pixel-level memorization
and encourage learning the actual rule.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Tuple, Optional, Dict


# =============================================================================
# Core Architecture
# =============================================================================

class CNNEncoder(nn.Module):
    """
    Small CNN to extract local spatial features before slot attention.
    """
    def __init__(self, num_colors: int = 10, hidden_dim: int = 48, num_layers: int = 3):
        super().__init__()
        self.color_embed = nn.Embedding(num_colors, hidden_dim)

        # Stack of 3x3 convs with residual connections
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
            ))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.GELU()

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: [B, H, W] discrete color indices
        Returns:
            features: [B, H, W, hidden_dim]
        """
        # Embed colors: [B, H, W] -> [B, H, W, D]
        x = self.color_embed(grid)

        # Reshape for conv: [B, H, W, D] -> [B, D, H, W]
        x = x.permute(0, 3, 1, 2)

        # Apply conv layers with residual connections
        for layer in self.layers:
            x = x + layer(x)  # Residual
            x = self.activation(x)

        # Back to [B, H, W, D]
        x = x.permute(0, 2, 3, 1)
        return x


class SlotEncoder(nn.Module):
    """
    Lightweight slot attention encoder.
    Extracts object slots with explicit properties.
    """
    def __init__(
        self,
        num_slots: int = 10,
        slot_dim: int = 48,
        num_colors: int = 10,
        num_iterations: int = 3,
        max_size: int = 30,
        use_cnn: bool = False,
        cnn_layers: int = 3,
        color_slots: bool = False
    ):
        super().__init__()
        self.num_colors = num_colors
        self.slot_dim = slot_dim
        self.use_cnn = use_cnn
        self.color_slots = color_slots

        # Color slots mode: exactly 10 slots, one per color
        if color_slots:
            self.num_slots = num_colors
        else:
            self.num_slots = num_slots

        # Input embedding - either CNN or simple embedding
        if use_cnn:
            self.cnn = CNNEncoder(num_colors, slot_dim, num_layers=cnn_layers)
            self.color_embed = None  # Not used when CNN is enabled
        else:
            self.cnn = None
            self.color_embed = nn.Embedding(num_colors, slot_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, max_size, max_size, slot_dim) * 0.02)
        self.norm_in = nn.LayerNorm(slot_dim)

        # Slot attention components (only needed when not using color_slots)
        if not color_slots:
            self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
            self.slot_sigma = nn.Parameter(torch.ones(1, 1, slot_dim) * 0.1)
            self.norm_slot = nn.LayerNorm(slot_dim)
            self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
            self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
            self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)
            self.gru = nn.GRUCell(slot_dim, slot_dim)
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, slot_dim * 2),
                nn.GELU(),
                nn.Linear(slot_dim * 2, slot_dim)
            )
            self.num_iterations = num_iterations

        # Property dimensions: size(1) + centroid(2) + bbox(4) + colors(num_colors)
        self.prop_dim = 7 + num_colors
        
    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grid: [B, H, W] input grid

        Returns:
            slots: [B, num_slots, slot_dim]
            attn: [B, num_slots, H, W]
            props: [B, num_slots, prop_dim]
        """
        B, H, W = grid.shape
        device = grid.device

        # Embed input - use CNN or simple embedding
        if self.use_cnn:
            x = self.cnn(grid) + self.pos_embed[:, :H, :W, :]
        else:
            x = self.color_embed(grid) + self.pos_embed[:, :H, :W, :]

        x = x.reshape(B, H * W, -1)
        x = self.norm_in(x)

        if self.color_slots:
            # Color slots mode: hard pooling by color
            grid_flat = grid.reshape(B, -1)  # [B, H*W]
            slots_list = []
            attn_list = []

            for c in range(self.num_colors):
                mask = (grid_flat == c).float()  # [B, H*W]
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

                # Pool features for this color
                pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / count  # [B, D]
                slots_list.append(pooled)

                # Attention mask (normalized)
                attn_list.append(mask / count)

            slots = torch.stack(slots_list, dim=1)  # [B, num_colors, D]
            attn = torch.stack(attn_list, dim=1).reshape(B, self.num_colors, H, W)
        else:
            # Standard slot attention
            slots = self.slot_mu + self.slot_sigma * torch.randn(
                B, self.num_slots, self.slot_dim, device=device
            )

            k = self.to_k(x)
            v = self.to_v(x)

            # Iterate
            for _ in range(self.num_iterations):
                slots_prev = slots
                q = self.to_q(self.norm_slot(slots))

                attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.slot_dim)
                attn = F.softmax(attn_logits, dim=1)  # Competition over slots
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

                updates = torch.bmm(attn, v)
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim)
                ).reshape(B, self.num_slots, -1)
                slots = slots + self.mlp(slots)

            # Final attention
            q = self.to_q(self.norm_slot(slots))
            attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.slot_dim)
            attn = F.softmax(attn_logits, dim=1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            attn = attn.reshape(B, self.num_slots, H, W)

        # Compute properties
        props = self._compute_props(attn, grid)

        return slots, attn, props
    
    def _compute_props(self, attn: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Compute explicit object properties from attention masks."""
        B, num_slots, H, W = attn.shape
        device = attn.device
        
        # Coordinate grids (normalized to [0, 1])
        yy = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, num_slots, H, W)
        xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, num_slots, H, W)
        
        # Size (attention mass)
        size = attn.sum(dim=[2, 3])  # [B, num_slots]
        size_norm = size / (H * W)
        
        # Centroid
        attn_safe = attn + 1e-8
        centroid_y = (attn_safe * yy).sum(dim=[2, 3]) / attn_safe.sum(dim=[2, 3])
        centroid_x = (attn_safe * xx).sum(dim=[2, 3]) / attn_safe.sum(dim=[2, 3])
        
        # Bounding box (approximate via weighted std)
        var_y = (attn_safe * (yy - centroid_y.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=[2, 3]) / attn_safe.sum(dim=[2, 3])
        var_x = (attn_safe * (xx - centroid_x.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=[2, 3]) / attn_safe.sum(dim=[2, 3])
        height = 2 * torch.sqrt(var_y + 1e-8)  # Approximate extent
        width = 2 * torch.sqrt(var_x + 1e-8)
        
        # Color distribution
        colors_onehot = F.one_hot(grid, self.num_colors).float()  # [B, H, W, C]
        colors_onehot = colors_onehot.permute(0, 3, 1, 2).unsqueeze(1)  # [B, 1, C, H, W]
        attn_exp = attn.unsqueeze(2)  # [B, num_slots, 1, H, W]
        color_dist = (attn_exp * colors_onehot).sum(dim=[3, 4])  # [B, num_slots, C]
        color_dist = color_dist / (color_dist.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Concatenate all properties
        props = torch.cat([
            size_norm.unsqueeze(-1),
            centroid_y.unsqueeze(-1),
            centroid_x.unsqueeze(-1),
            height.unsqueeze(-1),
            width.unsqueeze(-1),
            (height * width).unsqueeze(-1),  # Area proxy
            (height / (width + 1e-8)).unsqueeze(-1),  # Aspect ratio
            color_dist
        ], dim=-1)
        
        return props


class Reasoner(nn.Module):
    """
    Small transformer for slot-to-slot reasoning.
    """
    def __init__(self, slot_dim: int = 48, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=slot_dim,
                nhead=num_heads,
                dim_feedforward=slot_dim * 2,
                dropout=0.0,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            slots = layer(slots)
        return slots


class Decoder(nn.Module):
    """
    Cross-attention decoder: output positions query slots.
    """
    def __init__(self, slot_dim: int = 48, num_colors: int = 10, max_size: int = 30):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_colors = num_colors
        
        self.pos_embed = nn.Parameter(torch.randn(1, max_size, max_size, slot_dim) * 0.02)
        
        self.cross_attn = nn.MultiheadAttention(
            slot_dim, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(slot_dim)
        
        self.head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, num_colors)
        )
        
    def forward(self, slots: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            slots: [B, num_slots, slot_dim]
            output_size: (H, W)
            
        Returns:
            logits: [B, num_colors, H, W]
        """
        B = slots.shape[0]
        H, W = output_size
        
        # Position queries
        pos = self.pos_embed[:, :H, :W, :].expand(B, -1, -1, -1)
        pos = pos.reshape(B, H * W, self.slot_dim)
        
        # Cross-attention
        out, _ = self.cross_attn(pos, slots, slots)
        out = self.norm(pos + out)
        
        # Predict colors
        logits = self.head(out)
        logits = logits.reshape(B, H, W, self.num_colors).permute(0, 3, 1, 2)
        
        return logits


class SizePredictor(nn.Module):
    """
    Predicts output grid size from slot representations.
    Treats H and W as classification over 1-30.
    """
    def __init__(self, slot_dim: int = 48, num_slots: int = 10, max_size: int = 30):
        super().__init__()
        self.max_size = max_size

        # Pool slots and predict size
        self.pool = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
        )
        self.head_h = nn.Linear(slot_dim, max_size)  # Predict height (1 to max_size)
        self.head_w = nn.Linear(slot_dim, max_size)  # Predict width (1 to max_size)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: [B, num_slots, slot_dim]
        Returns:
            h_logits: [B, max_size] - logits for height
            w_logits: [B, max_size] - logits for width
        """
        # Mean pool over slots
        pooled = slots.mean(dim=1)  # [B, slot_dim]
        pooled = self.pool(pooled)

        h_logits = self.head_h(pooled)
        w_logits = self.head_w(pooled)

        return h_logits, w_logits

    def predict_size(self, slots: torch.Tensor) -> Tuple[int, int]:
        """Get predicted (H, W) as integers."""
        h_logits, w_logits = self.forward(slots)
        h = h_logits.argmax(dim=-1).item() + 1  # +1 because class 0 = size 1
        w = w_logits.argmax(dim=-1).item() + 1
        return (h, w)


class PuzzleSolver(nn.Module):
    """
    Complete puzzle solver: encode → reason → decode.
    """
    def __init__(
        self,
        num_slots: int = 10,
        slot_dim: int = 48,
        num_colors: int = 10,
        num_reasoning_layers: int = 2,
        max_size: int = 30,
        use_cnn: bool = False,
        cnn_layers: int = 3,
        color_slots: bool = False
    ):
        super().__init__()
        self.encoder = SlotEncoder(
            num_slots, slot_dim, num_colors,
            max_size=max_size, use_cnn=use_cnn, cnn_layers=cnn_layers,
            color_slots=color_slots
        )
        self.reasoner = Reasoner(slot_dim, num_layers=num_reasoning_layers)
        self.decoder = Decoder(slot_dim, num_colors, max_size)
        self.size_predictor = SizePredictor(slot_dim, num_slots, max_size)
        
    def forward(
        self,
        input_grid: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_grid: [B, H, W]
            output_size: (H_out, W_out), defaults to input size

        Returns:
            dict with: logits, slots, attn, props, size_h_logits, size_w_logits
        """
        B, H, W = input_grid.shape
        if output_size is None:
            output_size = (H, W)

        slots, attn, props = self.encoder(input_grid)
        slots = self.reasoner(slots)
        logits = self.decoder(slots, output_size)

        # Size prediction
        size_h_logits, size_w_logits = self.size_predictor(slots)

        return {
            'logits': logits,
            'slots': slots,
            'attn': attn,
            'props': props,
            'size_h_logits': size_h_logits,
            'size_w_logits': size_w_logits,
        }

    def predict(self, input_grid: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Get predicted grid. If output_size is None, uses predicted size."""
        # Encode and reason
        slots, _, _ = self.encoder(input_grid)
        slots = self.reasoner(slots)

        # Predict size if not given
        if output_size is None:
            output_size = self.size_predictor.predict_size(slots)

        # Decode
        logits = self.decoder(slots, output_size)
        return logits.argmax(dim=1)


# =============================================================================
# Augmentations (Critical for per-puzzle training!)
# =============================================================================

class AugmentPair:
    """
    Augmentations that apply consistently to input-output pairs.
    """
    def __init__(self, p_flip: float = 0.5, p_rot: float = 0.5, p_color_perm: float = 0.5):
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.p_color_perm = p_color_perm
        
    def __call__(
        self, 
        input_grid: torch.Tensor, 
        output_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply same augmentation to input and output.
        
        Args:
            input_grid: [H_in, W_in]
            output_grid: [H_out, W_out]
        """
        # Horizontal flip
        if random.random() < self.p_flip:
            input_grid = input_grid.flip(-1)
            output_grid = output_grid.flip(-1)
            
        # Vertical flip
        if random.random() < self.p_flip:
            input_grid = input_grid.flip(-2)
            output_grid = output_grid.flip(-2)
            
        # Rotation (0, 90, 180, 270)
        if random.random() < self.p_rot:
            k = random.randint(0, 3)
            input_grid = torch.rot90(input_grid, k, dims=[-2, -1])
            output_grid = torch.rot90(output_grid, k, dims=[-2, -1])
            
        # Color permutation (preserving 0 as background)
        if random.random() < self.p_color_perm:
            # Create random permutation for colors 1-9, keep 0 fixed
            perm = torch.cat([
                torch.tensor([0]),
                torch.randperm(9) + 1
            ])
            input_grid = perm[input_grid]
            output_grid = perm[output_grid]
            
        return input_grid, output_grid


# =============================================================================
# Per-Puzzle Training
# =============================================================================

class PuzzleTrainer:
    """
    Train a model on a single puzzle's examples.
    """
    def __init__(
        self,
        model: PuzzleSolver,
        lr: float = 1e-3,
        device: str = 'cuda',
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.augment = AugmentPair()

        # Mixed precision training for GPU speedup
        self.use_amp = use_amp and device != 'cpu'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
    def train_epoch(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        augment_factor: int = 10,
        size_loss_weight: float = 0.1,
        batch_size: int = 32
    ) -> float:
        """
        Train one epoch on puzzle examples with batched processing.

        Args:
            examples: List of (input_grid, output_grid) pairs
            augment_factor: How many augmented versions per example
            size_loss_weight: Weight for size prediction loss
            batch_size: Number of samples per batch (GPU efficiency)

        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        count = 0

        # Pre-generate all augmented examples for batching
        augmented_pairs = []
        for _ in range(augment_factor):
            for input_grid, output_grid in examples:
                inp, out = self.augment(input_grid.clone(), output_grid.clone())
                augmented_pairs.append((inp, out))

        # Shuffle for better training
        random.shuffle(augmented_pairs)

        # Process in batches grouped by output size (required for same-size batching)
        # Group by (input_H, input_W, output_H, output_W)
        size_groups: Dict[Tuple[int, int, int, int], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        for inp, out in augmented_pairs:
            key = (inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
            if key not in size_groups:
                size_groups[key] = []
            size_groups[key].append((inp, out))

        # Train on each size group
        for (_, _, out_h, out_w), pairs in size_groups.items():
            # Process in batches
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                B = len(batch_pairs)

                # Stack into batched tensors
                inp_batch = torch.stack([p[0] for p in batch_pairs]).to(self.device)
                out_batch = torch.stack([p[1] for p in batch_pairs]).to(self.device)

                # Target sizes (same for all in batch)
                target_h = torch.full((B,), out_h - 1, dtype=torch.long, device=self.device)
                target_w = torch.full((B,), out_w - 1, dtype=torch.long, device=self.device)

                self.optimizer.zero_grad()

                # Forward with mixed precision
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        result = self.model(inp_batch, (out_h, out_w))
                        pixel_loss = F.cross_entropy(result['logits'], out_batch)
                        size_loss_h = F.cross_entropy(result['size_h_logits'], target_h)
                        size_loss_w = F.cross_entropy(result['size_w_logits'], target_w)
                        size_loss = (size_loss_h + size_loss_w) / 2
                        loss = pixel_loss + size_loss_weight * size_loss

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    result = self.model(inp_batch, (out_h, out_w))
                    pixel_loss = F.cross_entropy(result['logits'], out_batch)
                    size_loss_h = F.cross_entropy(result['size_h_logits'], target_h)
                    size_loss_w = F.cross_entropy(result['size_w_logits'], target_w)
                    size_loss = (size_loss_h + size_loss_w) / 2
                    loss = pixel_loss + size_loss_weight * size_loss

                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * B
                count += B

        return total_loss / count if count > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[float, float, float]:
        """
        Evaluate on examples (no augmentation).

        Returns:
            (loss, pixel_accuracy, size_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        size_correct = 0

        for input_grid, output_grid in examples:
            inp = input_grid.unsqueeze(0).to(self.device)
            out = output_grid.unsqueeze(0).to(self.device)

            output_size = (out.shape[1], out.shape[2])
            result = self.model(inp, output_size)

            loss = F.cross_entropy(result['logits'], out)
            total_loss += loss.item()

            pred = result['logits'].argmax(dim=1)
            total_correct += (pred == out).sum().item()
            total_pixels += out.numel()

            # Check size prediction
            pred_h = result['size_h_logits'].argmax(dim=-1).item() + 1
            pred_w = result['size_w_logits'].argmax(dim=-1).item() + 1
            if pred_h == output_size[0] and pred_w == output_size[1]:
                size_correct += 1

        avg_loss = total_loss / len(examples)
        pixel_accuracy = total_correct / total_pixels
        size_accuracy = size_correct / len(examples)

        return avg_loss, pixel_accuracy, size_accuracy
    
    @torch.no_grad()
    def predict(
        self,
        input_grid: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Get prediction for test input. If output_size is None, uses predicted size."""
        self.model.eval()
        inp = input_grid.unsqueeze(0).to(self.device)
        return self.model.predict(inp, output_size).squeeze(0).cpu()

    @torch.no_grad()
    def predict_size(self, input_grid: torch.Tensor) -> Tuple[int, int]:
        """Get predicted output size for an input grid."""
        self.model.eval()
        inp = input_grid.unsqueeze(0).to(self.device)
        slots, _, _ = self.model.encoder(inp)
        slots = self.model.reasoner(slots)
        return self.model.size_predictor.predict_size(slots)
    
    @torch.no_grad()
    def get_slot_visualization(
        self,
        input_grid: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get slot attention masks and properties for visualization."""
        self.model.eval()
        inp = input_grid.unsqueeze(0).to(self.device)
        result = self.model(inp)
        
        return {
            'attn': result['attn'].squeeze(0).cpu(),  # [num_slots, H, W]
            'props': result['props'].squeeze(0).cpu(),  # [num_slots, prop_dim]
        }


def fit_puzzle(
    train_examples: List[Tuple[torch.Tensor, torch.Tensor]],
    test_input: torch.Tensor,
    num_epochs: int = 500,
    num_slots: int = 10,
    slot_dim: int = 48,
    lr: float = 1e-3,
    augment_factor: int = 10,
    device: str = 'cuda',
    verbose: bool = True,
    use_cnn: bool = False,
    cnn_layers: int = 3,
    color_slots: bool = False,
    batch_size: int = 32,
    use_amp: bool = True
) -> Tuple[torch.Tensor, PuzzleSolver, Tuple[int, int]]:
    """
    Fit a model to one puzzle and return prediction.

    Args:
        train_examples: List of (input, output) tensor pairs
        test_input: Test input grid
        num_epochs: Training epochs
        num_slots: Number of slots
        slot_dim: Slot dimension
        lr: Learning rate
        augment_factor: Augmentations per example per epoch
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        (predicted_output, trained_model, predicted_size)
    """
    # Create model
    model = PuzzleSolver(
        num_slots=num_slots,
        slot_dim=slot_dim,
        num_colors=10,
        num_reasoning_layers=2,
        max_size=30,
        use_cnn=use_cnn,
        cnn_layers=cnn_layers,
        color_slots=color_slots
    )

    if verbose:
        mode_parts = []
        if color_slots:
            mode_parts.append("color slots")
        if use_cnn:
            mode_parts.append(f"CNN ({cnn_layers} layers)")
        else:
            mode_parts.append("embedding only")
        print(f"Model mode: {', '.join(mode_parts)}")

    trainer = PuzzleTrainer(model, lr=lr, device=device, use_amp=use_amp)

    # Train
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(train_examples, augment_factor=augment_factor, batch_size=batch_size)

        if verbose and (epoch + 1) % 50 == 0:
            train_loss, pixel_acc, size_acc = trainer.evaluate(train_examples)
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, pixel_acc={pixel_acc:.4f}, size_acc={size_acc:.2f}")

    # Final evaluation
    if verbose:
        train_loss, pixel_acc, size_acc = trainer.evaluate(train_examples)
        print(f"Final: loss={train_loss:.4f}, pixel_acc={pixel_acc:.4f}, size_acc={size_acc:.2f}")

    # Predict size first
    predicted_size = trainer.predict_size(test_input)
    if verbose:
        print(f"Predicted output size: {predicted_size[0]}x{predicted_size[1]}")

    # Predict test using predicted size (not given ground truth size)
    prediction = trainer.predict(test_input, output_size=None)

    return prediction, model, predicted_size


# =============================================================================
# Visualization
# =============================================================================

def grid_to_str(grid: torch.Tensor) -> List[str]:
    """Convert a grid to list of strings, one per row."""
    return [' '.join(str(int(c)) for c in row) for row in grid]


def visualize_prediction(
    test_input: torch.Tensor,
    prediction: torch.Tensor,
    test_output: Optional[torch.Tensor] = None,
    title: str = "ARC Puzzle Result"
):
    """
    Print grids side by side in the console.

    Args:
        test_input: [H, W] input grid
        prediction: [H, W] predicted output
        test_output: [H, W] ground truth (optional)
    """
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)

    # Convert grids to string rows
    input_rows = grid_to_str(test_input)
    pred_rows = grid_to_str(prediction)
    gt_rows = grid_to_str(test_output) if test_output is not None else None

    # Calculate widths for alignment
    input_width = max(len(r) for r in input_rows)
    pred_width = max(len(r) for r in pred_rows)

    # Headers
    if gt_rows is not None:
        gt_width = max(len(r) for r in gt_rows)
        print(f"{'Test Input':<{input_width}}   {'Prediction':<{pred_width}}   {'Ground Truth'}")
        print(f"{'-' * input_width}   {'-' * pred_width}   {'-' * gt_width}")
    else:
        print(f"{'Test Input':<{input_width}}   {'Prediction'}")
        print(f"{'-' * input_width}   {'-' * pred_width}")

    # Find max height
    max_rows = max(len(input_rows), len(pred_rows), len(gt_rows) if gt_rows else 0)

    # Print rows side by side
    for i in range(max_rows):
        inp = input_rows[i] if i < len(input_rows) else ''
        pred = pred_rows[i] if i < len(pred_rows) else ''
        if gt_rows is not None:
            gt = gt_rows[i] if i < len(gt_rows) else ''
            print(f"{inp:<{input_width}}   {pred:<{pred_width}}   {gt}")
        else:
            print(f"{inp:<{input_width}}   {pred}")

    print()


# =============================================================================
# Puzzle Loading
# =============================================================================

def load_arc_puzzle(puzzle_id: str, data_dir: str = 'kaggle/combined', arc_version: int = 1):
    """
    Load an ARC puzzle from JSON files.

    Returns:
        train_examples: List of (input_tensor, output_tensor) pairs
        test_input: Test input tensor
        test_output: Test output tensor (if available, for evaluation)
    """
    import json
    import os

    # Determine which files to search
    if arc_version == 1:
        challenge_files = ['arc-agi_training_challenges.json', 'arc-agi_evaluation_challenges.json']
        solution_files = ['arc-agi_training_solutions.json', 'arc-agi_evaluation_solutions.json']
    else:
        challenge_files = ['arc-agi_training_challenges.json', 'arc-agi_training2_challenges.json',
                          'arc-agi_evaluation_challenges.json', 'arc-agi_evaluation2_challenges.json']
        solution_files = ['arc-agi_training_solutions.json', 'arc-agi_training2_solutions.json',
                         'arc-agi_evaluation_solutions.json', 'arc-agi_evaluation2_solutions.json']

    # Find the puzzle
    puzzle_data = None
    test_solution = None

    for cf, sf in zip(challenge_files, solution_files):
        cf_path = os.path.join(data_dir, cf)
        if not os.path.exists(cf_path):
            continue
        with open(cf_path, 'r') as f:
            challenges = json.load(f)
        if puzzle_id in challenges:
            puzzle_data = challenges[puzzle_id]
            sf_path = os.path.join(data_dir, sf)
            if os.path.exists(sf_path):
                with open(sf_path, 'r') as f:
                    solutions = json.load(f)
                if puzzle_id in solutions:
                    test_solution = solutions[puzzle_id]
            break

    if puzzle_data is None:
        raise ValueError(f"Puzzle '{puzzle_id}' not found")

    # Convert to tensors
    train_examples = []
    for ex in puzzle_data['train']:
        inp = torch.tensor(ex['input'], dtype=torch.long)
        out = torch.tensor(ex['output'], dtype=torch.long)
        train_examples.append((inp, out))

    test_input = torch.tensor(puzzle_data['test'][0]['input'], dtype=torch.long)
    test_output = None
    if test_solution:
        test_output = torch.tensor(test_solution[0], dtype=torch.long)

    return train_examples, test_input, test_output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Per-puzzle ARC solver')
    parser.add_argument('puzzle_id', type=str, help='ARC puzzle ID (e.g., 23b5c85d)')
    parser.add_argument('--data-dir', type=str, default='kaggle/combined', help='Directory containing ARC JSON files')
    parser.add_argument('--arc-version', type=int, default=1, choices=[1, 2], help='ARC dataset version')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--num-slots', type=int, default=10, help='Number of slots')
    parser.add_argument('--slot-dim', type=int, default=48, help='Slot dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--augment-factor', type=int, default=100, help='Augmentations per example per epoch')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use-cnn', action='store_true', help='Use CNN encoder for local feature extraction')
    parser.add_argument('--cnn-layers', type=int, default=3, help='Number of CNN layers (if --use-cnn)')
    parser.add_argument('--color-slots', action='store_true', help='Use rigid color-based slots (one slot per color)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for GPU efficiency')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')
    args = parser.parse_args()

    # Load puzzle
    print(f"Loading puzzle: {args.puzzle_id}")
    train_examples, test_input, test_output = load_arc_puzzle(
        args.puzzle_id, args.data_dir, args.arc_version
    )

    print(f"Train examples: {len(train_examples)}")
    print(f"Test input shape: {test_input.shape}")
    if test_output is not None:
        print(f"Test output shape: {test_output.shape}")

    # Fit the model (size is predicted, not given)
    prediction, model, predicted_size = fit_puzzle(
        train_examples,
        test_input,
        num_epochs=args.epochs,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        lr=args.lr,
        augment_factor=args.augment_factor,
        device=args.device,
        use_cnn=args.use_cnn,
        cnn_layers=args.cnn_layers,
        color_slots=args.color_slots,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
    )

    print(f"\nPrediction shape: {prediction.shape}")

    if test_output is not None:
        # Check size prediction accuracy
        true_size = test_output.shape
        size_correct = (predicted_size[0] == true_size[0] and predicted_size[1] == true_size[1])
        print(f"True output size: {true_size[0]}x{true_size[1]}")
        print(f"Size prediction: {'CORRECT' if size_correct else 'WRONG'}")

        # Only compute pixel accuracy if sizes match
        if size_correct:
            accuracy = (prediction == test_output).float().mean().item()
            exact_match = (prediction == test_output).all().item()
            print(f"Pixel accuracy: {accuracy:.4f}")
            print(f"Exact match: {exact_match}")
        else:
            print("Cannot compute pixel accuracy - size mismatch")

    # Visualize results
    visualize_prediction(test_input, prediction, test_output, title=f"Puzzle: {args.puzzle_id}")
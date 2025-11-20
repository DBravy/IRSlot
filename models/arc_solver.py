"""
ARCSlotSolver: Complete end-to-end ARC puzzle solver using slot-based reasoning.

This module integrates all components:
- CNNEncoder: Grids → Features
- SlotAttentionEncoder: Features → Slots
- ARCSlotSequenceBuilder: Puzzle → Slot Sequence
- Transformer: Slot Sequence → Refined Slots
- SpatialBroadcastDecoder: Slots → Output Grid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from pydantic import BaseModel

from models.trm import CNNEncoder, SpatialBroadcastDecoder
from models.slot_encoder import SlotAttentionEncoder
from models.arc_slot_builder import ARCSlotSequenceBuilder
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin


class ARCSlotSolverConfig(BaseModel):
    """Configuration for ARCSlotSolver."""

    # Grid encoding
    grid_channels: int = 1
    cnn_hidden_dim: int = 64
    slot_dim: int = 64

    # Slot attention
    num_slots_per_grid: int = 7
    slot_iterations: int = 3
    slot_mlp_hidden: int = 128

    # Transformer
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    expansion: float = 4.0

    # Puzzle handling
    max_train_examples: int = 10
    max_grid_size: int = 30

    # Decoder
    decoder_hidden_dim: int = 64
    output_channels: int = 10  # 10 colors in ARC

    # Training
    forward_dtype: str = "float32"
    rms_norm_eps: float = 1e-5

    # Position encodings
    use_rope: bool = True
    rope_theta: float = 10000.0


class TransformerBlock(nn.Module):
    """Single transformer block for reasoning over slots."""

    def __init__(self, config: ARCSlotSolverConfig):
        super().__init__()
        self.config = config

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False  # Bidirectional attention
        )

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos_sin: Optional rotary embeddings
            attention_mask: Optional [batch, seq_len] mask (currently not used by Attention layer)

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Self attention with post-norm
        # Note: attention_mask is accepted but not passed to self_attn
        # as the Attention class doesn't support it yet
        attn_output = self.self_attn(
            cos_sin=cos_sin,
            hidden_states=hidden_states
        )
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)

        # MLP with post-norm
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)

        return hidden_states


class ARCSlotSolver(nn.Module):
    """
    Complete ARC puzzle solver using slot-based reasoning.

    Pipeline:
        1. Encode all grids (train in/out, test in) to slots
        2. Build unified slot sequence with special tokens
        3. Transformer reasoning over slot sequence
        4. Extract predicted slots from output
        5. Decode slots to output grid

    Args:
        config: ARCSlotSolverConfig
    """

    def __init__(self, config: ARCSlotSolverConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # 1. CNN Encoder: Grid → Features
        self.cnn_encoder = CNNEncoder(
            input_channels=config.grid_channels,
            hidden_dim=config.cnn_hidden_dim,
            slot_dim=config.slot_dim,
            forward_dtype=self.forward_dtype
        )

        # 2. Slot Encoder: Features → Slots
        self.slot_encoder = SlotAttentionEncoder(
            num_slots=config.num_slots_per_grid,
            slot_dim=config.slot_dim,
            hidden_size=config.hidden_size,
            num_iterations=config.slot_iterations,
            mlp_hidden_size=config.slot_mlp_hidden,
            max_spatial_size=config.max_grid_size,
            forward_dtype=self.forward_dtype
        )

        # 3. Sequence Builder: Puzzle → Slot Sequence
        self.sequence_builder = ARCSlotSequenceBuilder(
            cnn_encoder=self.cnn_encoder,
            slot_encoder=self.slot_encoder,
            num_slots_per_grid=config.num_slots_per_grid,
            hidden_size=config.hidden_size,
            max_train_examples=config.max_train_examples,
            puzzle_emb_dim=0,  # No puzzle embeddings for now
        )

        # 4. Transformer Reasoning Layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Position encodings
        if config.use_rope:
            # Estimate max sequence length
            max_seq_len = (
                config.max_train_examples * 2 * (1 + config.num_slots_per_grid) +  # train examples
                2 * (1 + config.num_slots_per_grid)  # test + predict
            )
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=max_seq_len,
                base=config.rope_theta
            )
        else:
            self.rotary_emb = None

        # 5. Spatial Broadcast Decoder: Slots → Grid
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=config.slot_dim,
            hidden_size=config.hidden_size,
            output_channels=config.output_channels,
            output_height=config.max_grid_size,
            output_width=config.max_grid_size,
            decoder_hidden_dim=config.decoder_hidden_dim,
            forward_dtype=self.forward_dtype
        )

    def forward(
        self,
        puzzle_batch: Dict[str, torch.Tensor],
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Solve ARC puzzles.

        Args:
            puzzle_batch: Dict from ARCPuzzleDataset collate function
            return_intermediate: If True, return intermediate representations

        Returns:
            Dict with:
                - predicted_grids: [batch, output_channels, H, W]
                - predicted_slots: [batch, num_slots, hidden_size]
                - (optional) intermediate states
        """
        # 1. Build slot sequence
        sequence_output = self.sequence_builder(puzzle_batch)

        sequence = sequence_output['sequence']  # [batch, seq_len, hidden_size]
        attention_mask = sequence_output['attention_mask']  # [batch, seq_len]
        predict_positions = sequence_output['predict_positions']  # [batch, 2]

        # 2. Transformer reasoning
        hidden_states = sequence.to(self.forward_dtype)

        # Get rotary embeddings if using RoPE
        # Slice to match actual sequence length
        if self.rotary_emb is not None:
            cos_full, sin_full = self.rotary_emb()
            seq_len = hidden_states.shape[1]
            cos_sin = (cos_full[:seq_len], sin_full[:seq_len])
        else:
            cos_sin = None

        # Apply transformer layers
        intermediate_states = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin, attention_mask=attention_mask)
            if return_intermediate:
                intermediate_states.append(hidden_states.clone())

        # 3. Extract predicted slots
        batch_size = hidden_states.shape[0]
        predicted_slots_list = []

        for b in range(batch_size):
            start, end = predict_positions[b]
            # Skip the PREDICT token, get only the slots
            predicted_slots = hidden_states[b, start+1:end]  # [num_slots, hidden_size]
            predicted_slots_list.append(predicted_slots)

        predicted_slots = torch.stack(predicted_slots_list, dim=0)  # [batch, num_slots, hidden_size]

        # 4. Decode slots to grid
        # Get target output size from first puzzle
        target_shapes = puzzle_batch['test_output_shapes']
        # Use first test output shape from first puzzle as target
        # (In practice, should handle variable sizes per puzzle)
        if target_shapes[0][0] is not None:
            target_h, target_w = target_shapes[0][0]
        else:
            # Default to test input size if output not available
            target_h, target_w = puzzle_batch['test_input_shapes'][0][0]

        # Decode all predictions with same target size for now
        predicted_grids = self.decoder(predicted_slots)  # [batch, output_channels, max_h, max_w]

        # Crop to target size
        predicted_grids = predicted_grids[:, :, :target_h, :target_w]

        result = {
            'predicted_grids': predicted_grids,
            'predicted_slots': predicted_slots,
            'full_sequence': hidden_states,
            'predict_positions': predict_positions,
        }

        if return_intermediate:
            result['intermediate_states'] = intermediate_states

        return result

    def compute_loss(
        self,
        puzzle_batch: Dict[str, torch.Tensor],
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            puzzle_batch: Dict from ARCPuzzleDataset collate function
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Dict with:
                - loss: Total loss
                - pixel_accuracy: Accuracy metric
        """
        # Forward pass
        output = self.forward(puzzle_batch)
        predicted_grids = output['predicted_grids']  # [batch, output_channels, H, W]

        # Get ground truth
        test_outputs = puzzle_batch['test_outputs']  # [batch, max_test, H, W]
        test_output_available = puzzle_batch['test_output_available']  # [batch, max_test]

        # Use first test output
        targets = test_outputs[:, 0]  # [batch, H, W]
        available = test_output_available[:, 0]  # [batch]

        # Get target size
        batch_size, _, H, W = predicted_grids.shape

        # Ensure targets match prediction size (crop or pad)
        if targets.shape[1:] != (H, W):
            targets_resized = torch.zeros(batch_size, H, W, dtype=targets.dtype, device=targets.device)
            for b in range(batch_size):
                h, w = min(targets.shape[1], H), min(targets.shape[2], W)
                targets_resized[b, :h, :w] = targets[b, :h, :w]
            targets = targets_resized

        # Cross-entropy loss
        # predicted_grids: [batch, num_colors, H, W]
        # targets: [batch, H, W] with values 0-9

        loss_per_pixel = F.cross_entropy(
            predicted_grids,  # [batch, num_colors, H, W]
            targets,  # [batch, H, W]
            reduction='none'  # [batch, H, W]
        )

        # Mask out invalid examples
        loss_per_pixel = loss_per_pixel * available.view(-1, 1, 1).float()

        if reduction == 'mean':
            loss = loss_per_pixel.sum() / (available.sum() * H * W + 1e-8)
        elif reduction == 'sum':
            loss = loss_per_pixel.sum()
        else:
            loss = loss_per_pixel

        # Compute accuracy
        with torch.no_grad():
            predictions = predicted_grids.argmax(dim=1)  # [batch, H, W]
            correct = (predictions == targets).float()
            correct = correct * available.view(-1, 1, 1).float()
            pixel_accuracy = correct.sum() / (available.sum() * H * W + 1e-8)

        return {
            'loss': loss,
            'pixel_accuracy': pixel_accuracy,
        }

    @torch.no_grad()
    def predict(
        self,
        puzzle_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Predict output grids for puzzles.

        Args:
            puzzle_batch: Dict from ARCPuzzleDataset collate function

        Returns:
            predictions: [batch, H, W] with integer color predictions (0-9)
        """
        self.eval()

        output = self.forward(puzzle_batch)
        predicted_grids = output['predicted_grids']  # [batch, output_channels, H, W]

        # Convert to integer predictions
        predictions = predicted_grids.argmax(dim=1)  # [batch, H, W]

        return predictions

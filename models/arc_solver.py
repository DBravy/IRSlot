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
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import Counter
from pydantic import BaseModel

from models.trm import CNNEncoder, SpatialBroadcastDecoder
from models.slot_encoder import SlotAttentionEncoder
from models.arc_slot_builder import ARCSlotSequenceBuilder
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin
from memory_bank import MemoryBank
from loss import InfoNCELoss


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

    # Contrastive learning (multi-task)
    use_contrastive_loss: bool = False
    contrastive_embedding_dim: int = 128
    contrastive_temperature: float = 0.07
    contrastive_num_negatives: int = 512
    contrastive_loss_weight: float = 1.0


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

        # 6. Contrastive Learning Components (Optional Multi-Task Learning)
        self.use_contrastive = config.use_contrastive_loss
        if self.use_contrastive:
            # Projection head: Pool slots → embedding for contrastive learning
            # NOTE: Slots are already projected to hidden_size by slot_encoder,
            # so we project from hidden_size (not slot_dim) to embedding_dim
            self.contrastive_projection = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.contrastive_embedding_dim)
            )
            self.contrastive_criterion = InfoNCELoss(temperature=config.contrastive_temperature)
            self.contrastive_loss_weight = config.contrastive_loss_weight
            self.contrastive_num_negatives = config.contrastive_num_negatives
        else:
            self.contrastive_projection = None
            self.contrastive_criterion = None

    def forward(
        self,
        puzzle_batch: Dict[str, torch.Tensor],
        return_intermediate: bool = False,
        return_test_input_slots: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Solve ARC puzzles.

        Args:
            puzzle_batch: Dict from ARCPuzzleDataset collate function
            return_intermediate: If True, return intermediate representations
            return_test_input_slots: If True, return raw slots from test input (for contrastive learning)

        Returns:
            Dict with:
                - predicted_grids: [batch, output_channels, H, W]
                - predicted_slots: [batch, num_slots, hidden_size]
                - test_input_slots: [batch, num_slots, slot_dim] (only if return_test_input_slots=True)
                - (optional) intermediate states
        """
        # 1. Build slot sequence
        sequence_output = self.sequence_builder(puzzle_batch)

        sequence = sequence_output['sequence']  # [batch, seq_len, hidden_size]
        attention_mask = sequence_output['attention_mask']  # [batch, seq_len]
        predict_positions = sequence_output['predict_positions']  # [batch, 2]

        # 1b. Extract test input slots for contrastive learning (if needed)
        test_input_slots_raw = None
        if return_test_input_slots and self.use_contrastive:
            # Manually encode test input to get raw slots (before projection to hidden_size)
            test_inputs = puzzle_batch['test_inputs'][:, 0]  # [batch, H, W] - first test input
            batch_size, H, W = test_inputs.shape

            # Encode with CNN
            test_features = self.cnn_encoder(test_inputs)  # [batch, H*W, slot_dim]

            # Get raw slots (before projection to hidden_size)
            # We need to access the slot encoder's internals before projection
            # For now, we'll use the slot encoder normally and then project back
            # This is a workaround - ideally we'd modify SlotAttentionEncoder to return both
            test_slots_hidden = self.slot_encoder(test_features, spatial_size=(H, W))  # [batch, num_slots, hidden_size]

            # Since slots are already projected to hidden_size, we'll work with those
            # The contrastive projection will map from hidden_size to embedding_dim
            test_input_slots_raw = test_slots_hidden

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

        if return_test_input_slots and test_input_slots_raw is not None:
            result['test_input_slots'] = test_input_slots_raw

        return result

    def compute_loss(
        self,
        puzzle_batch: Dict[str, torch.Tensor],
        memory_bank: Optional[MemoryBank] = None,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss (with optional multi-task learning).

        Args:
            puzzle_batch: Dict from ARCPuzzleDataset collate function
            memory_bank: Optional MemoryBank for contrastive learning
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Dict with:
                - loss: Total loss (ARC + optional contrastive)
                - arc_loss: ARC puzzle solving loss
                - contrastive_loss: Contrastive learning loss (if enabled)
                - pixel_accuracy: Accuracy metric
                - contrastive_accuracy: Contrastive accuracy (if enabled)
        """
        # Forward pass
        return_slots = self.use_contrastive and memory_bank is not None
        output = self.forward(puzzle_batch, return_test_input_slots=return_slots)
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

        # ============================================
        # 1. ARC Puzzle Solving Loss (Cross-Entropy)
        # ============================================
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
            arc_loss = loss_per_pixel.sum() / (available.sum() * H * W + 1e-8)
        elif reduction == 'sum':
            arc_loss = loss_per_pixel.sum()
        else:
            arc_loss = loss_per_pixel

        # Compute pixel accuracy
        with torch.no_grad():
            predictions = predicted_grids.argmax(dim=1)  # [batch, H, W]
            correct = (predictions == targets).float()
            correct = correct * available.view(-1, 1, 1).float()
            pixel_accuracy = correct.sum() / (available.sum() * H * W + 1e-8)

        # ============================================
        # 2. Contrastive Learning Loss (Optional)
        # ============================================
        contrastive_loss = torch.tensor(0.0, device=arc_loss.device)
        contrastive_accuracy = torch.tensor(0.0, device=arc_loss.device)

        if self.use_contrastive and memory_bank is not None and 'test_input_slots' in output:
            # Get test input slots
            test_slots = output['test_input_slots']  # [batch, num_slots, hidden_size]

            # Pool slots (mean across slots)
            pooled_slots = test_slots.mean(dim=1)  # [batch, hidden_size]

            # Project to embedding space
            embeddings = self.contrastive_projection(pooled_slots)  # [batch, embedding_dim]
            embeddings = F.normalize(embeddings, dim=1)

            # Get puzzle indices (integer IDs for memory bank)
            # Each puzzle's test input is a unique instance
            puzzle_indices = puzzle_batch['puzzle_indices']  # [batch] - integer tensor

            # Get stored embeddings from memory bank
            stored_embeddings = memory_bank.get(puzzle_indices)

            # Sample negative embeddings
            negative_embeddings = memory_bank.sample_negatives(
                num_negatives=self.contrastive_num_negatives,
                exclude_ids=puzzle_indices
            )

            # Compute contrastive loss
            contrastive_loss, contrastive_metrics = self.contrastive_criterion(
                embeddings, stored_embeddings, negative_embeddings
            )

            contrastive_accuracy = torch.tensor(contrastive_metrics['accuracy'], device=arc_loss.device)

            # Update memory bank (no gradients)
            with torch.no_grad():
                memory_bank.update(puzzle_indices, embeddings.detach())

        # ============================================
        # 3. Combine Losses (Multi-Task Learning)
        # ============================================
        if self.use_contrastive and memory_bank is not None:
            total_loss = arc_loss + self.contrastive_loss_weight * contrastive_loss
        else:
            total_loss = arc_loss

        return {
            'loss': total_loss,
            'arc_loss': arc_loss,
            'contrastive_loss': contrastive_loss,
            'pixel_accuracy': pixel_accuracy,
            'contrastive_accuracy': contrastive_accuracy,
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

    @torch.no_grad()
    def predict_with_tta(
        self,
        puzzle_dict: Dict,
        num_augmentations: int = 100,
        apply_dihedral: bool = True,
        apply_color_permutation: bool = True,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Predict with test-time augmentation and majority voting.

        Args:
            puzzle_dict: Single puzzle dict from ARCPuzzleDataset.__getitem__
            num_augmentations: Number of augmented versions to run
            apply_dihedral: Whether to apply dihedral transforms
            apply_color_permutation: Whether to apply color permutations

        Returns:
            (voted_prediction, all_predictions): The majority-voted grid and list of all predictions
        """
        from augmentation import PuzzleAugmentation
        from dataset.arc_puzzle_dataset import collate_puzzle_batch

        self.eval()
        device = next(self.parameters()).device

        augmenter = PuzzleAugmentation(
            apply_dihedral=apply_dihedral,
            apply_color_permutation=apply_color_permutation
        )

        # Extract grids from puzzle dict
        train_inputs = [g.numpy() if isinstance(g, torch.Tensor) else g
                        for g in puzzle_dict['train_inputs']]
        train_outputs = [g.numpy() if isinstance(g, torch.Tensor) else g
                         for g in puzzle_dict['train_outputs']]
        test_inputs = [g.numpy() if isinstance(g, torch.Tensor) else g
                       for g in puzzle_dict['test_inputs']]
        test_outputs = [g.numpy() if isinstance(g, torch.Tensor) else g
                        for g in puzzle_dict['test_outputs'] if g is not None]

        # Get original output shape for reference
        original_shape = test_outputs[0].shape if test_outputs else test_inputs[0].shape

        all_predictions = []

        for _ in range(num_augmentations):
            # Apply augmentation to entire puzzle
            aug_train_in, aug_train_out, aug_test_in, params = augmenter.augment_puzzle(
                train_inputs, train_outputs, test_inputs
            )

            # Also augment test outputs to get correct target shape
            aug_test_out = [augmenter.apply_to_grid(g, params) for g in test_outputs] if test_outputs else []

            # Create augmented puzzle dict
            aug_puzzle = {
                'puzzle_id': puzzle_dict['puzzle_id'],
                'train_inputs': [torch.from_numpy(g) for g in aug_train_in],
                'train_outputs': [torch.from_numpy(g) for g in aug_train_out],
                'test_inputs': [torch.from_numpy(g) for g in aug_test_in],
                'test_outputs': [torch.from_numpy(g) for g in aug_test_out] if aug_test_out else [None],
                'num_train': puzzle_dict['num_train'],
                'num_test': puzzle_dict['num_test'],
            }

            # Collate into batch of 1
            batch = collate_puzzle_batch([aug_puzzle])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Get prediction
            pred = self.predict(batch)[0].cpu().numpy()  # [H, W]

            # Invert augmentation on prediction
            pred_inverted = augmenter.invert_grid(pred, params)

            # Crop to original shape (in case of padding differences)
            h, w = original_shape
            pred_inverted = pred_inverted[:h, :w]

            all_predictions.append(pred_inverted)

        # Majority voting (pixel-wise)
        voted = self._majority_vote(all_predictions)

        return voted, all_predictions

    def _majority_vote(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Pixel-wise majority voting over predictions.

        Args:
            predictions: List of [H, W] arrays with integer predictions

        Returns:
            voted: [H, W] array with majority-voted prediction
        """
        # Stack predictions: [N, H, W]
        stacked = np.stack(predictions, axis=0)
        H, W = stacked.shape[1], stacked.shape[2]

        voted = np.zeros((H, W), dtype=np.uint8)

        for i in range(H):
            for j in range(W):
                pixel_values = stacked[:, i, j]
                counter = Counter(pixel_values)
                voted[i, j] = counter.most_common(1)[0][0]

        return voted

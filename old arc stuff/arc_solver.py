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

from models.trm import (
    CNNEncoder, 
    SpatialBroadcastDecoder,
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Carry
)
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

    # TRM Recursive Reasoning (replaces simple transformer)
    hidden_size: int = 256
    num_heads: int = 8
    expansion: float = 4.0
    H_cycles: int = 2  # High-level reasoning cycles
    L_cycles: int = 2  # Low-level reasoning cycles
    L_layers: int = 4  # Number of transformer layers in L-level
    mlp_t: bool = False  # Use MLP instead of transformer in L-level
    
    # ACT (Adaptive Computation Time) for dynamic halting
    halt_max_steps: int = 1  # Set to 1 to disable ACT, >1 to enable
    halt_exploration_prob: float = 0.1
    no_ACT_continue: bool = True  # Use sigmoid halt instead of Q-learning

    # Puzzle handling
    max_train_examples: int = 10
    max_grid_size: int = 30
    puzzle_emb_ndim: int = 0  # Puzzle embeddings disabled by default
    num_puzzle_identifiers: int = 1000  # Max number of unique puzzles

    # Decoder
    decoder_hidden_dim: int = 64
    output_channels: int = 10  # 10 colors in ARC

    # Training
    forward_dtype: str = "float32"
    rms_norm_eps: float = 1e-5

    # Position encodings
    pos_encodings: str = "rope"  # "rope", "learned", or "none"
    rope_theta: float = 10000.0

    # Contrastive learning (multi-task)
    use_contrastive_loss: bool = False
    contrastive_embedding_dim: int = 128
    contrastive_temperature: float = 0.07
    contrastive_num_negatives: int = 512
    contrastive_loss_weight: float = 1.0




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
            forward_dtype=self.forward_dtype,
            num_colors=config.output_channels  # Use output_channels (10 for ARC colors)
        )

        # 2. Slot Encoder: Features → Slots
        self.slot_encoder = SlotAttentionEncoder(
            num_slots=config.num_slots_per_grid,
            slot_dim=config.slot_dim,
            feature_dim=config.slot_dim,
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

        # 4. TRM Recursive Reasoning Module
        # Estimate max sequence length for the slot sequence
        max_seq_len = (
            config.max_train_examples * 2 * (1 + config.num_slots_per_grid) +  # train examples
            2 * (1 + config.num_slots_per_grid)  # test + predict
        )
        
        # Create TRM config - TRM will handle the reasoning over the slot sequence
        trm_config_dict = {
            'batch_size': 32,  # Will be overridden dynamically
            'seq_len': max_seq_len,
            'puzzle_emb_ndim': config.puzzle_emb_ndim,
            'num_puzzle_identifiers': config.num_puzzle_identifiers,
            'vocab_size': 10,  # Not used in our case (we work with slots)
            'H_cycles': config.H_cycles,
            'L_cycles': config.L_cycles,
            'H_layers': 1,  # Not used, kept for compatibility
            'L_layers': config.L_layers,
            'hidden_size': config.hidden_size,
            'expansion': config.expansion,
            'num_heads': config.num_heads,
            'pos_encodings': config.pos_encodings,
            'rms_norm_eps': config.rms_norm_eps,
            'rope_theta': config.rope_theta,
            'halt_max_steps': config.halt_max_steps,
            'halt_exploration_prob': config.halt_exploration_prob,
            'forward_dtype': config.forward_dtype,
            'mlp_t': config.mlp_t,
            'puzzle_emb_len': 0,  # Auto-calculate
            'no_ACT_continue': config.no_ACT_continue,
            'use_slot_attention': False,  # We handle slot encoding externally
        }
        
        self.trm = TinyRecursiveReasoningModel_ACTV1(trm_config_dict)

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
        carry: Optional[TinyRecursiveReasoningModel_ACTV1Carry] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Solve ARC puzzles using TRM's recursive reasoning.

        Args:
            puzzle_batch: Dict from ARCPuzzleDataset collate function
            return_intermediate: If True, return intermediate representations
            carry: Optional TRM carry state (for multi-step reasoning with ACT)

        Returns:
            Dict with:
                - predicted_grids: [batch, output_channels, H, W]
                - predicted_slots: [batch, num_slots, hidden_size]
                - carry: TRM carry state (for ACT)
                - q_halt_logits: Halting Q-values (if ACT enabled)
                - (optional) intermediate states
        """
        # 1. Build slot sequence
        sequence_output = self.sequence_builder(puzzle_batch)

        sequence = sequence_output['sequence']  # [batch, seq_len, hidden_size]
        attention_mask = sequence_output['attention_mask'].to(sequence.device)  # [batch, seq_len]
        predict_positions = sequence_output['predict_positions']  # [batch, 2]

        # 2. TRM Recursive Reasoning
        batch_size, seq_len, hidden_size = sequence.shape
        hidden_states = sequence.to(self.forward_dtype)
        slot_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = hidden_states * slot_mask

        # Initialize TRM inner carry if not provided
        if carry is None:
            # Create initial carry state
            inner_carry = self.trm.inner.empty_carry(batch_size)
            inner_carry = self.trm.inner.reset_carry(
                torch.ones(batch_size, dtype=torch.bool, device=hidden_states.device),
                inner_carry
            )
        else:
            inner_carry = carry.inner_carry

        # Get position encodings
        if hasattr(self.trm.inner, 'rotary_emb'):
            cos_sin = self.trm.inner.rotary_emb()
            # Slice to match sequence length
            if seq_len < cos_sin[0].shape[0]:
                cos_sin = (cos_sin[0][:seq_len], cos_sin[1][:seq_len])
        else:
            cos_sin = None

        seq_info = dict(cos_sin=cos_sin)

        # Apply TRM recursive reasoning (H-cycles and L-cycles)
        # We treat the slot sequence as the input embedding (bypassing TRM's embedding layer)
        z_H, z_L = inner_carry.z_H, inner_carry.z_L
        
        # Ensure z_H and z_L match the sequence dimensions
        if z_H.shape[1] != seq_len:
            # Resize carry states to match sequence length
            z_H = F.interpolate(
                z_H.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            z_L = F.interpolate(
                z_L.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        slot_mask = slot_mask.to(z_H.dtype)
        z_H = z_H * slot_mask
        z_L = z_L * slot_mask

        intermediate_states = []
        
        # Recursive reasoning with gradient control (like TRM)
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.trm.inner.L_level(z_L, z_H + hidden_states, **seq_info)
                    z_L = z_L * slot_mask
                z_H = self.trm.inner.L_level(z_H, z_L, **seq_info)
                z_H = z_H * slot_mask
                if return_intermediate:
                    intermediate_states.append(z_H.clone())
        
        # Final H-cycle with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.trm.inner.L_level(z_L, z_H + hidden_states, **seq_info)
            z_L = z_L * slot_mask
        z_H = self.trm.inner.L_level(z_H, z_L, **seq_info)
        z_H = z_H * slot_mask
        
        if return_intermediate:
            intermediate_states.append(z_H.clone())

        # Store final hidden states
        hidden_states = z_H * slot_mask

        # Compute Q-values for halting (if ACT enabled)
        q_halt_logits = None
        if self.config.halt_max_steps > 1:
            # Use first position for Q-head (like TRM)
            q_logits = self.trm.inner.q_head(z_H[:, 0]).to(torch.float32)
            q_halt_logits = q_logits[..., 0]  # Halt Q-value

        # 3. Extract predicted slots
        predicted_slots_list = []
        for b in range(batch_size):
            start, end = predict_positions[b]
            # Skip the PREDICT token, get only the slots
            predicted_slots = hidden_states[b, start+1:end]  # [num_slots, hidden_size]
            predicted_slots_list.append(predicted_slots)

        predicted_slots = torch.stack(predicted_slots_list, dim=0)  # [batch, num_slots, hidden_size]

        # 4. Decode slots to grid
        target_shapes = puzzle_batch['test_output_shapes']
        predicted_grids = self.decoder(predicted_slots)  # [batch, output_channels, max_h, max_w]

        # Prepare result
        result = {
            'predicted_grids': predicted_grids,
            'predicted_slots': predicted_slots,
            'full_sequence': hidden_states,
            'predict_positions': predict_positions,
        }

        # Store carry state for ACT
        new_inner_carry = self.trm.inner.__class__.__bases__[0].__name__  # Get carry class name
        from models.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry
        result['carry'] = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), 
            z_L=z_L.detach()
        )

        if q_halt_logits is not None:
            result['q_halt_logits'] = q_halt_logits

        if return_intermediate:
            result['intermediate_states'] = intermediate_states

        return result

    def compute_loss(
        self,
        puzzle_batch: Dict[str, torch.Tensor],
        memory_bank: Optional[MemoryBank] = None,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ARC puzzle solving loss.

        NOTE: Contrastive learning is now handled separately via
        compute_contrastive_loss_on_grids() with interleaved training.

        Args:
            puzzle_batch: Dict from ARCPuzzleDataset collate function
            memory_bank: Unused (kept for backward compatibility)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Dict with:
                - loss: ARC puzzle solving loss
                - arc_loss: Same as loss
                - pixel_accuracy: Accuracy metric
                - contrastive_loss: Always 0 (for backward compatibility)
                - contrastive_accuracy: Always 0 (for backward compatibility)
                - avg_positive_sim: Always 0 (for backward compatibility)
                - avg_negative_sim: Always 0 (for backward compatibility)
        """
        # Forward pass
        output = self.forward(puzzle_batch)
        predicted_grids = output['predicted_grids']  # [batch, output_channels, H, W]

        # Get ground truth
        test_outputs = puzzle_batch['test_outputs']  # [batch, max_test, H, W]
        test_output_available = puzzle_batch['test_output_available']  # [batch, max_test]

        # Use first test output
        targets = test_outputs[:, 0]  # [batch, max_h, max_w]
        available = test_output_available[:, 0]  # [batch]

        batch_size, _, H, W = predicted_grids.shape

        # Build per-example spatial masks so loss is computed only on valid pixels
        spatial_mask = torch.zeros(batch_size, H, W, dtype=torch.bool, device=predicted_grids.device)
        target_shapes = puzzle_batch['test_output_shapes']
        input_shapes = puzzle_batch['test_input_shapes']
        targets_resized = torch.zeros(batch_size, H, W, dtype=targets.dtype, device=targets.device)
        for b in range(batch_size):
            target_shape = None
            if b < len(target_shapes) and len(target_shapes[b]) > 0:
                target_shape = target_shapes[b][0]
            if target_shape is None and b < len(input_shapes) and len(input_shapes[b]) > 0:
                target_shape = input_shapes[b][0]
            if target_shape is None:
                target_shape = (H, W)
            h, w = target_shape
            spatial_mask[b, :h, :w] = True
            targets_resized[b, :h, :w] = targets[b, :h, :w]
        targets = targets_resized

        valid_pixels = spatial_mask & available.view(-1, 1, 1)
        valid_pixel_count = valid_pixels.sum().clamp_min(1)

        # ARC Puzzle Solving Loss (Cross-Entropy)
        # predicted_grids: [batch, num_colors, H, W]
        # targets: [batch, H, W] with values 0-9

        loss_per_pixel = F.cross_entropy(
            predicted_grids,  # [batch, num_colors, H, W]
            targets,  # [batch, H, W]
            reduction='none'  # [batch, H, W]
        )

        # Mask out invalid pixels
        weighted_loss = loss_per_pixel * valid_pixels.float()

        if reduction == 'mean':
            arc_loss = weighted_loss.sum() / valid_pixel_count.float()
        elif reduction == 'sum':
            arc_loss = weighted_loss.sum()
        else:
            arc_loss = weighted_loss

        # Compute pixel accuracy
        with torch.no_grad():
            predictions = predicted_grids.argmax(dim=1)  # [batch, H, W]
            correct = ((predictions == targets) & valid_pixels).float()
            pixel_accuracy = correct.sum() / valid_pixel_count.float()

        # Return loss (contrastive fields are 0 for backward compatibility)
        return {
            'loss': arc_loss,
            'arc_loss': arc_loss,
            'pixel_accuracy': pixel_accuracy,
            'contrastive_loss': torch.tensor(0.0, device=arc_loss.device),
            'contrastive_accuracy': torch.tensor(0.0, device=arc_loss.device),
            'avg_positive_sim': torch.tensor(0.0, device=arc_loss.device),
            'avg_negative_sim': torch.tensor(0.0, device=arc_loss.device),
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

    def compute_contrastive_loss_on_grids(
        self,
        grid_batch: Dict[str, torch.Tensor],
        memory_bank: MemoryBank
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss on individual grids (for instance recognition).

        This is a separate, simpler forward pass that:
        1. Encodes individual grids through CNN + slot encoder
        2. Pools slots to get grid-level embeddings
        3. Computes contrastive loss against memory bank

        Args:
            grid_batch: Dict from GridDataset collate function with:
                - grids: [batch, H, W]
                - puzzle_indices: [batch]
                - shapes: List of (H, W) tuples
            memory_bank: MemoryBank for negative sampling

        Returns:
            Dict with:
                - loss: Contrastive loss
                - accuracy: Contrastive accuracy
                - avg_positive_sim: Average positive similarity
                - avg_negative_sim: Average negative similarity
        """
        grids = grid_batch['grids']  # [batch, H, W] with long dtype
        puzzle_indices = grid_batch['puzzle_indices']  # [batch]
        shapes = grid_batch['shapes']  # List of (H, W) tuples

        batch_size, H, W = grids.shape
        device = grids.device

        # 1. Encode grids through CNN
        features = self.cnn_encoder(grids)  # [batch, H*W, slot_dim]

        # 2. Apply slot attention
        slots = self.slot_encoder(features, spatial_size=(H, W))  # [batch, num_slots, hidden_size]

        # 3. Pool slots (mean across slots)
        pooled_slots = slots.mean(dim=1)  # [batch, hidden_size]

        # 4. Project to embedding space
        embeddings = self.contrastive_projection(pooled_slots)  # [batch, embedding_dim]
        embeddings = F.normalize(embeddings, dim=1)

        # 5. Get stored embeddings from memory bank
        stored_embeddings = memory_bank.get(puzzle_indices)

        # 6. Sample negative embeddings
        negative_embeddings = memory_bank.sample_negatives(
            num_negatives=self.contrastive_num_negatives,
            exclude_ids=puzzle_indices
        )

        # 7. Compute contrastive loss
        contrastive_loss, metrics = self.contrastive_criterion(
            embeddings, stored_embeddings, negative_embeddings
        )

        # 8. Update memory bank (no gradients)
        with torch.no_grad():
            memory_bank.update(puzzle_indices, embeddings.detach())

        return {
            'loss': contrastive_loss,
            'accuracy': torch.tensor(metrics['accuracy'], device=device),
            'avg_positive_sim': torch.tensor(metrics['avg_positive_sim'], device=device),
            'avg_negative_sim': torch.tensor(metrics['avg_negative_sim'], device=device),
        }

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

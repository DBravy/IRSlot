"""
ARCSlotSequenceBuilder: Converts ARC puzzles to slot sequences for transformer reasoning.

This module bridges the gap between puzzle data and transformer input, converting
multiple grids (train examples + test input) into a unified sequence of object slots
with special structural tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class ARCSlotSequenceBuilder(nn.Module):
    """
    Builds slot sequences from ARC puzzles for transformer reasoning.

    Architecture:
        For each grid in puzzle:
            Grid [H, W]
                → CNNEncoder
                → Features [H*W, slot_dim]
                → SlotAttentionEncoder
                → Slots [num_slots, hidden_size]

        Combine into sequence:
            [PUZZLE_EMB_1...K]
            [TRAIN_0_IN_TOKEN] [slot_1] ... [slot_N]
            [TRAIN_0_OUT_TOKEN] [slot_1] ... [slot_N]
            ...
            [TEST_IN_TOKEN] [slot_1] ... [slot_N]
            [PREDICT_TOKEN] [slot_1] ... [slot_N]  # To be predicted

    Args:
        cnn_encoder: CNNEncoder instance (converts grids to features)
        slot_encoder: SlotAttentionEncoder instance (converts features to slots)
        num_slots_per_grid: Number of slots to extract per grid
        hidden_size: Transformer hidden dimension
        max_train_examples: Maximum number of train examples to handle
        puzzle_emb_dim: Dimension for puzzle identifier embeddings (0 = no puzzle emb)
        num_puzzles: Number of unique puzzles (for puzzle embeddings)
    """

    def __init__(
        self,
        cnn_encoder: nn.Module,
        slot_encoder: nn.Module,
        num_slots_per_grid: int,
        hidden_size: int,
        max_train_examples: int = 10,
        puzzle_emb_dim: int = 0,
        num_puzzles: int = 1000,
    ):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.slot_encoder = slot_encoder
        self.num_slots_per_grid = num_slots_per_grid
        self.hidden_size = hidden_size
        self.max_train_examples = max_train_examples
        self.puzzle_emb_dim = puzzle_emb_dim

        # Puzzle embeddings (optional)
        if puzzle_emb_dim > 0:
            self.puzzle_embedding = nn.Embedding(num_puzzles, puzzle_emb_dim)
            # Project to hidden_size
            self.puzzle_emb_proj = nn.Linear(puzzle_emb_dim, hidden_size)
            # Number of puzzle embedding tokens
            self.num_puzzle_tokens = math.ceil(puzzle_emb_dim / hidden_size)
        else:
            self.num_puzzle_tokens = 0

        # Special token embeddings for structure
        # These tokens mark the beginning of each grid's slots
        self.train_input_tokens = nn.Parameter(
            torch.randn(max_train_examples, hidden_size) * 0.02
        )
        self.train_output_tokens = nn.Parameter(
            torch.randn(max_train_examples, hidden_size) * 0.02
        )
        self.test_input_token = nn.Parameter(
            torch.randn(1, hidden_size) * 0.02
        )
        self.predict_token = nn.Parameter(
            torch.randn(1, hidden_size) * 0.02
        )

    def _encode_grid_batch(
        self,
        grids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of grids to slots.

        Args:
            grids: [batch, num_grids, H, W] - Batch of grids (may be padded)
            mask: [batch, num_grids] - Boolean mask indicating valid grids

        Returns:
            slots: [batch, num_grids, num_slots, hidden_size]
            valid_mask: [batch, num_grids] - Mask for valid slots
        """
        batch_size, num_grids, H, W = grids.shape

        # Reshape to process all grids in batch
        grids_flat = grids.reshape(batch_size * num_grids, H, W)

        # CNN encode: [B*N, H, W] -> [B*N, H*W, slot_dim]
        features = self.cnn_encoder(grids_flat)

        # Extract slots: [B*N, H*W, slot_dim] -> [B*N, num_slots, hidden_size]
        slots = self.slot_encoder(features, spatial_size=(H, W))

        # Reshape back: [B*N, num_slots, hidden_size] -> [B, N, num_slots, hidden_size]
        slots = slots.reshape(batch_size, num_grids, self.num_slots_per_grid, self.hidden_size)

        # Use provided mask or assume all valid
        if mask is None:
            valid_mask = torch.ones(batch_size, num_grids, dtype=torch.bool, device=grids.device)
        else:
            valid_mask = mask

        return slots, valid_mask

    def forward(
        self,
        puzzle_batch: Dict[str, torch.Tensor],
        puzzle_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a batch of puzzles to slot sequences.

        Args:
            puzzle_batch: Dict from ARCPuzzleDataset collate function with keys:
                - train_inputs: [B, max_train, H, W]
                - train_outputs: [B, max_train, H, W]
                - test_inputs: [B, max_test, H, W]
                - train_mask: [B, max_train]
                - test_mask: [B, max_test]
                - (other metadata...)
            puzzle_ids: Optional [B] tensor of puzzle IDs for embeddings

        Returns:
            Dict with:
                - sequence: [B, total_seq_len, hidden_size] - Full slot sequence
                - attention_mask: [B, total_seq_len] - Valid positions
                - predict_positions: [B, 2] - (start, end) indices for prediction region
                - num_train: [B] - Number of train examples per puzzle
        """
        batch_size = puzzle_batch['train_inputs'].shape[0]
        device = puzzle_batch['train_inputs'].device

        # Extract data
        train_inputs = puzzle_batch['train_inputs']  # [B, max_train, H, W]
        train_outputs = puzzle_batch['train_outputs']
        test_inputs = puzzle_batch['test_inputs']  # [B, max_test, H, W]
        train_mask = puzzle_batch['train_mask']  # [B, max_train]
        test_mask = puzzle_batch['test_mask']  # [B, max_test]
        num_train = puzzle_batch['num_train']  # [B]

        # For now, only use first test example (can be extended later)
        test_inputs_first = test_inputs[:, 0]  # [B, H, W]
        test_inputs_first = test_inputs_first.unsqueeze(1)  # [B, 1, H, W]

        # Encode all grids to slots
        train_in_slots, train_in_valid = self._encode_grid_batch(train_inputs, train_mask)
        train_out_slots, train_out_valid = self._encode_grid_batch(train_outputs, train_mask)
        test_in_slots, _ = self._encode_grid_batch(test_inputs_first)

        # Build sequence for each puzzle in batch
        sequences = []
        attention_masks = []
        predict_positions = []

        for b in range(batch_size):
            seq_parts = []

            # 1. Puzzle embeddings (optional)
            if self.puzzle_emb_dim > 0 and puzzle_ids is not None:
                puzzle_emb = self.puzzle_embedding(puzzle_ids[b])  # [puzzle_emb_dim]
                puzzle_emb = self.puzzle_emb_proj(puzzle_emb)  # [hidden_size]
                seq_parts.append(puzzle_emb.unsqueeze(0))  # [1, hidden_size]

            # 2. Train examples
            n_train = num_train[b].item()
            for i in range(n_train):
                # Train input marker + slots
                train_in_marker = self.train_input_tokens[i].unsqueeze(0)  # [1, hidden_size]
                train_in_slot_seq = train_in_slots[b, i]  # [num_slots, hidden_size]
                seq_parts.append(train_in_marker)
                seq_parts.append(train_in_slot_seq)

                # Train output marker + slots
                train_out_marker = self.train_output_tokens[i].unsqueeze(0)
                train_out_slot_seq = train_out_slots[b, i]
                seq_parts.append(train_out_marker)
                seq_parts.append(train_out_slot_seq)

            # 3. Test input
            test_in_marker = self.test_input_token  # [1, hidden_size]
            test_in_slot_seq = test_in_slots[b, 0]  # [num_slots, hidden_size]
            seq_parts.append(test_in_marker)
            seq_parts.append(test_in_slot_seq)

            # 4. Prediction region (placeholder slots to be predicted)
            predict_marker = self.predict_token  # [1, hidden_size]
            # Initialize prediction slots to zeros (will be filled by transformer)
            predict_slots = torch.zeros(
                self.num_slots_per_grid, self.hidden_size,
                dtype=train_in_slots.dtype, device=device
            )

            # Track prediction region positions
            predict_start = sum(p.shape[0] for p in seq_parts)
            seq_parts.append(predict_marker)
            seq_parts.append(predict_slots)
            predict_end = sum(p.shape[0] for p in seq_parts)

            # Concatenate all parts
            sequence = torch.cat(seq_parts, dim=0)  # [total_seq_len, hidden_size]

            # Create attention mask (all positions valid)
            attention_mask = torch.ones(sequence.shape[0], dtype=torch.bool, device=device)

            sequences.append(sequence)
            attention_masks.append(attention_mask)
            predict_positions.append(torch.tensor([predict_start, predict_end], device=device))

        # Pad sequences to same length
        max_seq_len = max(seq.shape[0] for seq in sequences)

        padded_sequences = []
        padded_masks = []

        for seq, mask in zip(sequences, attention_masks):
            seq_len = seq.shape[0]
            if seq_len < max_seq_len:
                # Pad sequence
                padding = torch.zeros(
                    max_seq_len - seq_len, self.hidden_size,
                    dtype=seq.dtype, device=device
                )
                seq = torch.cat([seq, padding], dim=0)

                # Pad mask
                mask_padding = torch.zeros(
                    max_seq_len - seq_len, dtype=torch.bool, device=device
                )
                mask = torch.cat([mask, mask_padding], dim=0)

            padded_sequences.append(seq)
            padded_masks.append(mask)

        # Stack into batch
        sequence_batch = torch.stack(padded_sequences, dim=0)  # [B, max_seq_len, hidden_size]
        attention_mask_batch = torch.stack(padded_masks, dim=0)  # [B, max_seq_len]
        predict_positions_batch = torch.stack(predict_positions, dim=0)  # [B, 2]

        return {
            'sequence': sequence_batch,
            'attention_mask': attention_mask_batch,
            'predict_positions': predict_positions_batch,
            'num_train': num_train,
            'train_input_shapes': puzzle_batch['train_input_shapes'],
            'train_output_shapes': puzzle_batch['train_output_shapes'],
            'test_input_shapes': puzzle_batch['test_input_shapes'],
            'test_output_shapes': puzzle_batch['test_output_shapes'],
        }

    def get_sequence_structure(self, puzzle_batch: Dict[str, torch.Tensor]) -> Dict[str, list]:
        """
        Utility to understand the structure of the generated sequence.

        Returns a breakdown of which positions correspond to which components.
        Useful for debugging and visualization.
        """
        num_train = puzzle_batch['num_train'][0].item()  # Use first puzzle as example

        structure = {
            'puzzle_emb': [],
            'train_examples': [],
            'test_input': [],
            'prediction': []
        }

        pos = 0

        # Puzzle embeddings
        if self.puzzle_emb_dim > 0:
            structure['puzzle_emb'] = list(range(pos, pos + self.num_puzzle_tokens))
            pos += self.num_puzzle_tokens

        # Train examples
        for i in range(num_train):
            train_in_start = pos
            pos += 1  # marker
            pos += self.num_slots_per_grid  # slots
            train_in_end = pos

            train_out_start = pos
            pos += 1  # marker
            pos += self.num_slots_per_grid  # slots
            train_out_end = pos

            structure['train_examples'].append({
                'input': (train_in_start, train_in_end),
                'output': (train_out_start, train_out_end)
            })

        # Test input
        test_start = pos
        pos += 1  # marker
        pos += self.num_slots_per_grid  # slots
        test_end = pos
        structure['test_input'] = (test_start, test_end)

        # Prediction
        predict_start = pos
        pos += 1  # marker
        pos += self.num_slots_per_grid  # slots
        predict_end = pos
        structure['prediction'] = (predict_start, predict_end)

        structure['total_length'] = pos

        return structure

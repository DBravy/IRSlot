"""
GridDataset for instance recognition / contrastive learning.

Extracts individual grids from ARC puzzles for slot attention training.
Each grid is treated as a separate instance for contrastive learning.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

from dataset.common import dihedral_transform


class GridDataset(Dataset):
    """
    Dataset that extracts individual grids from ARC puzzles.

    Each puzzle contributes multiple grids (train inputs, train outputs, test inputs).
    Each grid is a separate instance for contrastive learning.

    Args:
        data_dir: Directory containing ARC JSON files
        split: 'train' or 'eval'
        arc_version: 'agi1' or 'agi2'
        augment: Whether to apply augmentation
        subset_size: Limit number of puzzles (for debugging)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        arc_version: str = 'agi1',
        augment: bool = True,
        subset_size: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.arc_version = arc_version
        self.augment = augment

        # Map split to directory name
        split_dir_map = {
            'train': 'arc-agi_training_challenges.json' if arc_version == 'agi1' else 'arc-agi_training_challenges_v2.json',
            'eval': 'arc-agi_evaluation_challenges.json' if arc_version == 'agi1' else 'arc-agi_evaluation_challenges_v2.json',
        }

        json_file = self.data_dir / split_dir_map[split]

        if not json_file.exists():
            raise FileNotFoundError(f"ARC data file not found: {json_file}")

        # Load all puzzles
        with open(json_file, 'r') as f:
            self.puzzles = json.load(f)

        if subset_size is not None:
            puzzle_ids = list(self.puzzles.keys())[:subset_size]
            self.puzzles = {pid: self.puzzles[pid] for pid in puzzle_ids}

        # Extract all individual grids with their puzzle IDs
        # Each grid gets a UNIQUE index for instance recognition
        self.grids = []  # List of (grid, puzzle_id, grid_idx)
        self.puzzle_id_to_idx = {}  # Map puzzle_id -> integer index (for reference)

        grid_idx = 0  # Unique index for each individual grid
        puzzle_idx = 0  # Index for each unique puzzle (for reference only)

        for puzzle_id, puzzle_data in self.puzzles.items():
            if puzzle_id not in self.puzzle_id_to_idx:
                self.puzzle_id_to_idx[puzzle_id] = puzzle_idx
                puzzle_idx += 1

            # Extract grids from train examples
            for train_example in puzzle_data['train']:
                # Add input grid with unique grid_idx
                self.grids.append((
                    np.array(train_example['input'], dtype=np.uint8),
                    puzzle_id,
                    grid_idx
                ))
                grid_idx += 1

                # Add output grid with unique grid_idx
                self.grids.append((
                    np.array(train_example['output'], dtype=np.uint8),
                    puzzle_id,
                    grid_idx
                ))
                grid_idx += 1

            # Extract grids from test examples
            for test_example in puzzle_data['test']:
                # Add test input with unique grid_idx
                self.grids.append((
                    np.array(test_example['input'], dtype=np.uint8),
                    puzzle_id,
                    grid_idx
                ))
                grid_idx += 1
                # Note: We don't add test outputs (usually not available during training)

        print(f"GridDataset ({split}): Loaded {len(self.grids)} individual grids from {len(self.puzzles)} puzzles")

    def __len__(self):
        return len(self.grids)

    def num_unique_grids(self):
        """Return the total number of unique grids (same as __len__)."""
        return len(self.grids)

    def num_unique_puzzles(self):
        """Return the number of unique puzzles."""
        return len(self.puzzle_id_to_idx)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single grid instance.

        Returns:
            Dict with:
                - grid: [H, W] tensor (long dtype for color indices)
                - puzzle_id: String puzzle identifier
                - puzzle_idx: Integer grid index (UNIQUE per grid, for memory bank)
        """
        grid, puzzle_id, grid_idx = self.grids[idx]

        # Apply augmentation if enabled
        if self.augment:
            # Dihedral transform (rotation + reflection)
            trans_id = np.random.randint(0, 8)
            grid = dihedral_transform(grid, trans_id)

            # Color permutation (preserve background=0)
            color_perm = np.arange(10, dtype=np.uint8)
            color_perm[1:] = np.random.permutation(np.arange(1, 10, dtype=np.uint8))
            grid = color_perm[grid]

        return {
            'grid': torch.from_numpy(grid).long(),
            'puzzle_id': puzzle_id,
            'puzzle_idx': grid_idx,  # Now using unique grid_idx
        }


def collate_grid_batch(batch: List[Dict]) -> Dict:
    """
    Collate grids into a batch.

    Pads grids to the same size within the batch.

    Args:
        batch: List of dicts from GridDataset.__getitem__

    Returns:
        Dict with:
            - grids: [batch, max_H, max_W] padded tensor
            - puzzle_ids: List of puzzle ID strings
            - puzzle_indices: [batch] tensor of unique grid indices (for memory bank)
            - shapes: List of (H, W) tuples for original shapes
    """
    grids = [item['grid'] for item in batch]
    puzzle_ids = [item['puzzle_id'] for item in batch]
    puzzle_indices = torch.tensor([item['puzzle_idx'] for item in batch], dtype=torch.long)

    # Get shapes
    shapes = [grid.shape for grid in grids]
    max_h = max(h for h, w in shapes)
    max_w = max(w for h, w in shapes)

    # Pad grids
    batch_size = len(grids)
    padded_grids = torch.zeros(batch_size, max_h, max_w, dtype=torch.long)

    for i, grid in enumerate(grids):
        h, w = grid.shape
        padded_grids[i, :h, :w] = grid

    return {
        'grids': padded_grids,
        'puzzle_ids': puzzle_ids,
        'puzzle_indices': puzzle_indices,  # Now contains unique grid indices
        'shapes': shapes,
    }

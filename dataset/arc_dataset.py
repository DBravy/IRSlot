"""
Dataset loader for ARC grids for instance recognition training.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from augmentation import ARCGridAugmentation


class ARCInstanceDataset(Dataset):
    """
    Dataset for ARC instance recognition training.

    Each item returns:
    - grid_id: Unique identifier for this grid
    - grid: Augmented view of the grid [H, W]
    - original_shape: (H, W) before padding
    """
    def __init__(self, data_dir, split='train', subset='all', augment=True, max_grid_size=30, max_puzzles=None):
        """
        Args:
            data_dir: Root directory containing the processed dataset
            split: 'train' or 'test'
            subset: 'all' (default subset name from build_arc_dataset.py)
            augment: Whether to apply augmentations
            max_grid_size: Maximum grid dimension (default 30 for ARC)
            max_puzzles: Maximum number of unique puzzles to load (None = all)
        """
        self.data_dir = data_dir
        self.split = split
        self.subset = subset
        self.max_grid_size = max_grid_size

        # Load metadata
        metadata_path = os.path.join(data_dir, split, 'dataset.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load data
        subset_path = os.path.join(data_dir, split, f"{subset}__")
        self.inputs = np.load(f"{subset_path}inputs.npy")
        self.labels = np.load(f"{subset_path}labels.npy")
        self.puzzle_identifiers = np.load(f"{subset_path}puzzle_identifiers.npy")
        self.puzzle_indices = np.load(f"{subset_path}puzzle_indices.npy")

        # Limit to max_puzzles if specified
        if max_puzzles is not None and max_puzzles < len(self.puzzle_identifiers):
            print(f"Limiting dataset to {max_puzzles} puzzles (out of {len(self.puzzle_identifiers)} available)")

            # Keep only the first max_puzzles
            old_puzzle_ids = self.puzzle_identifiers[:max_puzzles]

            # Create a mapping from old puzzle IDs to new sequential IDs (0, 1, 2, ..., max_puzzles-1)
            self.id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_puzzle_ids)}

            # Update puzzle_identifiers to be sequential
            self.puzzle_identifiers = np.arange(max_puzzles, dtype=self.puzzle_identifiers.dtype)

            # Find the index range that corresponds to these puzzles
            # puzzle_indices[i] is the starting index for puzzle i
            # We want all examples from puzzle 0 to puzzle max_puzzles-1
            if max_puzzles < len(self.puzzle_indices):
                end_idx = self.puzzle_indices[max_puzzles]
            else:
                end_idx = len(self.inputs)

            # Slice inputs and labels
            self.inputs = self.inputs[:end_idx]
            self.labels = self.labels[:end_idx]

            # Update puzzle_indices
            self.puzzle_indices = self.puzzle_indices[:max_puzzles + 1]
        else:
            # No limiting, no remapping needed
            self.id_mapping = None

        # Setup augmentation
        self.augment = augment
        if augment:
            self.augmentation = ARCGridAugmentation(
                apply_rotation=True,
                apply_color_permutation=True
            )
        else:
            self.augmentation = None

        print(f"Loaded {split}/{subset} dataset:")
        print(f"  Total examples: {len(self.inputs)}")
        print(f"  Total unique grids: {len(self.puzzle_identifiers)}")

    def __len__(self):
        """Return number of examples in the dataset."""
        return len(self.inputs)

    def _unflatten_grid(self, flat_grid):
        """
        Convert flattened grid back to 2D format.

        The grid is stored as a flattened 30x30 array with:
        - 0: padding
        - 1: end-of-sequence marker
        - 2-11: colors 0-9

        Returns:
            grid: [H, W] array with values 0-9
            original_shape: (H, W) tuple
        """
        # Reshape to 2D
        grid = flat_grid.reshape(self.max_grid_size, self.max_grid_size)

        # Find actual grid bounds (marked by EOS = 1)
        eos_positions = np.where(grid == 1)
        if len(eos_positions[0]) > 0:
            H = eos_positions[0].min() if eos_positions[0].min() > 0 else self.max_grid_size
            W = eos_positions[1].min() if eos_positions[1].min() > 0 else self.max_grid_size
        else:
            # No EOS found, use full grid
            H = W = self.max_grid_size

        # Extract actual grid and convert from encoding (2-11) to colors (0-9)
        grid = grid[:H, :W]
        grid = np.clip(grid - 2, 0, 9).astype(np.uint8)

        return grid, (H, W)

    def __getitem__(self, idx):
        """
        Get a single training example.

        Returns:
            dict with keys:
            - grid_id: int - Unique grid identifier
            - grid: torch.Tensor [H, W] - Augmented grid
            - original_shape: tuple (H, W) - Shape before padding
        """
        # Get flattened grid
        flat_grid = self.inputs[idx]

        # Unflatten to 2D
        grid, original_shape = self._unflatten_grid(flat_grid)

        # Get grid ID (puzzle identifier)
        # Find which puzzle this example belongs to
        puzzle_idx = np.searchsorted(self.puzzle_indices[1:], idx, side='right')
        grid_id = self.puzzle_identifiers[puzzle_idx]

        # Apply augmentation
        if self.augmentation is not None:
            grid = self.augmentation(grid)

        # Convert to tensor
        grid = torch.from_numpy(grid).long()

        return {
            'grid_id': grid_id,
            'grid': grid,
            'original_shape': original_shape
        }


def collate_fn_pad(batch):
    """
    Collate function that pads grids to the same size within a batch.

    Args:
        batch: List of dicts from dataset __getitem__

    Returns:
        dict with batched tensors:
        - grid_ids: [B] - Grid identifiers
        - grids: [B, H, W] - Padded grids
        - original_shapes: list of (H, W) tuples
    """
    grid_ids = torch.tensor([item['grid_id'] for item in batch])
    original_shapes = [item['original_shape'] for item in batch]

    # Find max dimensions in batch based on actual augmented grid shapes
    max_H = max(item['grid'].shape[0] for item in batch)
    max_W = max(item['grid'].shape[1] for item in batch)

    # Pad all grids to max size
    padded_grids = []
    for item in batch:
        grid = item['grid']
        H, W = grid.shape

        # Pad with zeros (black background)
        if H < max_H or W < max_W:
            padded = torch.zeros(max_H, max_W, dtype=grid.dtype)
            padded[:H, :W] = grid
            padded_grids.append(padded)
        else:
            padded_grids.append(grid)

    grids = torch.stack(padded_grids, dim=0)

    return {
        'grid_ids': grid_ids,
        'grids': grids,
        'original_shapes': original_shapes
    }

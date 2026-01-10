"""
Dataset loader for ARC grids for instance recognition training.
"""
import os
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from augmentation import ARCGridAugmentation
from dataset.common import dihedral_transform
from mask_supervision import extract_object_masks


# Constants for grid encoding
ARCMaxGridSize = 30
PuzzleIdSeparator = "|||"


def grid_hash(grid: np.ndarray):
    """Hash a grid for duplicate checking."""
    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def generate_augmentation():
    """Generate a random augmentation (dihedral transform + color permutation)."""
    trans_id = np.random.randint(0, 8)
    # Permute colors, excluding "0" (black)
    mapping = np.concatenate([
        np.arange(0, 1, dtype=np.uint8),
        np.random.permutation(np.arange(1, 10, dtype=np.uint8))
    ])
    return trans_id, mapping


def apply_augmentation(grid: np.ndarray, trans_id: int, color_mapping: np.ndarray):
    """Apply augmentation to a grid."""
    return dihedral_transform(color_mapping[grid], trans_id)


class ARCInstanceDataset(Dataset):
    """
    Dataset for ARC instance recognition training.

    Each item returns:
    - grid_id: Unique identifier for this grid
    - grid: Augmented view of the grid [H, W]
    - original_shape: (H, W) before padding
    """
    def __init__(self, data_dir, split='train', subset='all', augment=True, max_grid_size=30,
                 max_puzzles=None, puzzle_filter=None, arc_version=None, num_augmentations=200,
                 raw_data_dir='kaggle/combined', grid_type='both', num_slots=7,
                 return_masks=False):
        """
        Args:
            data_dir: Root directory containing the processed dataset (used when puzzle_filter is None)
            split: 'train' or 'test'
            subset: 'all' (default subset name from build_arc_dataset.py)
            augment: Whether to apply runtime augmentations
            max_grid_size: Maximum grid dimension (default 30 for ARC)
            max_puzzles: Maximum number of unique puzzles to load (None = all)
            puzzle_filter: If specified, only load grids from this puzzle (loads from raw JSON)
            arc_version: ARC version (1 or 2) - required when puzzle_filter is specified
            num_augmentations: Number of augmentations to generate when using puzzle_filter
            raw_data_dir: Directory containing raw ARC JSON files
            grid_type: Which grids to use for single puzzle mode: 'input', 'output', or 'both' (default)
            num_slots: Number of slots for mask generation (default 7)
            return_masks: Whether to return ground truth object masks (default False)
        """
        self.data_dir = data_dir
        self.split = split
        self.subset = subset
        self.max_grid_size = max_grid_size
        self.puzzle_filter = puzzle_filter
        self.grid_type = grid_type
        self.num_slots = num_slots
        self.return_masks = return_masks

        if puzzle_filter is not None:
            # Load from raw JSON and generate augmentations on-the-fly
            self._load_from_raw_json(puzzle_filter, arc_version, num_augmentations, raw_data_dir, grid_type)
        else:
            # Load from preprocessed dataset
            self._load_from_preprocessed(data_dir, split, subset, max_puzzles)

        # Setup runtime augmentation
        self.augment = augment
        if augment:
            self.augmentation = ARCGridAugmentation(
                apply_rotation=True,
                apply_color_permutation=False
            )
        else:
            self.augmentation = None

        print(f"Loaded dataset:")
        print(f"  Total examples: {len(self.inputs)}")
        print(f"  Total unique grids: {len(self.puzzle_identifiers)}")

    def _load_from_raw_json(self, puzzle_id, arc_version, num_augmentations, raw_data_dir, grid_type='both'):
        """Load a specific puzzle from raw JSON and generate augmentations.

        Args:
            puzzle_id: The puzzle ID to load
            arc_version: ARC version (1 or 2)
            num_augmentations: Number of augmentations to generate
            raw_data_dir: Directory containing raw ARC JSON files
            grid_type: Which grids to use: 'input', 'output', or 'both' (default)
        """
        if arc_version is None:
            raise ValueError("arc_version must be specified when using puzzle_filter")

        if grid_type not in ('input', 'output', 'both'):
            raise ValueError(f"grid_type must be 'input', 'output', or 'both', got '{grid_type}'")

        grid_type_desc = {'input': 'input grids only', 'output': 'output grids only', 'both': 'input and output grids'}[grid_type]
        print(f"Loading puzzle '{puzzle_id}' from raw JSON (ARC version {arc_version})")
        print(f"  Grid type: {grid_type_desc}")
        print(f"  Generating {num_augmentations} augmentations...")

        # Determine which JSON files to search
        if arc_version == 1:
            challenge_files = [
                os.path.join(raw_data_dir, 'arc-agi_training_challenges.json'),
                os.path.join(raw_data_dir, 'arc-agi_evaluation_challenges.json'),
            ]
            solution_files = [
                os.path.join(raw_data_dir, 'arc-agi_training_solutions.json'),
                os.path.join(raw_data_dir, 'arc-agi_evaluation_solutions.json'),
            ]
        elif arc_version == 2:
            challenge_files = [
                os.path.join(raw_data_dir, 'arc-agi_training_challenges.json'),
                os.path.join(raw_data_dir, 'arc-agi_training2_challenges.json'),
                os.path.join(raw_data_dir, 'arc-agi_evaluation_challenges.json'),
                os.path.join(raw_data_dir, 'arc-agi_evaluation2_challenges.json'),
            ]
            solution_files = [
                os.path.join(raw_data_dir, 'arc-agi_training_solutions.json'),
                os.path.join(raw_data_dir, 'arc-agi_training2_solutions.json'),
                os.path.join(raw_data_dir, 'arc-agi_evaluation_solutions.json'),
                os.path.join(raw_data_dir, 'arc-agi_evaluation2_solutions.json'),
            ]
        else:
            raise ValueError(f"arc_version must be 1 or 2, got {arc_version}")

        # Find the puzzle in the JSON files
        puzzle_data = None
        solutions = None
        for challenge_file, solution_file in zip(challenge_files, solution_files):
            if not os.path.exists(challenge_file):
                continue
            with open(challenge_file, 'r') as f:
                challenges = json.load(f)
            if puzzle_id in challenges:
                puzzle_data = challenges[puzzle_id]
                # Try to load solutions
                if os.path.exists(solution_file):
                    with open(solution_file, 'r') as f:
                        all_solutions = json.load(f)
                    if puzzle_id in all_solutions:
                        solutions = all_solutions[puzzle_id]
                break

        if puzzle_data is None:
            raise ValueError(f"Puzzle '{puzzle_id}' not found in ARC version {arc_version} files")

        # Extract grids from the puzzle based on grid_type
        base_grids = []
        for example in puzzle_data.get('train', []):
            if grid_type in ('input', 'both'):
                base_grids.append(np.array(example['input'], dtype=np.uint8))
            if grid_type in ('output', 'both'):
                base_grids.append(np.array(example['output'], dtype=np.uint8))
        for i, example in enumerate(puzzle_data.get('test', [])):
            if grid_type in ('input', 'both'):
                base_grids.append(np.array(example['input'], dtype=np.uint8))
            # Add test output if we have solutions
            if grid_type in ('output', 'both') and solutions and i < len(solutions):
                base_grids.append(np.array(solutions[i], dtype=np.uint8))

        print(f"  Found {len(base_grids)} base grids in puzzle")

        # Generate augmentations
        all_grids = []  # List of (grid, puzzle_variant_id)
        seen_hashes = set()

        # Add base puzzle grids (variant 0)
        for grid in base_grids:
            h = grid_hash(grid)
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_grids.append((grid, 0))

        # Generate augmented variants
        num_variants = 1  # Already have base
        max_attempts = num_augmentations * 5  # Allow retries for duplicates

        for _ in range(max_attempts):
            if num_variants >= num_augmentations + 1:
                break

            trans_id, color_mapping = generate_augmentation()

            # Check if this augmentation produces a new variant
            aug_grids = [apply_augmentation(g, trans_id, color_mapping) for g in base_grids]
            variant_hash = hashlib.sha256(
                '|'.join(grid_hash(g) for g in aug_grids).encode()
            ).hexdigest()

            if variant_hash not in seen_hashes:
                seen_hashes.add(variant_hash)
                for grid in aug_grids:
                    all_grids.append((grid, num_variants))
                num_variants += 1

        print(f"  Generated {num_variants} puzzle variants")
        print(f"  Total grids: {len(all_grids)}")

        # Convert to the format expected by the rest of the class
        # Each grid gets its own unique ID (not grouped by variant)
        self.inputs = []
        self.labels = []
        grid_ids = []

        for grid, variant_id in all_grids:
            # Encode grid: pad to max_grid_size, add 2 to colors
            encoded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.uint8)
            h, w = grid.shape
            encoded[:h, :w] = grid + 2  # Shift colors by 2 (0=pad, 1=eos, 2-11=colors)
            self.inputs.append(encoded.flatten())
            self.labels.append(encoded.flatten())  # Same as input for this task
            grid_ids.append(len(grid_ids))  # Each grid gets unique ID

        self.inputs = np.array(self.inputs, dtype=np.uint8)
        self.labels = np.array(self.labels, dtype=np.uint8)

        num_unique_grids = len(grid_ids)

        # Create puzzle_identifiers (one per unique grid)
        self.puzzle_identifiers = np.arange(num_unique_grids, dtype=np.int32)

        # Create puzzle_indices (for compatibility) - each grid is its own "puzzle"
        self.puzzle_indices = np.arange(num_unique_grids + 1, dtype=np.int32)

        # Store grid-to-puzzle mapping (each grid maps to itself)
        self._grid_puzzle_ids = np.array(grid_ids, dtype=np.int32)

        self.metadata = {
            'puzzle_filter': puzzle_id,
            'arc_version': arc_version,
            'grid_type': grid_type,
            'num_variants': num_variants,
            'num_base_grids': len(base_grids),
            'num_unique_grids': num_unique_grids
        }

    def _load_from_preprocessed(self, data_dir, split, subset, max_puzzles):
        """Load from preprocessed numpy files."""
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
        self._grid_puzzle_ids = None  # Will use puzzle_indices for lookup

        # Limit to max_puzzles if specified
        if max_puzzles is not None and max_puzzles < len(self.puzzle_identifiers):
            print(f"Limiting dataset to {max_puzzles} puzzles (out of {len(self.puzzle_identifiers)} available)")

            # Keep only the first max_puzzles
            old_puzzle_ids = self.puzzle_identifiers[:max_puzzles]

            # Create a mapping from old puzzle IDs to new sequential IDs
            self.id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_puzzle_ids)}

            # Update puzzle_identifiers to be sequential
            self.puzzle_identifiers = np.arange(max_puzzles, dtype=self.puzzle_identifiers.dtype)

            # Find the index range that corresponds to these puzzles
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
            self.id_mapping = None

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

        # Find actual grid content (values >= 2)
        content_mask = grid >= 2
        if content_mask.any():
            # Find bounding box of content
            rows_with_content = np.where(content_mask.any(axis=1))[0]
            cols_with_content = np.where(content_mask.any(axis=0))[0]

            min_row = rows_with_content.min()
            max_row = rows_with_content.max() + 1
            min_col = cols_with_content.min()
            max_col = cols_with_content.max() + 1

            # Extract content
            grid = grid[min_row:max_row, min_col:max_col]
        else:
            # No content, return empty 1x1 grid
            grid = np.array([[0]], dtype=np.uint8)

        # Convert from encoding (2-11) to colors (0-9)
        grid = grid.astype(np.int16) - 2
        grid = np.clip(grid, 0, 9).astype(np.uint8)

        return grid, grid.shape

    def __getitem__(self, idx):
        """
        Get a single training example.

        Returns:
            dict with keys:
            - grid_id: int - Unique grid identifier
            - grid: torch.Tensor [H, W] - Augmented grid
            - original_shape: tuple (H, W) - Shape before padding
            - masks: torch.Tensor [num_slots, H, W] - Object masks (if return_masks=True)
            - num_objects: int - Number of foreground objects (if return_masks=True)
        """
        # Get flattened grid
        flat_grid = self.inputs[idx]

        # Unflatten to 2D
        grid, original_shape = self._unflatten_grid(flat_grid)

        # Get grid ID (0-indexed puzzle index for memory bank compatibility)
        if self._grid_puzzle_ids is not None:
            # Using direct mapping from raw JSON loading
            grid_id = self._grid_puzzle_ids[idx]
        else:
            # Using puzzle_indices for preprocessed data
            # Use puzzle_idx directly (0-indexed) instead of puzzle_identifiers value
            # to ensure grid IDs are valid memory bank indices
            grid_id = np.searchsorted(self.puzzle_indices[1:], idx, side='right')

        # Apply runtime augmentation
        if self.augmentation is not None:
            grid = self.augmentation(grid)

        # Generate ground truth masks if requested
        if self.return_masks:
            masks, num_objects = extract_object_masks(grid, self.num_slots)
            masks = torch.from_numpy(masks).float()
        else:
            masks = None
            num_objects = 0

        # Convert to tensor (copy needed for negative strides from flips)
        grid = torch.from_numpy(grid.copy()).long()

        result = {
            'grid_id': grid_id,
            'grid': grid,
            'original_shape': original_shape
        }

        if self.return_masks:
            result['masks'] = masks
            result['num_objects'] = num_objects

        return result


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
        - masks: [B, num_slots, H, W] - Padded masks (if present in batch)
        - num_objects: [B] - Number of objects per sample (if masks present)
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

    result = {
        'grid_ids': grid_ids,
        'grids': grids,
        'original_shapes': original_shapes
    }

    # Handle masks if present
    if 'masks' in batch[0] and batch[0]['masks'] is not None:
        num_slots = batch[0]['masks'].shape[0]
        padded_masks = []

        for item in batch:
            masks = item['masks']  # [num_slots, H, W]
            H, W = masks.shape[1], masks.shape[2]

            # Pad masks to max size
            if H < max_H or W < max_W:
                padded = torch.zeros(num_slots, max_H, max_W, dtype=masks.dtype)
                padded[:, :H, :W] = masks
                padded_masks.append(padded)
            else:
                padded_masks.append(masks)

        result['masks'] = torch.stack(padded_masks, dim=0)
        result['num_objects'] = torch.tensor([item['num_objects'] for item in batch])

    return result

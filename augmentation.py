"""
Augmentation utilities for ARC grids.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataset.common import dihedral_transform, inverse_dihedral_transform


class AugmentationParams:
    """Tracks augmentation parameters for inversion."""
    def __init__(self, dihedral_id: int = 0, color_perm: Optional[np.ndarray] = None):
        self.dihedral_id = dihedral_id
        self.color_perm = color_perm  # Forward permutation array

        # Precompute inverse color permutation
        if color_perm is not None:
            self.color_perm_inv = np.zeros(10, dtype=np.uint8)
            for i, c in enumerate(color_perm):
                self.color_perm_inv[c] = i
        else:
            self.color_perm_inv = None


class ARCGridAugmentation:
    """
    Augmentation pipeline for ARC grids.

    Applies:
    1. Dihedral group transformations (rotations + reflections)
    2. Color permutations (excluding black background)
    """
    def __init__(self, apply_rotation=True, apply_color_permutation=True):
        self.apply_rotation = apply_rotation
        self.apply_color_permutation = apply_color_permutation

    def __call__(self, grid):
        """
        Apply augmentations to a grid.

        Args:
            grid: np.ndarray of shape [H, W] with values 0-9
                  OR torch.Tensor of shape [H, W]

        Returns:
            Augmented grid (same type and shape as input)
        """
        is_torch = isinstance(grid, torch.Tensor)
        if is_torch:
            device = grid.device
            dtype = grid.dtype
            grid = grid.cpu().numpy()

        grid = grid.astype(np.uint8)

        # Apply dihedral transformation
        if self.apply_rotation:
            trans_id = np.random.randint(0, 8)
            grid = dihedral_transform(grid, trans_id)

        # Apply color permutation (excluding 0 = black background)
        if self.apply_color_permutation:
            # Create permutation of colors 1-9, keep 0 fixed
            perm = np.arange(10, dtype=np.uint8)
            perm[1:] = np.random.permutation(np.arange(1, 10, dtype=np.uint8))
            grid = perm[grid]

        if is_torch:
            grid = torch.from_numpy(grid).to(device=device, dtype=dtype)

        return grid


class MultiViewAugmentation:
    """
    Creates multiple augmented views of the same grid.

    This is useful for contrastive learning where you want to create
    positive pairs from the same grid.
    """
    def __init__(self, n_views=2):
        self.n_views = n_views
        self.augmentation = ARCGridAugmentation(
            apply_rotation=True,
            apply_color_permutation=True
        )

    def __call__(self, grid):
        """
        Args:
            grid: Input grid [H, W]

        Returns:
            List of n_views augmented grids
        """
        return [self.augmentation(grid) for _ in range(self.n_views)]


class PuzzleAugmentation:
    """
    Applies consistent augmentation to an entire ARC puzzle (all grids).
    Tracks parameters for inverting the output prediction.
    """

    def __init__(self, apply_dihedral: bool = True, apply_color_permutation: bool = True):
        self.apply_dihedral = apply_dihedral
        self.apply_color_permutation = apply_color_permutation

    def sample_params(self) -> AugmentationParams:
        """Sample random augmentation parameters."""
        dihedral_id = np.random.randint(0, 8) if self.apply_dihedral else 0

        if self.apply_color_permutation:
            perm = np.arange(10, dtype=np.uint8)
            perm[1:] = np.random.permutation(np.arange(1, 10, dtype=np.uint8))
        else:
            perm = None

        return AugmentationParams(dihedral_id=dihedral_id, color_perm=perm)

    def apply_to_grid(self, grid: np.ndarray, params: AugmentationParams) -> np.ndarray:
        """Apply augmentation to a single grid."""
        grid = grid.astype(np.uint8)

        # Dihedral transform
        if params.dihedral_id != 0:
            grid = dihedral_transform(grid, params.dihedral_id)

        # Color permutation
        if params.color_perm is not None:
            grid = params.color_perm[grid]

        return grid

    def invert_grid(self, grid: np.ndarray, params: AugmentationParams) -> np.ndarray:
        """Invert augmentation on a grid (for predictions)."""
        grid = grid.astype(np.uint8)

        # Invert color permutation first (reverse order of application)
        if params.color_perm_inv is not None:
            grid = params.color_perm_inv[grid]

        # Invert dihedral transform
        if params.dihedral_id != 0:
            grid = inverse_dihedral_transform(grid, params.dihedral_id)

        return grid

    def augment_puzzle(
        self,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        test_inputs: List[np.ndarray],
        params: Optional[AugmentationParams] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], AugmentationParams]:
        """
        Apply consistent augmentation to entire puzzle.

        Returns:
            (aug_train_inputs, aug_train_outputs, aug_test_inputs, params)
        """
        if params is None:
            params = self.sample_params()

        aug_train_in = [self.apply_to_grid(g, params) for g in train_inputs]
        aug_train_out = [self.apply_to_grid(g, params) for g in train_outputs]
        aug_test_in = [self.apply_to_grid(g, params) for g in test_inputs]

        return aug_train_in, aug_train_out, aug_test_in, params

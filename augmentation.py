"""
Augmentation utilities for ARC grids.
"""
import torch
import numpy as np
from dataset.common import dihedral_transform


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

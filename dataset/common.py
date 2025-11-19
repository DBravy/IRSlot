"""
Common data structures and utilities for ARC dataset processing.
"""
import numpy as np
from pydantic import BaseModel
from typing import List


class PuzzleDatasetMetadata(BaseModel):
    """Metadata for the ARC puzzle dataset."""
    seq_len: int
    vocab_size: int
    pad_id: int
    ignore_label_id: int
    blank_identifier_id: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]


def dihedral_transform(grid: np.ndarray, trans_id: int) -> np.ndarray:
    """
    Apply a dihedral group transformation to a 2D grid.

    Dihedral group D4 has 8 elements:
    - 4 rotations (0°, 90°, 180°, 270°)
    - 4 reflections (horizontal, vertical, and 2 diagonal flips)

    Args:
        grid: 2D numpy array
        trans_id: Integer from 0-7 indicating which transformation to apply

    Returns:
        Transformed grid
    """
    if trans_id == 0:
        return grid
    elif trans_id == 1:
        return np.rot90(grid, k=1)
    elif trans_id == 2:
        return np.rot90(grid, k=2)
    elif trans_id == 3:
        return np.rot90(grid, k=3)
    elif trans_id == 4:
        return np.flip(grid, axis=0)  # Vertical flip
    elif trans_id == 5:
        return np.flip(np.rot90(grid, k=1), axis=0)
    elif trans_id == 6:
        return np.flip(grid, axis=1)  # Horizontal flip
    elif trans_id == 7:
        return np.flip(np.rot90(grid, k=3), axis=0)
    else:
        raise ValueError(f"Invalid trans_id: {trans_id}, must be 0-7")


def inverse_dihedral_transform(grid: np.ndarray, trans_id: int) -> np.ndarray:
    """
    Apply the inverse of a dihedral group transformation.

    Args:
        grid: 2D numpy array that has been transformed
        trans_id: Integer from 0-7 indicating which transformation was applied

    Returns:
        Original grid before transformation
    """
    if trans_id == 0:
        return grid
    elif trans_id == 1:
        return np.rot90(grid, k=3)
    elif trans_id == 2:
        return np.rot90(grid, k=2)
    elif trans_id == 3:
        return np.rot90(grid, k=1)
    elif trans_id == 4:
        return np.flip(grid, axis=0)
    elif trans_id == 5:
        return np.flip(np.rot90(grid, k=3), axis=0)
    elif trans_id == 6:
        return np.flip(grid, axis=1)
    elif trans_id == 7:
        return np.flip(np.rot90(grid, k=1), axis=0)
    else:
        raise ValueError(f"Invalid trans_id: {trans_id}, must be 0-7")

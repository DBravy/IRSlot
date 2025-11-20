"""
ARCPuzzleDataset for loading complete ARC puzzles for solving tasks.

This dataset loads entire puzzles (with multiple train examples and test cases)
rather than individual grids, enabling training of models that solve ARC puzzles
by learning from the train examples.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class ARCPuzzleDataset(Dataset):
    """
    Dataset for ARC puzzle solving.

    Unlike ARCInstanceDataset which returns individual grids for contrastive learning,
    this dataset returns complete puzzles with:
    - Multiple train examples (input-output pairs)
    - Test input(s)
    - Test output(s) (ground truth for evaluation)

    Each puzzle demonstrates a pattern through train examples, and the model must
    learn this pattern to predict the test output from the test input.

    Args:
        data_dir: Directory containing ARC JSON files (e.g., 'kaggle/combined/')
        split: Which split to load ('train', 'eval', or 'test')
        arc_version: 'agi1' or 'agi2'
        max_train_examples: Maximum number of train examples per puzzle (for batching)
        max_test_examples: Maximum number of test examples to include
        augment: Whether to apply augmentations (NOT IMPLEMENTED YET)
        subset_size: Optional limit on number of puzzles to load
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        arc_version: str = 'agi1',
        max_train_examples: int = 10,
        max_test_examples: int = 1,
        augment: bool = False,
        subset_size: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.arc_version = arc_version
        self.max_train_examples = max_train_examples
        self.max_test_examples = max_test_examples
        self.augment = augment

        # Load puzzles from JSON
        self.puzzles = self._load_puzzles()

        # Optionally limit dataset size
        if subset_size is not None and subset_size < len(self.puzzles):
            print(f"Limiting dataset to {subset_size} puzzles (out of {len(self.puzzles)} available)")
            self.puzzles = self.puzzles[:subset_size]

        print(f"Loaded {len(self.puzzles)} puzzles from {split} split (ARC-{arc_version.upper()})")

    def _load_puzzles(self) -> List[Dict]:
        """Load puzzles from JSON files."""
        # Determine file names based on split and version
        if self.arc_version == 'agi1':
            if self.split == 'train':
                challenges_file = 'arc-agi_training_challenges.json'
                solutions_file = 'arc-agi_training_solutions.json'
            elif self.split == 'eval':
                challenges_file = 'arc-agi_evaluation_challenges.json'
                solutions_file = 'arc-agi_evaluation_solutions.json'
            else:
                raise ValueError(f"Invalid split '{self.split}' for ARC-AGI-1. Use 'train' or 'eval'.")
        elif self.arc_version == 'agi2':
            if self.split == 'train':
                challenges_file = 'arc-agi_training2_challenges.json'
                solutions_file = 'arc-agi_training2_solutions.json'
            elif self.split == 'eval':
                challenges_file = 'arc-agi_evaluation2_challenges.json'
                solutions_file = 'arc-agi_evaluation2_solutions.json'
            else:
                raise ValueError(f"Invalid split '{self.split}' for ARC-AGI-2. Use 'train' or 'eval'.")
        else:
            raise ValueError(f"Invalid arc_version '{self.arc_version}'. Use 'agi1' or 'agi2'.")

        # Load JSON files
        challenges_path = self.data_dir / challenges_file
        solutions_path = self.data_dir / solutions_file

        if not challenges_path.exists():
            raise FileNotFoundError(f"Challenges file not found: {challenges_path}")
        if not solutions_path.exists():
            raise FileNotFoundError(f"Solutions file not found: {solutions_path}")

        with open(challenges_path, 'r') as f:
            challenges = json.load(f)

        with open(solutions_path, 'r') as f:
            solutions = json.load(f)

        # Convert to list of puzzle dicts
        puzzles = []
        for puzzle_id, challenge in challenges.items():
            puzzle = {
                'id': puzzle_id,
                'train': challenge['train'],
                'test': challenge['test'],
            }

            # Add solutions
            if puzzle_id in solutions:
                puzzle['test_solutions'] = solutions[puzzle_id]
            else:
                # Some puzzles might not have solutions in eval set
                puzzle['test_solutions'] = [None] * len(puzzle['test'])

            puzzles.append(puzzle)

        return puzzles

    def __len__(self) -> int:
        """Return number of puzzles in the dataset."""
        return len(self.puzzles)

    def _grid_to_tensor(self, grid: List[List[int]]) -> torch.Tensor:
        """Convert grid from list format to tensor."""
        arr = np.array(grid, dtype=np.int64)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single puzzle.

        Returns:
            dict with keys:
            - puzzle_id: str
            - train_inputs: List[Tensor] - [H_i, W_i] for each train example
            - train_outputs: List[Tensor] - [H_i, W_i] for each train example
            - test_inputs: List[Tensor] - [H_i, W_i] for each test example
            - test_outputs: List[Tensor] or List[None] - Ground truth outputs
            - num_train: int - Number of train examples
            - num_test: int - Number of test examples
        """
        puzzle = self.puzzles[idx]

        # Extract train examples
        train_inputs = []
        train_outputs = []
        for example in puzzle['train'][:self.max_train_examples]:
            train_inputs.append(self._grid_to_tensor(example['input']))
            train_outputs.append(self._grid_to_tensor(example['output']))

        # Extract test examples
        test_inputs = []
        test_outputs = []
        for i, test_example in enumerate(puzzle['test'][:self.max_test_examples]):
            test_inputs.append(self._grid_to_tensor(test_example['input']))

            # Add solution if available
            if puzzle['test_solutions'][i] is not None:
                test_outputs.append(self._grid_to_tensor(puzzle['test_solutions'][i]))
            else:
                test_outputs.append(None)

        return {
            'puzzle_id': puzzle['id'],
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'test_inputs': test_inputs,
            'test_outputs': test_outputs,
            'num_train': len(train_inputs),
            'num_test': len(test_inputs),
        }


def collate_puzzle_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for batching puzzles.

    Handles variable number of train examples and variable grid sizes by padding.

    Args:
        batch: List of puzzle dicts from __getitem__

    Returns:
        Batched dict with padded tensors and masks
    """
    batch_size = len(batch)

    # Find maximum dimensions
    max_num_train = max(item['num_train'] for item in batch)
    max_num_test = max(item['num_test'] for item in batch)

    # Find maximum grid dimensions across all grids in batch
    max_h = 0
    max_w = 0

    for item in batch:
        for grid in item['train_inputs'] + item['train_outputs'] + item['test_inputs']:
            h, w = grid.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)

        for grid in item['test_outputs']:
            if grid is not None:
                h, w = grid.shape
                max_h = max(max_h, h)
                max_w = max(max_w, w)

    # Initialize padded tensors
    train_inputs = torch.zeros(batch_size, max_num_train, max_h, max_w, dtype=torch.long)
    train_outputs = torch.zeros(batch_size, max_num_train, max_h, max_w, dtype=torch.long)
    test_inputs = torch.zeros(batch_size, max_num_test, max_h, max_w, dtype=torch.long)
    test_outputs = torch.zeros(batch_size, max_num_test, max_h, max_w, dtype=torch.long)

    # Masks to indicate valid (non-padded) positions
    train_mask = torch.zeros(batch_size, max_num_train, dtype=torch.bool)
    test_mask = torch.zeros(batch_size, max_num_test, dtype=torch.bool)
    test_output_available = torch.zeros(batch_size, max_num_test, dtype=torch.bool)

    # Store original shapes
    train_input_shapes = []
    train_output_shapes = []
    test_input_shapes = []
    test_output_shapes = []

    # Fill tensors
    for b, item in enumerate(batch):
        num_train = item['num_train']
        num_test = item['num_test']

        # Train examples
        train_input_shapes_b = []
        train_output_shapes_b = []
        for i in range(num_train):
            inp = item['train_inputs'][i]
            out = item['train_outputs'][i]
            h_in, w_in = inp.shape
            h_out, w_out = out.shape

            train_inputs[b, i, :h_in, :w_in] = inp
            train_outputs[b, i, :h_out, :w_out] = out
            train_mask[b, i] = True

            train_input_shapes_b.append((h_in, w_in))
            train_output_shapes_b.append((h_out, w_out))

        train_input_shapes.append(train_input_shapes_b)
        train_output_shapes.append(train_output_shapes_b)

        # Test examples
        test_input_shapes_b = []
        test_output_shapes_b = []
        for i in range(num_test):
            inp = item['test_inputs'][i]
            h_in, w_in = inp.shape

            test_inputs[b, i, :h_in, :w_in] = inp
            test_mask[b, i] = True
            test_input_shapes_b.append((h_in, w_in))

            # Test output (if available)
            if item['test_outputs'][i] is not None:
                out = item['test_outputs'][i]
                h_out, w_out = out.shape
                test_outputs[b, i, :h_out, :w_out] = out
                test_output_available[b, i] = True
                test_output_shapes_b.append((h_out, w_out))
            else:
                test_output_shapes_b.append(None)

        test_input_shapes.append(test_input_shapes_b)
        test_output_shapes.append(test_output_shapes_b)

    return {
        'puzzle_ids': [item['puzzle_id'] for item in batch],
        'train_inputs': train_inputs,
        'train_outputs': train_outputs,
        'test_inputs': test_inputs,
        'test_outputs': test_outputs,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'test_output_available': test_output_available,
        'train_input_shapes': train_input_shapes,
        'train_output_shapes': train_output_shapes,
        'test_input_shapes': test_input_shapes,
        'test_output_shapes': test_output_shapes,
        'num_train': torch.tensor([item['num_train'] for item in batch]),
        'num_test': torch.tensor([item['num_test'] for item in batch]),
    }

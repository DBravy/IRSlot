"""
Debug script to check if dataset is loading correctly.
"""
import numpy as np
from dataset.arc_dataset import ARCInstanceDataset

# Load dataset WITHOUT augmentation
dataset = ARCInstanceDataset(
    data_dir='data/processed_agi2',
    split='train',
    subset='all',
    augment=False,  # No augmentation
    max_grid_size=30,
    max_puzzles=100  # Just check first 100 puzzles
)

print(f"Dataset size: {len(dataset)}")
print(f"Unique puzzles: {len(dataset.puzzle_identifiers)}")
print()

# Check first 20 samples
print("Checking first 20 samples:")
uniform_count = 0
diverse_count = 0

for i in range(min(20, len(dataset))):
    sample = dataset[i]
    grid = sample['grid'].numpy()
    unique_colors = np.unique(grid)

    is_uniform = len(unique_colors) <= 2  # Only 1-2 colors
    if is_uniform:
        uniform_count += 1
    else:
        diverse_count += 1

    print(f"\nSample {i}:")
    print(f"  Grid ID: {sample['grid_id']}")
    print(f"  Shape: {grid.shape}")
    print(f"  Original shape: {sample['original_shape']}")
    print(f"  Unique colors: {unique_colors} (count: {len(unique_colors)})")
    print(f"  {'⚠️  UNIFORM COLOR!' if is_uniform else '✓ Diverse'}")
    print(f"  Grid preview (first 5x10):")
    print(f"    {grid[:min(5, grid.shape[0]), :min(10, grid.shape[1])]}")

print(f"\n{'='*60}")
print(f"Summary of first 20 samples:")
print(f"  Uniform/simple grids (≤2 colors): {uniform_count}")
print(f"  Diverse grids (>2 colors): {diverse_count}")
print(f"{'='*60}")

"""
Debug script to check if augmentation is causing the uniform color issue.
"""
import numpy as np
from dataset.arc_dataset import ARCInstanceDataset

print("="*60)
print("Testing WITHOUT augmentation:")
print("="*60)

# Load dataset WITHOUT augmentation
dataset_no_aug = ARCInstanceDataset(
    data_dir='data/processed_agi2',
    split='train',
    subset='all',
    augment=False,
    max_grid_size=30,
    max_puzzles=10
)

sample_no_aug = dataset_no_aug[0]
grid_no_aug = sample_no_aug['grid'].numpy()

print(f"Grid ID: {sample_no_aug['grid_id']}")
print(f"Shape: {grid_no_aug.shape}")
print(f"Unique colors: {np.unique(grid_no_aug)}")
print(f"Grid:\n{grid_no_aug}")

print("\n" + "="*60)
print("Testing WITH augmentation (5 samples from same grid):")
print("="*60)

# Load dataset WITH augmentation
dataset_aug = ARCInstanceDataset(
    data_dir='data/processed_agi2',
    split='train',
    subset='all',
    augment=True,
    max_grid_size=30,
    max_puzzles=10
)

for i in range(5):
    sample_aug = dataset_aug[0]
    grid_aug = sample_aug['grid'].numpy()

    print(f"\nAugmented sample {i}:")
    print(f"  Unique colors: {np.unique(grid_aug)}")
    print(f"  Grid (first 5x10):")
    print(f"    {grid_aug[:min(5, grid_aug.shape[0]), :min(10, grid_aug.shape[1])]}")

    # Check if all values are the same
    if len(np.unique(grid_aug)) == 1:
        print(f"  ⚠️  WARNING: Grid is uniform color {grid_aug[0, 0]}!")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print("If augmented grids show uniform colors but the non-augmented")
print("grid has multiple colors, then the augmentation is the problem.")
print("If both are uniform, then the dataset itself contains simple grids.")

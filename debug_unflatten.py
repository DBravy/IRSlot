"""
Debug the unflatten_grid function to see what's going wrong.
"""
import numpy as np

# Load first input
inputs = np.load('data/processed_agi2/train/all__inputs.npy')
first_input = inputs[0]

print("Raw flattened grid (first 100 values):")
print(first_input[:100])
print(f"\nUnique values: {np.unique(first_input)}")

# Reshape
grid = first_input.reshape(30, 30)
print(f"\nReshaped to 30x30:")
print(f"Grid shape: {grid.shape}")
print(f"Unique values: {np.unique(grid)}")

# Find EOS
eos_positions = np.where(grid == 1)
print(f"\nEOS positions:")
print(f"  Rows with EOS: {eos_positions[0]}")
print(f"  Cols with EOS: {eos_positions[1]}")

if len(eos_positions[0]) > 0:
    H = eos_positions[0].min() if eos_positions[0].min() > 0 else 30
    W = eos_positions[1].min() if eos_positions[1].min() > 0 else 30
else:
    H = W = 30

print(f"\nDetected dimensions: H={H}, W={W}")

# Extract and decode
extracted = grid[:H, :W]
print(f"\nExtracted grid (before decoding):")
print(extracted)
print(f"Unique values: {np.unique(extracted)}")

decoded = np.clip(extracted - 2, 0, 9).astype(np.uint8)
print(f"\nDecoded grid (after subtracting 2 and clipping):")
print(decoded)
print(f"Unique values: {np.unique(decoded)}")

# Now check a "bad" sample - let's look at sample 1 which debug showed as uniform 9s
print("\n" + "="*60)
print("Checking sample 1 (which was reported as uniform 9s):")
print("="*60)

second_input = inputs[1]
grid2 = second_input.reshape(30, 30)
eos_pos2 = np.where(grid2 == 1)

print(f"EOS positions:")
print(f"  Rows with EOS: {eos_pos2[0][:20]}")  # First 20
print(f"  Cols with EOS: {eos_pos2[1][:20]}")

if len(eos_pos2[0]) > 0:
    H2 = eos_pos2[0].min() if eos_pos2[0].min() > 0 else 30
    W2 = eos_pos2[1].min() if eos_pos2[1].min() > 0 else 30
else:
    H2 = W2 = 30

print(f"Detected dimensions: H={H2}, W={W2}")

extracted2 = grid2[:H2, :W2]
print(f"\nExtracted grid (before decoding) - first 10x10:")
print(extracted2[:10, :10])
print(f"Unique values in extracted: {np.unique(extracted2)}")

decoded2 = np.clip(extracted2 - 2, 0, 9).astype(np.uint8)
print(f"\nDecoded grid - first 10x10:")
print(decoded2[:10, :10])
print(f"Unique values in decoded: {np.unique(decoded2)}")

if len(np.unique(decoded2)) == 1:
    print(f"\n⚠️  PROBLEM FOUND: Grid is uniform color {decoded2[0,0]}!")
    print(f"   This means the extracted region had uniform value {extracted2[0,0]}")
    print(f"   Let's check what the full 30x30 grid looks like:")
    print(f"\nFull grid:")
    print(grid2)

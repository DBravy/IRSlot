"""
Test the exact bug scenario that was causing uniform color 9 grids.

This replicates the debug output where:
- Stored data had colors 0,1,2,3 (encoded as 2,3,4,5)
- Grid was at rows 13-15, cols 6-8
- OLD bug: extracted [:13, :6] which was all padding (0s)
- OLD bug: 0 - 2 = -2 in uint8 wraps to 254, clips to 9 → uniform 9
- NEW fix: finds bounding box, converts to int16 → correct colors
"""
import numpy as np

def _unflatten_grid_OLD_BUGGY(flat_grid, max_grid_size=30):
    """OLD BUGGY VERSION - for comparison"""
    grid = flat_grid.reshape(max_grid_size, max_grid_size)

    # OLD BUGGY CODE: Assumed grid starts at (0,0) and used EOS markers
    eos_positions = np.where(grid == 1)
    if len(eos_positions[0]) > 0:
        H = eos_positions[0].min() if eos_positions[0].min() > 0 else 1
        W = eos_positions[1].min() if eos_positions[1].min() > 0 else 1
        grid = grid[:H, :W]
    else:
        # No EOS found, probably take some default
        grid = grid[:10, :10]

    # BUG: uint8 underflow
    grid = (grid.astype(np.uint8) - 2)
    grid = np.clip(grid, 0, 9).astype(np.uint8)

    return grid, grid.shape


def _unflatten_grid_FIXED(flat_grid, max_grid_size=30):
    """FIXED VERSION"""
    grid = flat_grid.reshape(max_grid_size, max_grid_size)

    # NEW: Find bounding box of actual content
    content_mask = grid >= 2
    if content_mask.any():
        rows_with_content = np.where(content_mask.any(axis=1))[0]
        cols_with_content = np.where(content_mask.any(axis=0))[0]

        min_row = rows_with_content.min()
        max_row = rows_with_content.max() + 1
        min_col = cols_with_content.min()
        max_col = cols_with_content.max() + 1

        grid = grid[min_row:max_row, min_col:max_col]
    else:
        grid = np.array([[0]], dtype=np.uint8)

    # NEW: Convert to int16 to avoid underflow
    grid = grid.astype(np.int16) - 2
    grid = np.clip(grid, 0, 9).astype(np.uint8)

    return grid, grid.shape


print("="*70)
print("EXACT BUG SCENARIO FROM DEBUG OUTPUT")
print("="*70)
print()

# Create the exact scenario from debug: grid at rows 13-15, cols 6-8
flat_grid = np.zeros(30*30, dtype=np.uint8)
grid_2d = flat_grid.reshape(30, 30)

# Place 3x3 grid with colors 0,1,2,3 (encoded as 2,3,4,5)
grid_2d[13:16, 6:9] = np.array([
    [2, 3, 4],
    [5, 2, 3],
    [4, 5, 2]
], dtype=np.uint8)

# Add EOS markers (value 1) at boundaries
# EOS at row 16, cols 6-8
grid_2d[16, 6:9] = 1
# EOS at col 9, rows 13-15
grid_2d[13:16, 9] = 1

flat_grid = grid_2d.flatten()

print("Original 30x30 grid (showing relevant region):")
print(f"  Rows 12-17, Cols 5-11:")
print(grid_2d[12:18, 5:11])
print()
print(f"  Content is at rows 13-15, cols 6-8")
print(f"  Encoded values: {np.unique(grid_2d[13:16, 6:9])}")
print(f"  EOS markers at row 16 and col 9")
print()

# Test OLD BUGGY version
print("-" * 70)
print("OLD BUGGY VERSION:")
print("-" * 70)
old_grid, old_shape = _unflatten_grid_OLD_BUGGY(flat_grid)
print(f"Extracted shape: {old_shape}")
print(f"Extracted grid (first 5x5):")
print(old_grid[:5, :5] if old_shape[0] >= 5 and old_shape[1] >= 5 else old_grid)
print(f"Unique colors: {np.unique(old_grid)}")

if len(np.unique(old_grid)) == 1 and old_grid[0, 0] == 9:
    print("❌ BUG CONFIRMED: Grid is uniform color 9!")
    print("   Reason: Extracted padding region [:13, :6], all 0s")
    print("   Then: 0 - 2 = -2 in uint8 wraps to 254, clips to 9")
else:
    print("   (Bug might not reproduce exactly - EOS positions vary)")
print()

# Test FIXED version
print("-" * 70)
print("FIXED VERSION:")
print("-" * 70)
new_grid, new_shape = _unflatten_grid_FIXED(flat_grid)
print(f"Extracted shape: {new_shape}")
print(f"Extracted grid:")
print(new_grid)
print(f"Unique colors: {np.unique(new_grid)}")

if new_shape == (3, 3) and len(np.unique(new_grid)) > 1:
    print("✅ FIX CONFIRMED: Correctly extracted 3x3 grid with multiple colors!")
    print("   Correctly found bounding box of content at (13:16, 6:9)")
    print("   Correctly decoded colors without underflow")
else:
    print("❌ Fix may have issues")

print()
print("="*70)
print("CONCLUSION:")
print("="*70)
print("The fix solves both bugs:")
print("  1. ✅ Finds actual content bounding box (not EOS-based)")
print("  2. ✅ Uses int16 to prevent uint8 underflow")
print()
print("Before fix: 15/20 samples were uniform color 9 (padding bug)")
print("After fix: All samples should have correct diverse colors")
print("="*70)

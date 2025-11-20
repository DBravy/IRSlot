"""
Simple test to verify the unflatten fix works correctly.
This only tests the numpy logic without needing torch.
"""
import numpy as np

def _unflatten_grid_fixed(flat_grid, max_grid_size=30):
    """Fixed version of _unflatten_grid"""
    # Reshape to 2D
    grid = flat_grid.reshape(max_grid_size, max_grid_size)

    # Find actual grid content (values >= 2)
    # Due to translational augmentation, grid can be anywhere in the 30x30 space
    content_mask = grid >= 2
    if content_mask.any():
        # Find bounding box of content
        rows_with_content = np.where(content_mask.any(axis=1))[0]
        cols_with_content = np.where(content_mask.any(axis=0))[0]

        min_row = rows_with_content.min()
        max_row = rows_with_content.max() + 1  # +1 for exclusive end
        min_col = cols_with_content.min()
        max_col = cols_with_content.max() + 1

        # Extract content
        grid = grid[min_row:max_row, min_col:max_col]
    else:
        # No content, return empty 1x1 grid
        grid = np.array([[0]], dtype=np.uint8)

    # Convert from encoding (2-11) to colors (0-9)
    # Convert to int16 first to avoid underflow
    grid = grid.astype(np.int16) - 2
    grid = np.clip(grid, 0, 9).astype(np.uint8)

    return grid, grid.shape


# Test case 1: Grid at position (13, 6) with colors 0,1,2,3
print("="*60)
print("Test 1: Grid with translational augmentation")
print("="*60)

flat_grid = np.zeros(30*30, dtype=np.uint8)
grid_2d = flat_grid.reshape(30, 30)

# Place a 3x3 grid at rows 13-15, cols 6-8
# With encoded values: 2,3,4,5 (representing colors 0,1,2,3)
grid_2d[13:16, 6:9] = np.array([
    [2, 3, 4],
    [5, 2, 3],
    [4, 5, 2]
], dtype=np.uint8)

flat_grid = grid_2d.flatten()

# Apply fix
decoded_grid, shape = _unflatten_grid_fixed(flat_grid)

print(f"Original encoded values at (13:16, 6:9):")
print(grid_2d[13:16, 6:9])
print()
print(f"Decoded grid shape: {shape}")
print(f"Decoded grid:")
print(decoded_grid)
print(f"Unique colors: {np.unique(decoded_grid)}")
print()

if len(np.unique(decoded_grid)) == 1:
    print("❌ FAIL: Grid is uniform color!")
else:
    print("✅ PASS: Grid has multiple colors!")

# Test case 2: Grid at position (0, 0) - no translation
print("\n" + "="*60)
print("Test 2: Grid at origin (no translation)")
print("="*60)

flat_grid2 = np.zeros(30*30, dtype=np.uint8)
grid_2d2 = flat_grid2.reshape(30, 30)

# Place a 2x4 grid at rows 0-1, cols 0-3
grid_2d2[0:2, 0:4] = np.array([
    [2, 5, 7, 3],
    [4, 6, 2, 8]
], dtype=np.uint8)

flat_grid2 = grid_2d2.flatten()

decoded_grid2, shape2 = _unflatten_grid_fixed(flat_grid2)

print(f"Decoded grid shape: {shape2}")
print(f"Decoded grid:")
print(decoded_grid2)
print(f"Unique colors: {np.unique(decoded_grid2)}")
print()

if len(np.unique(decoded_grid2)) == 1:
    print("❌ FAIL: Grid is uniform color!")
else:
    print("✅ PASS: Grid has multiple colors!")

# Test case 3: Verify no underflow
print("\n" + "="*60)
print("Test 3: Verify no integer underflow with padding")
print("="*60)

flat_grid3 = np.zeros(30*30, dtype=np.uint8)  # All padding (value 0)
grid_2d3 = flat_grid3.reshape(30, 30)

# Place small grid with value 2 (color 0)
grid_2d3[10:12, 10:12] = 2

flat_grid3 = grid_2d3.flatten()

decoded_grid3, shape3 = _unflatten_grid_fixed(flat_grid3)

print(f"Encoded value: 2 (should decode to color 0)")
print(f"Decoded grid:")
print(decoded_grid3)
print(f"Unique colors: {np.unique(decoded_grid3)}")
print()

if np.all(decoded_grid3 == 0):
    print("✅ PASS: Correctly decoded to color 0 (no underflow to 9)!")
else:
    print(f"❌ FAIL: Expected all 0s, got {np.unique(decoded_grid3)}")

print("\n" + "="*60)
print("Summary: All tests verify the fix handles:")
print("  1. Translational augmentation (grids anywhere in 30x30)")
print("  2. No integer underflow (0-2 doesn't wrap to 254→9)")
print("="*60)

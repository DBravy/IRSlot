"""
Comprehensive test script for ARCPuzzleDataset.

Tests loading, batching, and handling of ARC puzzles for solving tasks.
"""
import torch
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset.arc_puzzle_dataset import ARCPuzzleDataset, collate_puzzle_batch


class TestColors:
    """ANSI color codes for pretty terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_test_header(test_name):
    """Print a formatted test header"""
    print(f"\n{TestColors.HEADER}{TestColors.BOLD}{'='*70}{TestColors.ENDC}")
    print(f"{TestColors.HEADER}{TestColors.BOLD}TEST: {test_name}{TestColors.ENDC}")
    print(f"{TestColors.HEADER}{TestColors.BOLD}{'='*70}{TestColors.ENDC}\n")


def print_success(message):
    """Print success message"""
    print(f"{TestColors.OKGREEN}✓ {message}{TestColors.ENDC}")


def print_info(message):
    """Print info message"""
    print(f"{TestColors.OKCYAN}  {message}{TestColors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{TestColors.FAIL}✗ {message}{TestColors.ENDC}")


def print_warning(message):
    """Print warning message"""
    print(f"{TestColors.WARNING}⚠ {message}{TestColors.ENDC}")


def test_dataset_loading():
    """Test basic dataset loading"""
    print_test_header("Dataset Loading")

    data_dir = "kaggle/combined"

    try:
        # Test ARC-AGI-1 training set
        dataset_train = ARCPuzzleDataset(
            data_dir=data_dir,
            split='train',
            arc_version='agi1',
            subset_size=10  # Load only 10 puzzles for testing
        )

        print_info(f"ARC-AGI-1 train set size: {len(dataset_train)} puzzles")
        print_success("Successfully loaded ARC-AGI-1 training set")

        # Test ARC-AGI-1 evaluation set
        dataset_eval = ARCPuzzleDataset(
            data_dir=data_dir,
            split='eval',
            arc_version='agi1',
            subset_size=5
        )

        print_info(f"ARC-AGI-1 eval set size: {len(dataset_eval)} puzzles")
        print_success("Successfully loaded ARC-AGI-1 evaluation set")

        return True

    except FileNotFoundError as e:
        print_error(f"Dataset files not found: {e}")
        print_warning("Make sure kaggle/combined/ directory exists with ARC JSON files")
        return False
    except Exception as e:
        print_error(f"Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_puzzle():
    """Test loading a single puzzle"""
    print_test_header("Single Puzzle Loading")

    data_dir = "kaggle/combined"

    try:
        dataset = ARCPuzzleDataset(
            data_dir=data_dir,
            split='train',
            arc_version='agi1',
            subset_size=5
        )

        # Get first puzzle
        puzzle = dataset[0]

        print_info(f"Puzzle ID: {puzzle['puzzle_id']}")
        print_info(f"Number of train examples: {puzzle['num_train']}")
        print_info(f"Number of test examples: {puzzle['num_test']}")

        # Check structure
        assert 'puzzle_id' in puzzle
        assert 'train_inputs' in puzzle
        assert 'train_outputs' in puzzle
        assert 'test_inputs' in puzzle
        assert 'test_outputs' in puzzle
        print_success("Puzzle has correct structure")

        # Check train examples
        print_info(f"\nTrain examples:")
        for i, (inp, out) in enumerate(zip(puzzle['train_inputs'], puzzle['train_outputs'])):
            print_info(f"  Example {i}: input {tuple(inp.shape)}, output {tuple(out.shape)}")
            assert inp.dtype == torch.long
            assert out.dtype == torch.long
            assert (inp >= 0).all() and (inp <= 9).all(), "Input contains invalid color values"
            assert (out >= 0).all() and (out <= 9).all(), "Output contains invalid color values"

        print_success("Train examples have valid shapes and values")

        # Check test examples
        print_info(f"\nTest examples:")
        for i, inp in enumerate(puzzle['test_inputs']):
            out = puzzle['test_outputs'][i]
            out_shape = tuple(out.shape) if out is not None else "N/A"
            print_info(f"  Example {i}: input {tuple(inp.shape)}, output {out_shape}")
            assert inp.dtype == torch.long
            assert (inp >= 0).all() and (inp <= 9).all(), "Test input contains invalid color values"

        print_success("Test examples have valid shapes and values")

        return True

    except Exception as e:
        print_error(f"Single puzzle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_puzzle_sizes():
    """Test that puzzles have different sizes and structures"""
    print_test_header("Variable Puzzle Sizes")

    data_dir = "kaggle/combined"

    try:
        dataset = ARCPuzzleDataset(
            data_dir=data_dir,
            split='train',
            arc_version='agi1',
            subset_size=20
        )

        num_train_examples = []
        grid_sizes = []

        print_info("Sampling puzzle statistics:")
        for i in range(min(10, len(dataset))):
            puzzle = dataset[i]
            num_train_examples.append(puzzle['num_train'])

            # Collect grid sizes
            for inp in puzzle['train_inputs']:
                grid_sizes.append(inp.shape)
            for out in puzzle['train_outputs']:
                grid_sizes.append(out.shape)

        print_info(f"  Train examples per puzzle: {set(num_train_examples)}")
        print_info(f"  Unique grid sizes (sample): {set(list(grid_sizes)[:10])}")

        # Check variability
        assert len(set(num_train_examples)) > 1, "All puzzles have same number of train examples"
        print_success("Puzzles have variable number of train examples")

        assert len(set(grid_sizes)) > 1, "All grids have the same size"
        print_success("Grids have variable sizes")

        return True

    except Exception as e:
        print_error(f"Variable size test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collate_function():
    """Test the collate function for batching"""
    print_test_header("Collate Function (Batching)")

    data_dir = "kaggle/combined"

    try:
        dataset = ARCPuzzleDataset(
            data_dir=data_dir,
            split='train',
            arc_version='agi1',
            subset_size=10
        )

        # Manually create a batch
        batch = [dataset[i] for i in range(min(4, len(dataset)))]

        print_info(f"Creating batch from {len(batch)} puzzles")

        # Collate
        collated = collate_puzzle_batch(batch)

        print_info(f"\nCollated batch keys: {list(collated.keys())}")

        # Check tensor shapes
        batch_size = len(batch)
        print_info(f"\nBatch tensors:")
        print_info(f"  train_inputs: {tuple(collated['train_inputs'].shape)}")
        print_info(f"  train_outputs: {tuple(collated['train_outputs'].shape)}")
        print_info(f"  test_inputs: {tuple(collated['test_inputs'].shape)}")
        print_info(f"  test_outputs: {tuple(collated['test_outputs'].shape)}")

        # Check masks
        print_info(f"\nMasks:")
        print_info(f"  train_mask: {tuple(collated['train_mask'].shape)}")
        print_info(f"  test_mask: {tuple(collated['test_mask'].shape)}")
        print_info(f"  test_output_available: {tuple(collated['test_output_available'].shape)}")

        # Validate shapes
        assert collated['train_inputs'].shape[0] == batch_size
        assert collated['train_outputs'].shape[0] == batch_size
        assert collated['test_inputs'].shape[0] == batch_size
        assert collated['test_outputs'].shape[0] == batch_size
        print_success("Batch tensors have correct batch dimension")

        # Check masks match actual data
        for b in range(batch_size):
            num_train = collated['num_train'][b].item()
            num_test = collated['num_test'][b].item()

            assert collated['train_mask'][b, :num_train].all(), "Train mask incorrect"
            assert collated['test_mask'][b, :num_test].all(), "Test mask incorrect"

        print_success("Masks correctly indicate valid examples")

        # Check shapes metadata
        assert len(collated['train_input_shapes']) == batch_size
        assert len(collated['test_input_shapes']) == batch_size
        print_success("Shape metadata included")

        return True

    except Exception as e:
        print_error(f"Collate function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test using dataset with PyTorch DataLoader"""
    print_test_header("DataLoader Integration")

    data_dir = "kaggle/combined"

    try:
        dataset = ARCPuzzleDataset(
            data_dir=data_dir,
            split='train',
            arc_version='agi1',
            subset_size=12
        )

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            collate_fn=collate_puzzle_batch
        )

        print_info(f"Created DataLoader with {len(dataloader)} batches")

        # Iterate through a few batches
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Only test first 2 batches
                break

            print_info(f"\nBatch {i+1}:")
            print_info(f"  Batch size: {len(batch['puzzle_ids'])}")
            print_info(f"  Train inputs shape: {tuple(batch['train_inputs'].shape)}")
            print_info(f"  Test inputs shape: {tuple(batch['test_inputs'].shape)}")
            print_info(f"  Puzzle IDs: {batch['puzzle_ids']}")

            # Validate
            assert batch['train_inputs'].dtype == torch.long
            assert batch['test_inputs'].dtype == torch.long

        print_success("DataLoader iteration works correctly")

        return True

    except Exception as e:
        print_error(f"DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arc_agi2():
    """Test loading ARC-AGI-2 dataset"""
    print_test_header("ARC-AGI-2 Dataset")

    data_dir = "kaggle/combined"

    try:
        # Test ARC-AGI-2 training set
        dataset_train = ARCPuzzleDataset(
            data_dir=data_dir,
            split='train',
            arc_version='agi2',
            subset_size=5
        )

        print_info(f"ARC-AGI-2 train set size: {len(dataset_train)} puzzles")
        print_success("Successfully loaded ARC-AGI-2 training set")

        # Get a puzzle
        puzzle = dataset_train[0]
        print_info(f"Sample puzzle ID: {puzzle['puzzle_id']}")
        print_info(f"  Train examples: {puzzle['num_train']}")
        print_info(f"  Test examples: {puzzle['num_test']}")

        return True

    except FileNotFoundError as e:
        print_warning(f"ARC-AGI-2 files not found: {e}")
        print_info("This is OK if you only have ARC-AGI-1 data")
        return True  # Don't fail test if AGI-2 not available
    except Exception as e:
        print_error(f"ARC-AGI-2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_max_examples_limit():
    """Test limiting max train/test examples"""
    print_test_header("Max Examples Limit")

    data_dir = "kaggle/combined"

    try:
        # Create dataset with limits
        dataset = ARCPuzzleDataset(
            data_dir=data_dir,
            split='train',
            arc_version='agi1',
            max_train_examples=3,
            max_test_examples=1,
            subset_size=10
        )

        # Check that limits are enforced
        for i in range(len(dataset)):
            puzzle = dataset[i]
            assert puzzle['num_train'] <= 3, f"Puzzle has {puzzle['num_train']} train examples (max 3)"
            assert puzzle['num_test'] <= 1, f"Puzzle has {puzzle['num_test']} test examples (max 1)"

        print_success("Max examples limits enforced correctly")

        return True

    except Exception as e:
        print_error(f"Max examples test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_puzzle_content():
    """Test that puzzle content is valid and makes sense"""
    print_test_header("Puzzle Content Validation")

    data_dir = "kaggle/combined"

    try:
        dataset = ARCPuzzleDataset(
            data_dir=data_dir,
            split='train',
            arc_version='agi1',
            subset_size=5
        )

        puzzle = dataset[0]

        print_info(f"Validating puzzle: {puzzle['puzzle_id']}")

        # Check that grids contain valid ARC colors (0-9)
        for i, (inp, out) in enumerate(zip(puzzle['train_inputs'], puzzle['train_outputs'])):
            min_val = inp.min().item()
            max_val = inp.max().item()
            assert 0 <= min_val <= 9, f"Train input {i} has invalid min value: {min_val}"
            assert 0 <= max_val <= 9, f"Train input {i} has invalid max value: {max_val}"

            min_val = out.min().item()
            max_val = out.max().item()
            assert 0 <= min_val <= 9, f"Train output {i} has invalid min value: {min_val}"
            assert 0 <= max_val <= 9, f"Train output {i} has invalid max value: {max_val}"

        print_success("All grids contain valid ARC color values (0-9)")

        # Check grid sizes are reasonable (max 30x30 in ARC)
        for i, (inp, out) in enumerate(zip(puzzle['train_inputs'], puzzle['train_outputs'])):
            assert inp.shape[0] <= 30 and inp.shape[1] <= 30, f"Train input {i} too large: {inp.shape}"
            assert out.shape[0] <= 30 and out.shape[1] <= 30, f"Train output {i} too large: {out.shape}"

        print_success("All grids are within ARC size limits (≤30×30)")

        # Print a sample grid
        print_info(f"\nSample train input (grid 0):")
        sample_grid = puzzle['train_inputs'][0].numpy()
        h, w = sample_grid.shape
        print_info(f"  Shape: {h}×{w}")
        print_info(f"  Colors present: {sorted(set(sample_grid.flatten().tolist()))}")

        return True

    except Exception as e:
        print_error(f"Puzzle content validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test cases"""
    print(f"\n{TestColors.BOLD}{TestColors.HEADER}")
    print("="*70)
    print(" ARCPuzzleDataset Comprehensive Test Suite")
    print("="*70)
    print(f"{TestColors.ENDC}\n")

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Single Puzzle", test_single_puzzle),
        ("Variable Puzzle Sizes", test_variable_puzzle_sizes),
        ("Collate Function", test_collate_function),
        ("DataLoader Integration", test_dataloader),
        ("ARC-AGI-2", test_arc_agi2),
        ("Max Examples Limit", test_max_examples_limit),
        ("Puzzle Content", test_puzzle_content),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print(f"\n{TestColors.BOLD}{TestColors.HEADER}")
    print("="*70)
    print(" Test Summary")
    print("="*70)
    print(f"{TestColors.ENDC}\n")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = f"{TestColors.OKGREEN}PASSED{TestColors.ENDC}" if passed else f"{TestColors.FAIL}FAILED{TestColors.ENDC}"
        print(f"  {test_name:.<50} {status}")

    print(f"\n{TestColors.BOLD}Total: {passed_count}/{total_count} tests passed{TestColors.ENDC}\n")

    if passed_count == total_count:
        print(f"{TestColors.OKGREEN}{TestColors.BOLD}🎉 All tests passed!{TestColors.ENDC}\n")
        return 0
    else:
        print(f"{TestColors.FAIL}{TestColors.BOLD}❌ Some tests failed{TestColors.ENDC}\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

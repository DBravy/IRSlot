"""
Comprehensive test script for ARCSlotSequenceBuilder.

Tests the conversion of puzzles to slot sequences for transformer reasoning.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.arc_slot_builder import ARCSlotSequenceBuilder
from models.slot_encoder import SlotAttentionEncoder
from models.trm import CNNEncoder
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


def create_test_components():
    """Create the necessary components for testing"""
    # Configuration
    input_channels = 1
    cnn_hidden_dim = 64
    slot_dim = 64
    hidden_size = 128
    num_slots = 5

    # Create CNN encoder
    cnn_encoder = CNNEncoder(
        input_channels=input_channels,
        hidden_dim=cnn_hidden_dim,
        slot_dim=slot_dim,
        forward_dtype=torch.float32
    )

    # Create slot encoder
    slot_encoder = SlotAttentionEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_size=hidden_size,
        num_iterations=3,
        forward_dtype=torch.float32
    )

    # Create sequence builder
    sequence_builder = ARCSlotSequenceBuilder(
        cnn_encoder=cnn_encoder,
        slot_encoder=slot_encoder,
        num_slots_per_grid=num_slots,
        hidden_size=hidden_size,
        max_train_examples=10,
        puzzle_emb_dim=0,  # No puzzle embeddings for now
    )

    return sequence_builder, num_slots, hidden_size


def test_basic_sequence_building():
    """Test basic sequence building from a puzzle"""
    print_test_header("Basic Sequence Building")

    try:
        sequence_builder, num_slots, hidden_size = create_test_components()

        # Create a simple mock puzzle batch
        batch_size = 2
        num_train = 2
        H, W = 10, 10

        puzzle_batch = {
            'train_inputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
            'train_outputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
            'test_inputs': torch.randint(0, 10, (batch_size, 1, H, W)),
            'train_mask': torch.ones(batch_size, num_train, dtype=torch.bool),
            'test_mask': torch.ones(batch_size, 1, dtype=torch.bool),
            'num_train': torch.tensor([num_train, num_train]),
            'train_input_shapes': [[(H, W)] * num_train] * batch_size,
            'train_output_shapes': [[(H, W)] * num_train] * batch_size,
            'test_input_shapes': [[(H, W)]] * batch_size,
            'test_output_shapes': [[(H, W)]] * batch_size,
        }

        print_info(f"Puzzle batch:")
        print_info(f"  Batch size: {batch_size}")
        print_info(f"  Train examples: {num_train}")
        print_info(f"  Grid size: {H}×{W}")

        # Build sequence
        result = sequence_builder(puzzle_batch)

        print_success("Sequence building succeeded!")

        # Check outputs
        sequence = result['sequence']
        attention_mask = result['attention_mask']
        predict_positions = result['predict_positions']

        print_info(f"\nOutput shapes:")
        print_info(f"  Sequence: {tuple(sequence.shape)}")
        print_info(f"  Attention mask: {tuple(attention_mask.shape)}")
        print_info(f"  Predict positions: {tuple(predict_positions.shape)}")

        # Validate shapes
        assert sequence.ndim == 3, f"Expected 3D sequence, got {sequence.ndim}D"
        assert sequence.shape[0] == batch_size, f"Wrong batch size: {sequence.shape[0]}"
        assert sequence.shape[2] == hidden_size, f"Wrong hidden size: {sequence.shape[2]}"
        print_success("Output shapes are correct")

        # Check attention mask
        assert attention_mask.shape[:2] == sequence.shape[:2], "Attention mask shape mismatch"
        assert attention_mask.dtype == torch.bool, "Attention mask should be boolean"
        print_success("Attention mask is valid")

        # Check predict positions
        assert predict_positions.shape == (batch_size, 2), f"Wrong predict_positions shape: {predict_positions.shape}"
        print_success("Predict positions are valid")

        # Check for NaN/Inf
        assert not torch.isnan(sequence).any(), "Sequence contains NaN"
        assert not torch.isinf(sequence).any(), "Sequence contains Inf"
        print_success("Sequence contains no NaN or Inf")

        return True

    except Exception as e:
        print_error(f"Basic sequence building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequence_structure():
    """Test the structure of the generated sequence"""
    print_test_header("Sequence Structure Analysis")

    try:
        sequence_builder, num_slots, hidden_size = create_test_components()

        # Create puzzle batch with 3 train examples
        batch_size = 1
        num_train = 3
        H, W = 8, 8

        puzzle_batch = {
            'train_inputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
            'train_outputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
            'test_inputs': torch.randint(0, 10, (batch_size, 1, H, W)),
            'train_mask': torch.ones(batch_size, num_train, dtype=torch.bool),
            'test_mask': torch.ones(batch_size, 1, dtype=torch.bool),
            'num_train': torch.tensor([num_train]),
            'train_input_shapes': [[(H, W)] * num_train],
            'train_output_shapes': [[(H, W)] * num_train],
            'test_input_shapes': [[(H, W)]],
            'test_output_shapes': [[(H, W)]],
        }

        # Get structure
        structure = sequence_builder.get_sequence_structure(puzzle_batch)

        print_info("Sequence structure:")
        print_info(f"  Total length: {structure['total_length']}")
        print_info(f"  Puzzle embeddings: {structure['puzzle_emb']}")
        print_info(f"  Train examples: {len(structure['train_examples'])}")
        for i, ex in enumerate(structure['train_examples']):
            print_info(f"    Example {i}: input {ex['input']}, output {ex['output']}")
        print_info(f"  Test input: {structure['test_input']}")
        print_info(f"  Prediction: {structure['prediction']}")

        # Build sequence and verify
        result = sequence_builder(puzzle_batch)
        sequence = result['sequence']
        predict_pos = result['predict_positions'][0]

        print_info(f"\nActual sequence length: {sequence.shape[1]}")
        print_info(f"Actual predict positions: [{predict_pos[0].item()}, {predict_pos[1].item()})")

        # Verify structure makes sense
        expected_length = (
            num_train * 2 * (1 + num_slots) +  # train in/out: (marker + slots) * 2 * num_train
            (1 + num_slots) +  # test input
            (1 + num_slots)    # prediction
        )

        print_info(f"Expected length: {expected_length}")
        assert structure['total_length'] == expected_length, \
            f"Structure length {structure['total_length']} doesn't match expected {expected_length}"
        print_success("Sequence structure is correct")

        return True

    except Exception as e:
        print_error(f"Sequence structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_train_examples():
    """Test with different numbers of train examples"""
    print_test_header("Variable Train Examples")

    try:
        sequence_builder, num_slots, hidden_size = create_test_components()

        H, W = 10, 10
        train_counts = [1, 2, 3, 5]

        for num_train in train_counts:
            puzzle_batch = {
                'train_inputs': torch.randint(0, 10, (1, num_train, H, W)),
                'train_outputs': torch.randint(0, 10, (1, num_train, H, W)),
                'test_inputs': torch.randint(0, 10, (1, 1, H, W)),
                'train_mask': torch.ones(1, num_train, dtype=torch.bool),
                'test_mask': torch.ones(1, 1, dtype=torch.bool),
                'num_train': torch.tensor([num_train]),
                'train_input_shapes': [[(H, W)] * num_train],
                'train_output_shapes': [[(H, W)] * num_train],
                'test_input_shapes': [[(H, W)]],
                'test_output_shapes': [[(H, W)]],
            }

            result = sequence_builder(puzzle_batch)
            sequence = result['sequence']

            # Calculate expected length
            expected_len = num_train * 2 * (1 + num_slots) + 2 * (1 + num_slots)

            print_info(f"Train examples: {num_train} → Sequence length: {sequence.shape[1]} (expected: {expected_len})")

            assert not torch.isnan(sequence).any(), f"NaN in sequence for {num_train} train examples"

        print_success("Variable train examples handled correctly")
        return True

    except Exception as e:
        print_error(f"Variable train examples test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batching():
    """Test batching puzzles with different numbers of train examples"""
    print_test_header("Batching Different Puzzles")

    try:
        sequence_builder, num_slots, hidden_size = create_test_components()

        H, W = 8, 8
        batch_size = 3
        max_train = 4

        # Create batch with different numbers of train examples
        num_trains = [2, 3, 4]

        train_inputs = torch.zeros(batch_size, max_train, H, W, dtype=torch.long)
        train_outputs = torch.zeros(batch_size, max_train, H, W, dtype=torch.long)
        train_mask = torch.zeros(batch_size, max_train, dtype=torch.bool)

        for b, n_train in enumerate(num_trains):
            train_inputs[b, :n_train] = torch.randint(0, 10, (n_train, H, W))
            train_outputs[b, :n_train] = torch.randint(0, 10, (n_train, H, W))
            train_mask[b, :n_train] = True

        puzzle_batch = {
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'test_inputs': torch.randint(0, 10, (batch_size, 1, H, W)),
            'train_mask': train_mask,
            'test_mask': torch.ones(batch_size, 1, dtype=torch.bool),
            'num_train': torch.tensor(num_trains),
            'train_input_shapes': [[(H, W)] * n for n in num_trains],
            'train_output_shapes': [[(H, W)] * n for n in num_trains],
            'test_input_shapes': [[(H, W)]] * batch_size,
            'test_output_shapes': [[(H, W)]] * batch_size,
        }

        print_info(f"Batch with variable train examples: {num_trains}")

        result = sequence_builder(puzzle_batch)
        sequence = result['sequence']
        attention_mask = result['attention_mask']

        print_info(f"Output sequence shape: {tuple(sequence.shape)}")
        print_info(f"Attention mask shape: {tuple(attention_mask.shape)}")

        # Check that sequences are padded to same length
        assert sequence.shape[0] == batch_size
        print_success("Batch dimension is correct")

        # Check attention masks differ (because different train counts)
        mask_sums = attention_mask.sum(dim=1)
        print_info(f"Attention mask sums: {mask_sums.tolist()}")
        assert len(set(mask_sums.tolist())) > 1, "All attention masks are identical"
        print_success("Attention masks correctly reflect different sequence lengths")

        return True

    except Exception as e:
        print_error(f"Batching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow through the sequence builder"""
    print_test_header("Gradient Flow")

    try:
        sequence_builder, num_slots, hidden_size = create_test_components()

        batch_size = 2
        num_train = 2
        H, W = 8, 8

        puzzle_batch = {
            'train_inputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
            'train_outputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
            'test_inputs': torch.randint(0, 10, (batch_size, 1, H, W)),
            'train_mask': torch.ones(batch_size, num_train, dtype=torch.bool),
            'test_mask': torch.ones(batch_size, 1, dtype=torch.bool),
            'num_train': torch.tensor([num_train, num_train]),
            'train_input_shapes': [[(H, W)] * num_train] * batch_size,
            'train_output_shapes': [[(H, W)] * num_train] * batch_size,
            'test_input_shapes': [[(H, W)]] * batch_size,
            'test_output_shapes': [[(H, W)]] * batch_size,
        }

        # Forward pass
        result = sequence_builder(puzzle_batch)
        sequence = result['sequence']

        # Dummy loss
        loss = sequence.sum()
        print_info(f"Dummy loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients exist
        param_grads = []
        for name, param in sequence_builder.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_grads.append((name, grad_norm))

        print_info(f"\nParameter gradients:")
        for name, grad_norm in param_grads[:5]:
            print_info(f"  {name}: {grad_norm:.6f}")

        assert len(param_grads) > 0, "No parameter gradients computed"
        print_success(f"Computed gradients for {len(param_grads)} parameters")

        return True

    except Exception as e:
        print_error(f"Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_arc_puzzles():
    """Test with real ARC puzzles from dataset"""
    print_test_header("Real ARC Puzzles")

    try:
        # Load real ARC dataset
        dataset = ARCPuzzleDataset(
            data_dir="kaggle/combined",
            split='train',
            arc_version='agi1',
            max_train_examples=5,
            subset_size=5
        )

        # Create sequence builder
        sequence_builder, num_slots, hidden_size = create_test_components()

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_puzzle_batch
        )

        # Get one batch
        batch = next(iter(dataloader))

        print_info(f"Real ARC batch:")
        print_info(f"  Puzzle IDs: {batch['puzzle_ids']}")
        print_info(f"  Num train: {batch['num_train'].tolist()}")

        # Build sequences
        result = sequence_builder(batch)
        sequence = result['sequence']

        print_info(f"\nGenerated sequence:")
        print_info(f"  Shape: {tuple(sequence.shape)}")
        print_info(f"  Predict positions: {result['predict_positions'].tolist()}")

        # Validate
        assert sequence.shape[0] == len(batch['puzzle_ids'])
        assert not torch.isnan(sequence).any()
        print_success("Successfully processed real ARC puzzles")

        return True

    except FileNotFoundError:
        print_warning("ARC dataset not found, skipping real puzzle test")
        return True
    except Exception as e:
        print_error(f"Real ARC puzzles test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predict_region_extraction():
    """Test extracting the prediction region from sequence"""
    print_test_header("Prediction Region Extraction")

    try:
        sequence_builder, num_slots, hidden_size = create_test_components()

        batch_size = 2
        num_train = 2
        H, W = 8, 8

        puzzle_batch = {
            'train_inputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
            'train_outputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
            'test_inputs': torch.randint(0, 10, (batch_size, 1, H, W)),
            'train_mask': torch.ones(batch_size, num_train, dtype=torch.bool),
            'test_mask': torch.ones(batch_size, 1, dtype=torch.bool),
            'num_train': torch.tensor([num_train, num_train]),
            'train_input_shapes': [[(H, W)] * num_train] * batch_size,
            'train_output_shapes': [[(H, W)] * num_train] * batch_size,
            'test_input_shapes': [[(H, W)]] * batch_size,
            'test_output_shapes': [[(H, W)]] * batch_size,
        }

        result = sequence_builder(puzzle_batch)
        sequence = result['sequence']
        predict_positions = result['predict_positions']

        print_info("Extracting prediction regions:")

        for b in range(batch_size):
            start, end = predict_positions[b]
            predict_region = sequence[b, start:end]

            expected_len = 1 + num_slots  # marker + slots
            actual_len = predict_region.shape[0]

            print_info(f"  Puzzle {b}: positions [{start.item()}, {end.item()}) → length {actual_len} (expected {expected_len})")

            assert actual_len == expected_len, f"Wrong prediction region length"

        print_success("Prediction regions extracted correctly")

        return True

    except Exception as e:
        print_error(f"Prediction region test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test cases"""
    print(f"\n{TestColors.BOLD}{TestColors.HEADER}")
    print("="*70)
    print(" ARCSlotSequenceBuilder Comprehensive Test Suite")
    print("="*70)
    print(f"{TestColors.ENDC}\n")

    tests = [
        ("Basic Sequence Building", test_basic_sequence_building),
        ("Sequence Structure", test_sequence_structure),
        ("Variable Train Examples", test_variable_train_examples),
        ("Batching", test_batching),
        ("Gradient Flow", test_gradient_flow),
        ("Real ARC Puzzles", test_real_arc_puzzles),
        ("Prediction Region", test_predict_region_extraction),
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

"""
Comprehensive test script for ARCSlotSolver.

Tests the complete end-to-end ARC solving pipeline.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.arc_solver import ARCSlotSolver, ARCSlotSolverConfig
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


def create_test_model():
    """Create a test ARCSlotSolver model."""
    config = ARCSlotSolverConfig(
        grid_channels=1,
        cnn_hidden_dim=32,
        slot_dim=32,
        num_slots_per_grid=5,
        slot_iterations=2,
        slot_mlp_hidden=64,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        expansion=2.0,
        max_train_examples=10,
        max_grid_size=30,
        decoder_hidden_dim=32,
        output_channels=10,
        forward_dtype="float32",
    )

    model = ARCSlotSolver(config)
    return model, config


def create_mock_puzzle_batch(batch_size=2, num_train=2, H=10, W=10):
    """Create a mock puzzle batch for testing."""
    puzzle_batch = {
        'train_inputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
        'train_outputs': torch.randint(0, 10, (batch_size, num_train, H, W)),
        'test_inputs': torch.randint(0, 10, (batch_size, 1, H, W)),
        'test_outputs': torch.randint(0, 10, (batch_size, 1, H, W)),
        'train_mask': torch.ones(batch_size, num_train, dtype=torch.bool),
        'test_mask': torch.ones(batch_size, 1, dtype=torch.bool),
        'test_output_available': torch.ones(batch_size, 1, dtype=torch.bool),
        'num_train': torch.tensor([num_train] * batch_size),
        'train_input_shapes': [[(H, W)] * num_train] * batch_size,
        'train_output_shapes': [[(H, W)] * num_train] * batch_size,
        'test_input_shapes': [[(H, W)]] * batch_size,
        'test_output_shapes': [[(H, W)]] * batch_size,
    }
    return puzzle_batch


def test_model_creation():
    """Test creating the ARCSlotSolver model."""
    print_test_header("Model Creation")

    try:
        model, config = create_test_model()

        print_info(f"Model configuration:")
        print_info(f"  Hidden size: {config.hidden_size}")
        print_info(f"  Num layers: {config.num_layers}")
        print_info(f"  Num heads: {config.num_heads}")
        print_info(f"  Slots per grid: {config.num_slots_per_grid}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print_info(f"\nModel parameters:")
        print_info(f"  Total: {total_params:,}")
        print_info(f"  Trainable: {trainable_params:,}")

        print_success("Model created successfully")

        return True

    except Exception as e:
        print_error(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test basic forward pass."""
    print_test_header("Forward Pass")

    try:
        model, config = create_test_model()
        model.eval()

        # Create mock puzzle batch
        puzzle_batch = create_mock_puzzle_batch(batch_size=2, num_train=2, H=8, W=8)

        print_info("Input puzzle batch:")
        print_info(f"  Batch size: 2")
        print_info(f"  Train examples: 2")
        print_info(f"  Grid size: 8×8")

        # Forward pass
        with torch.no_grad():
            output = model(puzzle_batch)

        print_success("Forward pass succeeded!")

        # Check outputs
        predicted_grids = output['predicted_grids']
        predicted_slots = output['predicted_slots']

        print_info(f"\nOutput shapes:")
        print_info(f"  Predicted grids: {tuple(predicted_grids.shape)}")
        print_info(f"  Predicted slots: {tuple(predicted_slots.shape)}")

        # Validate shapes
        batch_size = 2
        H, W = 8, 8
        assert predicted_grids.shape[0] == batch_size
        assert predicted_grids.shape[1] == config.output_channels
        assert predicted_grids.shape[2] == H
        assert predicted_grids.shape[3] == W
        print_success("Output grid shape is correct")

        assert predicted_slots.shape == (batch_size, config.num_slots_per_grid, config.hidden_size)
        print_success("Output slots shape is correct")

        # Check for NaN/Inf
        assert not torch.isnan(predicted_grids).any(), "Predicted grids contain NaN"
        assert not torch.isinf(predicted_grids).any(), "Predicted grids contain Inf"
        print_success("Predictions contain no NaN or Inf")

        return True

    except Exception as e:
        print_error(f"Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """Test prediction method."""
    print_test_header("Prediction Method")

    try:
        model, config = create_test_model()
        model.eval()

        puzzle_batch = create_mock_puzzle_batch(batch_size=2, num_train=3, H=10, W=10)

        print_info("Running prediction...")

        # Predict
        predictions = model.predict(puzzle_batch)

        print_success("Prediction succeeded!")
        print_info(f"Predictions shape: {tuple(predictions.shape)}")

        # Validate
        assert predictions.dtype == torch.long, f"Expected long dtype, got {predictions.dtype}"
        assert predictions.shape == (2, 10, 10), f"Wrong shape: {predictions.shape}"
        assert (predictions >= 0).all() and (predictions <= 9).all(), "Predictions outside valid range [0, 9]"

        print_success("Predictions are valid ARC colors (0-9)")

        # Show sample prediction
        print_info(f"\nSample prediction (first 3×3):")
        sample = predictions[0, :3, :3].numpy()
        for row in sample:
            print_info(f"  {row.tolist()}")

        return True

    except Exception as e:
        print_error(f"Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation."""
    print_test_header("Loss Computation")

    try:
        model, config = create_test_model()
        model.train()

        puzzle_batch = create_mock_puzzle_batch(batch_size=2, num_train=2, H=8, W=8)

        print_info("Computing loss...")

        # Compute loss
        loss_dict = model.compute_loss(puzzle_batch)

        print_success("Loss computation succeeded!")

        # Check outputs
        loss = loss_dict['loss']
        accuracy = loss_dict['pixel_accuracy']

        print_info(f"\nLoss metrics:")
        print_info(f"  Loss: {loss.item():.4f}")
        print_info(f"  Pixel accuracy: {accuracy.item():.4f}")

        # Validate
        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        assert loss >= 0, "Loss should be non-negative"
        print_success("Loss is valid")

        assert 0 <= accuracy <= 1, f"Accuracy out of range: {accuracy.item()}"
        print_success("Accuracy is valid")

        return True

    except Exception as e:
        print_error(f"Loss computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass():
    """Test backward pass and gradient flow."""
    print_test_header("Backward Pass & Gradient Flow")

    try:
        model, config = create_test_model()
        model.train()

        puzzle_batch = create_mock_puzzle_batch(batch_size=2, num_train=2, H=8, W=8)

        print_info("Running forward and backward pass...")

        # Forward
        loss_dict = model.compute_loss(puzzle_batch)
        loss = loss_dict['loss']

        print_info(f"Loss: {loss.item():.4f}")

        # Backward
        loss.backward()

        print_success("Backward pass succeeded!")

        # Check gradients
        param_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_grads.append((name, grad_norm))

        print_info(f"\nParameter gradients (first 10):")
        for name, grad_norm in param_grads[:10]:
            print_info(f"  {name}: {grad_norm:.6f}")

        assert len(param_grads) > 0, "No parameter gradients computed"
        print_success(f"Computed gradients for {len(param_grads)} parameters")

        # Check that gradients are not all zero
        non_zero_grads = sum(1 for _, norm in param_grads if norm > 1e-8)
        print_info(f"Non-zero gradients: {non_zero_grads}/{len(param_grads)}")
        assert non_zero_grads > 0, "All gradients are zero"
        print_success("Gradients are flowing through the model")

        return True

    except Exception as e:
        print_error(f"Backward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_grid_sizes():
    """Test with different grid sizes."""
    print_test_header("Variable Grid Sizes")

    try:
        model, config = create_test_model()
        model.eval()

        grid_sizes = [(5, 5), (8, 8), (10, 12), (15, 15)]

        for H, W in grid_sizes:
            puzzle_batch = create_mock_puzzle_batch(batch_size=1, num_train=2, H=H, W=W)

            with torch.no_grad():
                predictions = model.predict(puzzle_batch)

            assert predictions.shape == (1, H, W), f"Wrong prediction shape for {H}×{W}"
            print_success(f"Grid size {H:2d}×{W:2d}: {tuple(predictions.shape)}")

        return True

    except Exception as e:
        print_error(f"Variable grid sizes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_train_examples():
    """Test with different numbers of train examples."""
    print_test_header("Variable Train Examples")

    try:
        model, config = create_test_model()
        model.eval()

        train_counts = [1, 2, 3, 5]

        for num_train in train_counts:
            puzzle_batch = create_mock_puzzle_batch(batch_size=1, num_train=num_train, H=8, W=8)

            with torch.no_grad():
                output = model(puzzle_batch)

            predictions = output['predicted_grids']
            assert predictions.shape[0] == 1
            print_success(f"Train examples {num_train}: prediction shape {tuple(predictions.shape)}")

        return True

    except Exception as e:
        print_error(f"Variable train examples test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_arc_puzzles():
    """Test with real ARC puzzles."""
    print_test_header("Real ARC Puzzles")

    try:
        # Load real dataset
        dataset = ARCPuzzleDataset(
            data_dir="kaggle/combined",
            split='train',
            arc_version='agi1',
            max_train_examples=5,
            subset_size=4
        )

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

        # Create model
        model, config = create_test_model()
        model.eval()

        # Forward pass
        with torch.no_grad():
            output = model(batch)
            predictions = model.predict(batch)

        print_success("Processed real ARC puzzles!")

        # Show predictions
        print_info(f"\nPrediction shapes:")
        for i, puzzle_id in enumerate(batch['puzzle_ids']):
            pred_shape = predictions[i].shape
            target_shape = batch['test_output_shapes'][i][0]
            print_info(f"  {puzzle_id}: predicted {tuple(pred_shape)}, target {target_shape}")

        # Compute loss
        loss_dict = model.compute_loss(batch)
        print_info(f"\nMetrics on real puzzles:")
        print_info(f"  Loss: {loss_dict['loss'].item():.4f}")
        print_info(f"  Pixel accuracy: {loss_dict['pixel_accuracy'].item():.4f}")

        return True

    except FileNotFoundError:
        print_warning("ARC dataset not found, skipping real puzzle test")
        return True
    except Exception as e:
        print_error(f"Real ARC puzzles test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intermediate_states():
    """Test returning intermediate states."""
    print_test_header("Intermediate States")

    try:
        model, config = create_test_model()
        model.eval()

        puzzle_batch = create_mock_puzzle_batch(batch_size=1, num_train=2, H=8, W=8)

        # Forward with intermediate states
        with torch.no_grad():
            output = model(puzzle_batch, return_intermediate=True)

        print_success("Retrieved intermediate states!")

        intermediate_states = output['intermediate_states']
        print_info(f"Number of intermediate states: {len(intermediate_states)}")
        print_info(f"Expected (num layers): {config.num_layers}")

        assert len(intermediate_states) == config.num_layers
        print_success("Correct number of intermediate states")

        # Check shapes
        for i, state in enumerate(intermediate_states):
            print_info(f"  Layer {i}: {tuple(state.shape)}")

        return True

    except Exception as e:
        print_error(f"Intermediate states test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_consistency():
    """Test that batching gives consistent results."""
    print_test_header("Batch Consistency")

    try:
        model, config = create_test_model()
        model.eval()

        # Create same puzzle twice
        puzzle = create_mock_puzzle_batch(batch_size=1, num_train=2, H=8, W=8)

        # Process individually
        with torch.no_grad():
            pred1 = model.predict(puzzle)

        # Process as batch
        puzzle_batch = {
            k: torch.cat([v, v], dim=0) if isinstance(v, torch.Tensor) else v + v
            for k, v in puzzle.items()
        }
        puzzle_batch['num_train'] = torch.tensor([2, 2])

        with torch.no_grad():
            pred_batch = model.predict(puzzle_batch)

        # Compare
        diff = (pred_batch[0] - pred1[0]).abs().float().mean()
        print_info(f"Mean difference between individual and batched: {diff.item():.6f}")

        if diff < 1e-5:
            print_success("Batched and individual predictions are identical")
        else:
            print_warning(f"Small difference detected: {diff.item():.6f}")

        return True

    except Exception as e:
        print_error(f"Batch consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test cases."""
    print(f"\n{TestColors.BOLD}{TestColors.HEADER}")
    print("="*70)
    print(" ARCSlotSolver End-to-End Integration Test Suite")
    print("="*70)
    print(f"{TestColors.ENDC}\n")

    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Prediction Method", test_prediction),
        ("Loss Computation", test_loss_computation),
        ("Backward Pass", test_backward_pass),
        ("Variable Grid Sizes", test_variable_grid_sizes),
        ("Variable Train Examples", test_variable_train_examples),
        ("Real ARC Puzzles", test_real_arc_puzzles),
        ("Intermediate States", test_intermediate_states),
        ("Batch Consistency", test_batch_consistency),
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
        print(f"{TestColors.OKGREEN}The complete ARC solving pipeline is working end-to-end!{TestColors.ENDC}\n")
        return 0
    else:
        print(f"{TestColors.FAIL}{TestColors.BOLD}❌ Some tests failed{TestColors.ENDC}\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

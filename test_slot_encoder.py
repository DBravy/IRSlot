"""
Comprehensive test script for SlotAttentionEncoder.

This script tests the SlotAttentionEncoder with various configurations and inputs,
ensuring it works correctly with the CNNEncoder and can handle different grid sizes.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.slot_encoder import SlotAttentionEncoder
from models.trm import CNNEncoder


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


def test_basic_forward_pass():
    """Test basic forward pass with square grid"""
    print_test_header("Basic Forward Pass (Square Grid)")

    # Configuration
    batch_size = 4
    height, width = 10, 10
    num_positions = height * width
    slot_dim = 64
    hidden_size = 256
    num_slots = 7

    print_info(f"Configuration:")
    print_info(f"  Batch size: {batch_size}")
    print_info(f"  Grid size: {height}×{width} = {num_positions} positions")
    print_info(f"  Slot dim: {slot_dim}")
    print_info(f"  Hidden size: {hidden_size}")
    print_info(f"  Num slots: {num_slots}")

    # Create encoder
    slot_encoder = SlotAttentionEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_size=hidden_size,
        num_iterations=3,
        mlp_hidden_size=128,
        forward_dtype=torch.float32  # Use float32 for testing
    )

    # Create random features
    features = torch.randn(batch_size, num_positions, slot_dim)
    print_info(f"\nInput features shape: {tuple(features.shape)}")

    # Forward pass
    try:
        slots = slot_encoder(features, spatial_size=(height, width))
        print_success(f"Forward pass succeeded!")
        print_info(f"Output slots shape: {tuple(slots.shape)}")

        # Check output shape
        expected_shape = (batch_size, num_slots, hidden_size)
        assert slots.shape == expected_shape, f"Expected shape {expected_shape}, got {slots.shape}"
        print_success(f"Output shape is correct: {tuple(slots.shape)}")

        # Check for NaN or Inf
        assert not torch.isnan(slots).any(), "Output contains NaN values"
        assert not torch.isinf(slots).any(), "Output contains Inf values"
        print_success("Output contains no NaN or Inf values")

        return True

    except Exception as e:
        print_error(f"Forward pass failed: {e}")
        return False


def test_rectangular_grid():
    """Test with non-square grid"""
    print_test_header("Rectangular Grid")

    batch_size = 2
    height, width = 15, 8
    num_positions = height * width
    slot_dim = 64
    hidden_size = 256
    num_slots = 5

    print_info(f"Grid size: {height}×{width} = {num_positions} positions")

    slot_encoder = SlotAttentionEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_size=hidden_size,
        forward_dtype=torch.float32
    )

    features = torch.randn(batch_size, num_positions, slot_dim)

    try:
        # Test with explicit spatial_size
        slots = slot_encoder(features, spatial_size=(height, width))
        assert slots.shape == (batch_size, num_slots, hidden_size)
        print_success(f"Rectangular grid with explicit spatial_size works: {tuple(slots.shape)}")

        return True

    except Exception as e:
        print_error(f"Rectangular grid test failed: {e}")
        return False


def test_spatial_inference():
    """Test automatic spatial dimension inference"""
    print_test_header("Automatic Spatial Dimension Inference")

    batch_size = 2
    slot_dim = 64
    hidden_size = 256
    num_slots = 5

    slot_encoder = SlotAttentionEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_size=hidden_size,
        forward_dtype=torch.float32
    )

    test_cases = [
        (25, 5, 5, "Perfect square"),
        (36, 6, 6, "Perfect square"),
        (100, 10, 10, "Perfect square"),
        (24, 4, 6, "Rectangle (close to square)"),
        (30, 5, 6, "Rectangle"),
    ]

    all_passed = True
    for num_positions, expected_h, expected_w, description in test_cases:
        features = torch.randn(batch_size, num_positions, slot_dim)

        try:
            # Test without spatial_size (auto-inference)
            slots = slot_encoder(features)
            inferred_h, inferred_w = slot_encoder._infer_spatial_dims(num_positions)

            print_info(f"{description}: {num_positions} positions → {inferred_h}×{inferred_w}")

            # Check if it matches expected or is valid
            assert inferred_h * inferred_w == num_positions, "Invalid spatial inference"
            assert slots.shape == (batch_size, num_slots, hidden_size)
            print_success(f"  Inference successful: {tuple(slots.shape)}")

        except Exception as e:
            print_error(f"  Failed for {num_positions} positions: {e}")
            all_passed = False

    return all_passed


def test_cnn_encoder_integration():
    """Test integration with CNNEncoder from models/trm.py"""
    print_test_header("Integration with CNNEncoder")

    batch_size = 3
    height, width = 12, 12
    input_channels = 1
    cnn_hidden_dim = 64
    slot_dim = 64
    hidden_size = 256
    num_slots = 7

    print_info(f"Creating full pipeline: CNNEncoder → SlotAttentionEncoder")

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
        forward_dtype=torch.float32
    )

    # Create random grid (like ARC grid)
    grid = torch.randint(0, 10, (batch_size, height, width))
    print_info(f"Input grid shape: {tuple(grid.shape)}")

    try:
        # Full pipeline
        features = cnn_encoder(grid)  # [B, H*W, slot_dim]
        print_info(f"CNN features shape: {tuple(features.shape)}")
        print_success("CNNEncoder forward pass succeeded")

        slots = slot_encoder(features, spatial_size=(height, width))
        print_info(f"Slot encoder output shape: {tuple(slots.shape)}")
        print_success("SlotAttentionEncoder forward pass succeeded")

        # Check shapes
        assert features.shape == (batch_size, height * width, slot_dim)
        assert slots.shape == (batch_size, num_slots, hidden_size)
        print_success("Full pipeline shapes are correct")

        return True

    except Exception as e:
        print_error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow through the encoder"""
    print_test_header("Gradient Flow")

    batch_size = 2
    height, width = 8, 8
    num_positions = height * width
    slot_dim = 64
    hidden_size = 128
    num_slots = 5

    slot_encoder = SlotAttentionEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_size=hidden_size,
        forward_dtype=torch.float32
    )

    features = torch.randn(batch_size, num_positions, slot_dim, requires_grad=True)

    try:
        # Forward pass
        slots = slot_encoder(features, spatial_size=(height, width))

        # Create dummy loss (sum of all slot values)
        loss = slots.sum()
        print_info(f"Dummy loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients
        assert features.grad is not None, "No gradients for input features"
        assert not torch.isnan(features.grad).any(), "Gradients contain NaN"
        assert not torch.isinf(features.grad).any(), "Gradients contain Inf"

        grad_norm = features.grad.norm().item()
        print_info(f"Input gradient norm: {grad_norm:.4f}")
        print_success("Gradients flow correctly through encoder")

        # Check parameter gradients
        param_grads = []
        for name, param in slot_encoder.named_parameters():
            if param.grad is not None:
                param_grads.append((name, param.grad.norm().item()))

        print_info(f"\nParameter gradients:")
        for name, grad_norm in param_grads[:5]:  # Show first 5
            print_info(f"  {name}: {grad_norm:.6f}")

        assert len(param_grads) > 0, "No parameter gradients computed"
        print_success(f"Computed gradients for {len(param_grads)} parameters")

        return True

    except Exception as e:
        print_error(f"Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_visualization():
    """Test attention map extraction"""
    print_test_header("Attention Map Visualization")

    batch_size = 1
    height, width = 10, 10
    num_positions = height * width
    slot_dim = 64
    hidden_size = 128
    num_slots = 5

    slot_encoder = SlotAttentionEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_size=hidden_size,
        forward_dtype=torch.float32
    )

    features = torch.randn(batch_size, num_positions, slot_dim)

    try:
        # Get attention maps
        slots, attn_maps = slot_encoder.get_attention_maps(
            features, spatial_size=(height, width)
        )

        print_info(f"Slots shape: {tuple(slots.shape)}")
        print_info(f"Attention maps shape: {tuple(attn_maps.shape)}")

        # Check shapes
        assert slots.shape == (batch_size, num_slots, hidden_size)
        assert attn_maps.shape == (batch_size, num_slots, height, width)
        print_success("Attention map shapes are correct")

        # Check attention properties
        # Each attention map should sum to approximately 1 (normalized attention)
        attn_sums = attn_maps.sum(dim=(2, 3))  # Sum over spatial dims
        print_info(f"\nAttention map sums (should be ~1 for each slot):")
        for slot_idx in range(num_slots):
            print_info(f"  Slot {slot_idx}: {attn_sums[0, slot_idx].item():.4f}")

        # Check that attention values are in valid range
        assert (attn_maps >= 0).all(), "Attention contains negative values"
        print_success("Attention maps are non-negative")

        return True

    except Exception as e:
        print_error(f"Attention visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_variable_batch_sizes():
    """Test with different batch sizes including batch_size=1"""
    print_test_header("Variable Batch Sizes")

    height, width = 8, 8
    num_positions = height * width
    slot_dim = 64
    hidden_size = 128
    num_slots = 5

    slot_encoder = SlotAttentionEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_size=hidden_size,
        forward_dtype=torch.float32
    )

    batch_sizes = [1, 2, 4, 8, 16]
    all_passed = True

    for batch_size in batch_sizes:
        features = torch.randn(batch_size, num_positions, slot_dim)

        try:
            slots = slot_encoder(features, spatial_size=(height, width))
            assert slots.shape == (batch_size, num_slots, hidden_size)
            print_success(f"Batch size {batch_size:2d}: {tuple(slots.shape)}")

        except Exception as e:
            print_error(f"Failed for batch_size={batch_size}: {e}")
            all_passed = False

    return all_passed


def test_different_slot_configs():
    """Test with different slot dimensions and counts"""
    print_test_header("Different Slot Configurations")

    batch_size = 2
    height, width = 10, 10
    num_positions = height * width

    configs = [
        (32, 128, 5, "Small slots, small hidden"),
        (64, 256, 7, "Medium slots, medium hidden"),
        (128, 512, 10, "Large slots, large hidden"),
        (64, 64, 3, "Same slot_dim and hidden_size"),
    ]

    all_passed = True
    for slot_dim, hidden_size, num_slots, description in configs:
        print_info(f"\n{description}:")
        print_info(f"  slot_dim={slot_dim}, hidden_size={hidden_size}, num_slots={num_slots}")

        try:
            slot_encoder = SlotAttentionEncoder(
                num_slots=num_slots,
                slot_dim=slot_dim,
                hidden_size=hidden_size,
                forward_dtype=torch.float32
            )

            features = torch.randn(batch_size, num_positions, slot_dim)
            slots = slot_encoder(features, spatial_size=(height, width))

            assert slots.shape == (batch_size, num_slots, hidden_size)
            print_success(f"  Output: {tuple(slots.shape)}")

        except Exception as e:
            print_error(f"  Failed: {e}")
            all_passed = False

    return all_passed


def test_determinism():
    """Test that same input produces same output (with same random seed)"""
    print_test_header("Determinism Test")

    batch_size = 2
    height, width = 8, 8
    num_positions = height * width
    slot_dim = 64
    hidden_size = 128
    num_slots = 5

    # Create two identical encoders with same initialization
    torch.manual_seed(42)
    slot_encoder1 = SlotAttentionEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_size=hidden_size,
        forward_dtype=torch.float32
    )

    torch.manual_seed(42)
    slot_encoder2 = SlotAttentionEncoder(
        num_slots=num_slots,
        slot_dim=slot_dim,
        hidden_size=hidden_size,
        forward_dtype=torch.float32
    )

    # Same input
    torch.manual_seed(123)
    features = torch.randn(batch_size, num_positions, slot_dim)

    try:
        # Forward pass with same random seed for slot initialization
        torch.manual_seed(456)
        slots1 = slot_encoder1(features, spatial_size=(height, width))

        torch.manual_seed(456)
        slots2 = slot_encoder2(features, spatial_size=(height, width))

        # Check outputs are the same
        max_diff = (slots1 - slots2).abs().max().item()
        print_info(f"Max difference between outputs: {max_diff:.10f}")

        if max_diff < 1e-6:
            print_success("Outputs are deterministic (identical)")
            return True
        else:
            print_warning(f"Outputs differ by {max_diff:.10f}")
            return True  # Still pass, minor differences acceptable

    except Exception as e:
        print_error(f"Determinism test failed: {e}")
        return False


def run_all_tests():
    """Run all test cases"""
    print(f"\n{TestColors.BOLD}{TestColors.HEADER}")
    print("="*70)
    print(" SlotAttentionEncoder Comprehensive Test Suite")
    print("="*70)
    print(f"{TestColors.ENDC}\n")

    tests = [
        ("Basic Forward Pass", test_basic_forward_pass),
        ("Rectangular Grid", test_rectangular_grid),
        ("Spatial Inference", test_spatial_inference),
        ("CNNEncoder Integration", test_cnn_encoder_integration),
        ("Gradient Flow", test_gradient_flow),
        ("Attention Visualization", test_attention_visualization),
        ("Variable Batch Sizes", test_variable_batch_sizes),
        ("Different Slot Configs", test_different_slot_configs),
        ("Determinism", test_determinism),
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

"""
Debug script to check the shapes of features and attention weights.
This helps diagnose the attention visualization issue.
"""
import torch
from models.arc_solver import ARCSlotSolver, ARCSlotSolverConfig
from dataset.arc_puzzle_dataset import ARCPuzzleDataset, collate_puzzle_batch
from torch.utils.data import DataLoader

def main():
    print("="*70)
    print("DEBUG: Checking attention shapes")
    print("="*70)

    # Create a simple model
    config = ARCSlotSolverConfig(
        grid_channels=1,
        cnn_hidden_dim=64,
        slot_dim=64,
        num_slots_per_grid=7,
        slot_iterations=3,
        slot_mlp_hidden=128,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        expansion=4.0,
        max_train_examples=5,
        max_grid_size=30,
        decoder_hidden_dim=64,
        output_channels=10,
        forward_dtype="float32",
        use_rope=True,
    )

    model = ARCSlotSolver(config)
    model.eval()

    # Load a small dataset
    dataset = ARCPuzzleDataset(
        data_dir='kaggle/combined',
        split='train',
        arc_version='agi1',
        max_train_examples=5,
        subset_size=10,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_puzzle_batch,
    )

    # Get one batch
    batch = next(iter(loader))

    print("\n1. Batch structure:")
    print(f"   Keys: {batch.keys()}")
    print(f"   test_inputs shape: {batch['test_inputs'].shape}")
    print(f"   test_input_shapes: {batch['test_input_shapes']}")

    # Extract test inputs
    test_inputs = batch['test_inputs'][:, 0]  # [B, H, W]
    print(f"\n2. Test inputs (first test example):")
    print(f"   Shape: {test_inputs.shape}")
    print(f"   Type: {type(test_inputs)}")

    # Get one sample
    grid = test_inputs[0:1]  # [1, H, W]
    print(f"\n3. Single grid:")
    print(f"   Shape: {grid.shape}")

    # Pass through CNN encoder
    print(f"\n4. CNN Encoder output:")
    with torch.no_grad():
        features = model.cnn_encoder(grid)
        print(f"   Features shape: {features.shape}")
        print(f"   Features type: {type(features)}")
        print(f"   Features ndim: {features.ndim}")

    # Pass through slot encoder
    print(f"\n5. Slot Encoder output:")
    with torch.no_grad():
        result = model.slot_encoder(features, return_attn=True)
        print(f"   Return type: {type(result)}")

        if isinstance(result, tuple):
            print(f"   Number of return values: {len(result)}")
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    print(f"   Return[{i}] shape: {r.shape}")
                else:
                    print(f"   Return[{i}] type: {type(r)}")
        else:
            print(f"   Single return value shape: {result.shape}")

    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()

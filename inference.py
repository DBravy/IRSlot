"""
Inference script for testing trained slot attention model.
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import SlotInstanceModel
from dataset.arc_dataset import ARCInstanceDataset, collate_fn_pad


def parse_args():
    parser = argparse.ArgumentParser(description='Test trained Slot Attention model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed dataset directory')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use (default: train)')
    parser.add_argument('--subset', type=str, default='all',
                        help='Dataset subset to use (default: all)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process (default: 10)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable augmentation during inference')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Slot Attention Inference")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    checkpoint_args = checkpoint['args']

    # Create model
    print("Creating model...")
    model = SlotInstanceModel(
        num_colors=10,
        encoder_feature_dim=checkpoint_args['encoder_feature_dim'],
        encoder_hidden_dim=checkpoint_args['encoder_hidden_dim'],
        num_slots=checkpoint_args['num_slots'],
        slot_dim=checkpoint_args['slot_dim'],
        num_iterations=checkpoint_args['num_iterations'],
        embedding_dim=checkpoint_args['embedding_dim'],
        max_grid_size=30
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    print(f"Training accuracy: {checkpoint['accuracy']:.4f}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = ARCInstanceDataset(
        data_dir=args.data_dir,
        split=args.split,
        subset=args.subset,
        augment=not args.no_augment,
        max_grid_size=30
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_pad
    )

    print(f"Dataset size: {len(dataset)}")
    print()

    # Process samples
    print("Processing samples...")
    print("=" * 80)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_samples:
                break

            grid_ids = batch['grid_ids'].to(args.device)
            grids = batch['grids'].to(args.device)
            original_shapes = batch['original_shapes']

            # Forward pass
            embeddings, slots = model(grids)

            # Print results
            print(f"\nSample {i + 1}:")
            print(f"  Grid ID: {grid_ids[0].item()}")
            print(f"  Original shape: {original_shapes[0]}")
            print(f"  Grid shape: {grids[0].shape}")
            print(f"  Embedding shape: {embeddings[0].shape}")
            print(f"  Slots shape: {slots[0].shape}")
            print(f"  Embedding norm: {embeddings[0].norm().item():.4f}")

            # Print slot statistics
            slot_norms = slots[0].norm(dim=1)
            print(f"  Slot norms: min={slot_norms.min().item():.4f}, "
                  f"max={slot_norms.max().item():.4f}, "
                  f"mean={slot_norms.mean().item():.4f}")

            # Print grid visualization (first 10x10 only)
            grid_np = grids[0].cpu().numpy()
            H, W = min(10, grid_np.shape[0]), min(10, grid_np.shape[1])
            print(f"  Grid preview ({H}x{W}):")
            for row in range(H):
                print("    " + " ".join(str(grid_np[row, col]) for col in range(W)))

    print()
    print("=" * 80)
    print("Inference completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

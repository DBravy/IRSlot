"""
Supervised training script for slot attention with ground truth mask supervision.

Uses connectivity-based object segmentation as ground truth to teach slots
to segment objects. Slot 0 learns background, slots 1+ learn foreground objects.
"""
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SlotInstanceModel
from mask_supervision import MaskSupervisionLoss
from dataset.arc_dataset import ARCInstanceDataset, collate_fn_pad


def parse_args():
    parser = argparse.ArgumentParser(description='Train Slot Attention with Mask Supervision')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed dataset directory')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use (default: train)')
    parser.add_argument('--subset', type=str, default='all',
                        help='Dataset subset to use (default: all)')

    # Model architecture
    parser.add_argument('--num_slots', type=int, default=7,
                        help='Number of slots (default: 7). Slot 0 = background.')
    parser.add_argument('--slot_dim', type=int, default=64,
                        help='Slot dimension (default: 64)')
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='Number of slot attention iterations (default: 5)')
    parser.add_argument('--encoder_feature_dim', type=int, default=64,
                        help='Encoder feature dimension (default: 64)')
    parser.add_argument('--encoder_hidden_dim', type=int, default=128,
                        help='Encoder hidden dimension (default: 128)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Final embedding dimension (default: 128)')
    parser.add_argument('--hard_attention', action='store_true',
                        help='Use Gumbel-Softmax for hard attention (default: False)')
    parser.add_argument('--gumbel_temperature', type=float, default=1.0,
                        help='Gumbel-Softmax temperature (default: 1.0)')

    # Loss weights
    parser.add_argument('--bg_weight', type=float, default=1.0,
                        help='Weight for background slot loss (default: 1.0)')
    parser.add_argument('--fg_weight', type=float, default=1.0,
                        help='Weight for foreground slot losses (default: 1.0)')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_supervised',
                        help='Directory to save checkpoints (default: checkpoints_supervised)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available, else cpu)')

    return parser.parse_args()


def train_epoch(model, criterion, dataloader, optimizer, device):
    """
    Train for one epoch with mask supervision.
    """
    model.train()
    total_loss = 0.0
    total_bg_loss = 0.0
    total_fg_loss = 0.0
    total_matched = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        grids = batch['grids'].to(device)
        gt_masks = batch['masks'].to(device)
        num_objects = batch['num_objects'].to(device)

        # Forward pass with attention weights
        embeddings, slots, attn_weights = model(grids, return_attn=True)

        # Compute mask supervision loss
        loss, metrics = criterion(attn_weights, gt_masks, num_objects)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += metrics['loss']
        total_bg_loss += metrics['bg_loss']
        total_fg_loss += metrics['fg_loss']
        total_matched += metrics['num_matched']
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'bg': f"{metrics['bg_loss']:.4f}",
            'fg': f"{metrics['fg_loss']:.4f}",
            'matched': metrics['num_matched']
        })

    avg_loss = total_loss / num_batches
    avg_bg_loss = total_bg_loss / num_batches
    avg_fg_loss = total_fg_loss / num_batches
    avg_matched = total_matched / num_batches

    return {
        'loss': avg_loss,
        'bg_loss': avg_bg_loss,
        'fg_loss': avg_fg_loss,
        'avg_matched': avg_matched
    }


@torch.no_grad()
def validate_epoch(model, criterion, dataloader, device):
    """
    Validate for one epoch.
    """
    model.eval()
    total_loss = 0.0
    total_bg_loss = 0.0
    total_fg_loss = 0.0
    total_matched = 0
    num_batches = 0

    for batch in dataloader:
        grids = batch['grids'].to(device)
        gt_masks = batch['masks'].to(device)
        num_objects = batch['num_objects'].to(device)

        # Forward pass with attention weights
        embeddings, slots, attn_weights = model(grids, return_attn=True)

        # Compute mask supervision loss
        loss, metrics = criterion(attn_weights, gt_masks, num_objects)

        # Track metrics
        total_loss += metrics['loss']
        total_bg_loss += metrics['bg_loss']
        total_fg_loss += metrics['fg_loss']
        total_matched += metrics['num_matched']
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_bg_loss = total_bg_loss / num_batches
    avg_fg_loss = total_fg_loss / num_batches
    avg_matched = total_matched / num_batches

    return {
        'loss': avg_loss,
        'bg_loss': avg_bg_loss,
        'fg_loss': avg_fg_loss,
        'avg_matched': avg_matched
    }


def main():
    args = parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.checkpoint_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("Training Slot Attention with Mask Supervision")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Number of slots: {args.num_slots} (slot 0 = background)")
    print()

    # Load dataset with mask generation enabled
    print("Loading dataset...")
    dataset = ARCInstanceDataset(
        data_dir=args.data_dir,
        split=args.split,
        subset=args.subset,
        augment=True,
        max_grid_size=30,
        num_slots=args.num_slots,
        return_masks=True  # Enable mask generation
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=True if args.device == 'cuda' else False
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches per epoch: {len(dataloader)}")
    print()

    # Create model
    print("Creating model...")
    model = SlotInstanceModel(
        num_colors=10,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        num_iterations=args.num_iterations,
        embedding_dim=args.embedding_dim,
        max_grid_size=30,
        hard_attention=args.hard_attention,
        gumbel_temperature=args.gumbel_temperature
    ).to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create loss function
    criterion = MaskSupervisionLoss(
        bg_weight=args.bg_weight,
        fg_weight=args.fg_weight
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    best_loss = float('inf')

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, best loss: {best_loss:.4f}")
        print()

    # Training loop
    print("Starting training...")
    print("=" * 80)
    print()

    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")

        # Train
        train_metrics = train_epoch(
            model=model,
            criterion=criterion,
            dataloader=dataloader,
            optimizer=optimizer,
            device=args.device
        )

        print(f"  Loss: {train_metrics['loss']:.4f} (bg: {train_metrics['bg_loss']:.4f}, fg: {train_metrics['fg_loss']:.4f})")
        print(f"  Avg matched objects: {train_metrics['avg_matched']:.1f}")
        print()

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_metrics['loss'],
                'best_loss': best_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_metrics['loss'],
                'best_loss': best_loss,
                'args': vars(args)
            }, best_path)
            print(f"  New best model! Saved to: {best_path}")

        print()

    print("=" * 80)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()

"""
Training script for ARCSlotSolver.

Trains the slot-based ARC puzzle solver with transformer reasoning.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
from tqdm import tqdm
import time

from models.arc_solver import ARCSlotSolver, ARCSlotSolverConfig
from dataset.arc_puzzle_dataset import ARCPuzzleDataset, collate_puzzle_batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ARCSlotSolver')

    # Data
    parser.add_argument('--data_dir', type=str, default='kaggle/combined',
                        help='Directory containing ARC JSON files')
    parser.add_argument('--arc_version', type=str, default='agi1', choices=['agi1', 'agi2'],
                        help='ARC dataset version')
    parser.add_argument('--max_train_examples', type=int, default=5,
                        help='Maximum number of train examples per puzzle')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Limit dataset size (for debugging)')

    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Transformer hidden size')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_slots', type=int, default=7,
                        help='Number of slots per grid')
    parser.add_argument('--slot_dim', type=int, default=64,
                        help='Dimension of slots before projection')
    parser.add_argument('--slot_iterations', type=int, default=3,
                        help='Number of slot attention iterations')

    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_arc_solver',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    # Logging
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log metrics every N batches')

    # Evaluation
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Run detailed evaluation every N epochs')

    return parser.parse_args()


def create_model(args):
    """Create ARCSlotSolver model."""
    config = ARCSlotSolverConfig(
        grid_channels=1,
        cnn_hidden_dim=64,
        slot_dim=args.slot_dim,
        num_slots_per_grid=args.num_slots,
        slot_iterations=args.slot_iterations,
        slot_mlp_hidden=128,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        expansion=4.0,
        max_train_examples=args.max_train_examples,
        max_grid_size=30,
        decoder_hidden_dim=64,
        output_channels=10,
        forward_dtype="float32",
        use_rope=True,
    )

    model = ARCSlotSolver(config)
    return model, config


def create_datasets(args):
    """Create train and validation datasets."""
    # Training set
    train_dataset = ARCPuzzleDataset(
        data_dir=args.data_dir,
        split='train',
        arc_version=args.arc_version,
        max_train_examples=args.max_train_examples,
        subset_size=args.subset_size,
    )

    # Validation set (use eval split)
    val_dataset = ARCPuzzleDataset(
        data_dir=args.data_dir,
        split='eval',
        arc_version=args.arc_version,
        max_train_examples=args.max_train_examples,
        subset_size=args.subset_size // 4 if args.subset_size else None,
    )

    return train_dataset, val_dataset


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}')

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        loss_dict = model.compute_loss(batch)
        loss = loss_dict['loss']
        accuracy = loss_dict['pixel_accuracy']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1

        # Update progress bar
        if batch_idx % args.log_every == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate the model (quick version for every epoch)."""
    model.eval()

    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc='Validation'):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        loss_dict = model.compute_loss(batch)

        total_loss += loss_dict['loss'].item()
        total_accuracy += loss_dict['pixel_accuracy'].item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


@torch.no_grad()
def detailed_evaluation(model, dataloader, device, epoch):
    """
    Detailed evaluation with per-puzzle metrics.

    Computes:
    - Pixel accuracy (per-pixel correctness)
    - Exact match accuracy (entire grid must match)
    - Per-puzzle statistics

    Weights are frozen (@torch.no_grad()).
    """
    model.eval()

    results = {
        'epoch': epoch,
        'pixel_accuracy': 0.0,
        'exact_match_accuracy': 0.0,
        'total_puzzles': 0,
        'exact_matches': 0,
        'per_puzzle': []
    }

    total_pixels_correct = 0
    total_pixels = 0
    exact_matches = 0
    total_puzzles = 0

    print(f"\n{'='*70}")
    print(f" Detailed Evaluation - Epoch {epoch+1}")
    print(f"{'='*70}\n")

    for batch in tqdm(dataloader, desc='Detailed Eval'):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Get predictions
        predictions = model.predict(batch)  # [batch, H, W]

        # Get ground truth
        test_outputs = batch['test_outputs'][:, 0]  # [batch, H, W]
        test_available = batch['test_output_available'][:, 0]  # [batch]
        test_shapes = batch['test_output_shapes']  # List of lists

        batch_size = predictions.shape[0]

        for b in range(batch_size):
            if not test_available[b]:
                continue

            puzzle_id = batch['puzzle_ids'][b]
            target_shape = test_shapes[b][0]

            if target_shape is None:
                continue

            H, W = target_shape
            pred = predictions[b, :H, :W]
            target = test_outputs[b, :H, :W]

            # Pixel accuracy
            correct_pixels = (pred == target).sum().item()
            total_pixels_this = H * W
            pixel_acc = correct_pixels / total_pixels_this

            # Exact match
            exact_match = (pred == target).all().item()

            # Accumulate
            total_pixels_correct += correct_pixels
            total_pixels += total_pixels_this
            if exact_match:
                exact_matches += 1
            total_puzzles += 1

            # Store per-puzzle result
            puzzle_result = {
                'puzzle_id': puzzle_id,
                'pixel_accuracy': pixel_acc,
                'exact_match': bool(exact_match),
                'grid_size': f'{H}x{W}'
            }
            results['per_puzzle'].append(puzzle_result)

    # Compute overall metrics
    results['pixel_accuracy'] = total_pixels_correct / total_pixels if total_pixels > 0 else 0.0
    results['exact_match_accuracy'] = exact_matches / total_puzzles if total_puzzles > 0 else 0.0
    results['total_puzzles'] = total_puzzles
    results['exact_matches'] = exact_matches

    # Print summary
    print(f"\nEvaluation Results:")
    print(f"  Total puzzles: {total_puzzles}")
    print(f"  Exact matches: {exact_matches} ({results['exact_match_accuracy']*100:.2f}%)")
    print(f"  Pixel accuracy: {results['pixel_accuracy']*100:.2f}%")
    print(f"\nSample results:")
    for i, puzzle in enumerate(results['per_puzzle'][:5]):
        status = "✓" if puzzle['exact_match'] else "✗"
        print(f"  {status} {puzzle['puzzle_id']}: {puzzle['pixel_accuracy']*100:.1f}% ({puzzle['grid_size']})")

    return results


def save_checkpoint(model, optimizer, scheduler, epoch, args, metrics, checkpoint_dir):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'args': vars(args),
    }

    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
    torch.save(checkpoint, checkpoint_path)

    # Also save as latest
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)

    print(f'Saved checkpoint to {checkpoint_path}')


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']

    print(f'Loaded checkpoint from epoch {epoch+1}')
    print(f'Metrics: {metrics}')

    return epoch + 1


def main():
    """Main training function."""
    args = parse_args()

    print("="*70)
    print(" ARCSlotSolver Training")
    print("="*70)
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Save args
    with open(checkpoint_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset, val_dataset = create_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_puzzle_batch,
        num_workers=0,  # Set to 0 for debugging, increase for speed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_puzzle_batch,
        num_workers=0,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model, config = create_model(args)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }

    # Evaluation history (periodic detailed evaluations)
    evaluation_results = []

    # Training loop
    print("\nStarting training...")
    best_val_accuracy = 0
    best_exact_match = 0

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()

        # Train
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args
        )

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start_time

        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.num_epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Detailed evaluation (periodic)
        if (epoch + 1) % args.eval_every == 0:
            eval_results = detailed_evaluation(model, val_loader, device, epoch)
            evaluation_results.append(eval_results)

            # Save evaluation results
            with open(checkpoint_dir / 'evaluation_results.json', 'w') as f:
                json.dump(evaluation_results, f, indent=2)

            # Track best exact match model
            if eval_results['exact_match_accuracy'] > best_exact_match:
                best_exact_match = eval_results['exact_match_accuracy']
                best_exact_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': {
                        'exact_match_accuracy': eval_results['exact_match_accuracy'],
                        'pixel_accuracy': eval_results['pixel_accuracy'],
                    },
                    'args': vars(args),
                }
                torch.save(best_exact_checkpoint, checkpoint_dir / 'checkpoint_best_exact_match.pt')
                print(f"  🎯 New best exact match! {best_exact_match*100:.2f}%")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }
            save_checkpoint(model, optimizer, scheduler, epoch, args, metrics, checkpoint_dir)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': {
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                },
                'args': vars(args),
            }
            torch.save(best_checkpoint, checkpoint_dir / 'checkpoint_best.pt')
            print(f"  New best model! Val Acc: {val_accuracy:.4f}")

        # Save history
        with open(checkpoint_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print(" Training Complete!")
    print("="*70)
    print(f"\nBest Metrics:")
    print(f"  Validation pixel accuracy: {best_val_accuracy:.4f}")
    print(f"  Exact match accuracy: {best_exact_match*100:.2f}%")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"  checkpoint_best.pt - Best validation pixel accuracy")
    print(f"  checkpoint_best_exact_match.pt - Best exact match accuracy")
    print(f"\nEvaluation results saved to: {checkpoint_dir}/evaluation_results.json")


if __name__ == '__main__':
    main()

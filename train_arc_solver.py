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
from dataset.grid_dataset import GridDataset, collate_grid_batch
from memory_bank import MemoryBank


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
    parser.add_argument('--augmentations_per_puzzle', type=int, default=2,
                        help='Number of augmented views per puzzle when contrastive loss is enabled')

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

    # Multi-task learning (Contrastive)
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Enable multi-task learning with contrastive loss')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                        help='Weight for contrastive loss')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--contrastive_num_negatives', type=int, default=512,
                        help='Number of negative samples for contrastive learning')
    parser.add_argument('--memory_bank_momentum', type=float, default=0.5,
                        help='Momentum for memory bank updates')

    # Interleaved training (instance recognition vs ARC solving)
    parser.add_argument('--instance_to_arc_ratio', type=int, default=10,
                        help='Number of instance recognition batches per 1 ARC solver batch')

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
        # Contrastive learning
        use_contrastive_loss=args.use_contrastive,
        contrastive_embedding_dim=128,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_num_negatives=args.contrastive_num_negatives,
        contrastive_loss_weight=args.contrastive_weight,
    )

    model = ARCSlotSolver(config)
    return model, config


def create_datasets(args):
    """Create train and validation datasets."""
    # ARC Puzzle Dataset (for puzzle solving task)
    # No augmentation here - augmentation is handled in GridDataset for contrastive learning
    train_dataset = ARCPuzzleDataset(
        data_dir=args.data_dir,
        split='train',
        arc_version=args.arc_version,
        max_train_examples=args.max_train_examples,
        subset_size=args.subset_size,
        augment=False,  # No augmentation for ARC solving
        augmentations_per_puzzle=1,
    )

    # Validation set (use eval split)
    val_dataset = ARCPuzzleDataset(
        data_dir=args.data_dir,
        split='eval',
        arc_version=args.arc_version,
        max_train_examples=args.max_train_examples,
        subset_size=args.subset_size // 4 if args.subset_size else None,
        augment=False,
    )

    # Grid Dataset (for instance recognition / contrastive learning)
    grid_train_dataset = None
    if args.use_contrastive:
        grid_train_dataset = GridDataset(
            data_dir=args.data_dir,
            split='train',
            arc_version=args.arc_version,
            augment=True,  # Apply augmentation for contrastive learning
            subset_size=args.subset_size,
        )

    return train_dataset, val_dataset, grid_train_dataset


def train_epoch(model, arc_dataloader, grid_dataloader, optimizer, scheduler, device, epoch, args, memory_bank=None):
    """
    Train for one epoch with interleaved instance recognition and ARC solving.

    If use_contrastive is enabled, alternates between:
    - N instance recognition batches (contrastive loss on individual grids)
    - 1 ARC solver batch (puzzle solving loss)

    Otherwise, just trains on ARC solving batches.
    """
    model.train()

    total_loss = 0
    total_arc_loss = 0
    total_contrastive_loss = 0
    total_accuracy = 0
    total_contrastive_accuracy = 0
    total_pos_sim = 0
    total_neg_sim = 0
    num_arc_batches = 0
    num_instance_batches = 0

    # Create iterators
    arc_iter = iter(arc_dataloader)
    grid_iter = iter(grid_dataloader) if grid_dataloader is not None else None

    # Calculate total steps for progress bar
    total_steps = len(arc_dataloader)
    if args.use_contrastive and grid_dataloader is not None:
        # Each ARC batch is preceded by N instance recognition batches
        total_steps = len(arc_dataloader) * (1 + args.instance_to_arc_ratio)

    pbar = tqdm(total=total_steps, desc=f'Epoch {epoch+1}/{args.num_epochs}')

    arc_exhausted = False

    while not arc_exhausted:
        # ============================================
        # Phase 1: Instance Recognition (Contrastive)
        # ============================================
        if args.use_contrastive and grid_iter is not None:
            for _ in range(args.instance_to_arc_ratio):
                try:
                    grid_batch = next(grid_iter)
                except StopIteration:
                    # Grid dataset exhausted, restart iterator
                    grid_iter = iter(grid_dataloader)
                    grid_batch = next(grid_iter)

                # Move batch to device
                grid_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in grid_batch.items()}

                # Compute contrastive loss on individual grids
                loss_dict = model.compute_contrastive_loss_on_grids(grid_batch, memory_bank)
                loss = loss_dict['loss']

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()

                # Accumulate metrics
                total_contrastive_loss += loss.item()
                total_contrastive_accuracy += loss_dict['accuracy'].item()
                total_pos_sim += loss_dict['avg_positive_sim'].item()
                total_neg_sim += loss_dict['avg_negative_sim'].item()
                num_instance_batches += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'mode': 'instance',
                    'cont_loss': f'{loss.item():.4f}',
                    'cont_acc': f'{loss_dict["accuracy"].item():.3f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })

        # ============================================
        # Phase 2: ARC Puzzle Solving
        # ============================================
        try:
            arc_batch = next(arc_iter)
        except StopIteration:
            # ARC dataset exhausted, end epoch
            arc_exhausted = True
            break

        # Move batch to device
        arc_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in arc_batch.items()}

        # Forward pass (ARC solving only, no contrastive)
        loss_dict = model.compute_loss(arc_batch, memory_bank=None)
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
        total_arc_loss += loss_dict['arc_loss'].item()
        total_accuracy += accuracy.item()
        num_arc_batches += 1

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'mode': 'ARC',
            'arc_loss': f'{loss.item():.4f}',
            'acc': f'{accuracy.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    pbar.close()

    # Compute averages
    avg_arc_loss = total_arc_loss / num_arc_batches if num_arc_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_arc_batches if num_arc_batches > 0 else 0.0
    avg_contrastive_loss = total_contrastive_loss / num_instance_batches if num_instance_batches > 0 else 0.0
    avg_contrastive_accuracy = total_contrastive_accuracy / num_instance_batches if num_instance_batches > 0 else 0.0
    avg_pos_sim = total_pos_sim / num_instance_batches if num_instance_batches > 0 else 0.0
    avg_neg_sim = total_neg_sim / num_instance_batches if num_instance_batches > 0 else 0.0

    # Total loss is weighted average
    if num_arc_batches > 0 and num_instance_batches > 0:
        avg_loss = (total_arc_loss + args.contrastive_weight * total_contrastive_loss) / (num_arc_batches + num_instance_batches)
    elif num_arc_batches > 0:
        avg_loss = avg_arc_loss
    else:
        avg_loss = avg_contrastive_loss

    return {
        'loss': avg_loss,
        'arc_loss': avg_arc_loss,
        'contrastive_loss': avg_contrastive_loss,
        'accuracy': avg_accuracy,
        'contrastive_accuracy': avg_contrastive_accuracy,
        'avg_positive_sim': avg_pos_sim,
        'avg_negative_sim': avg_neg_sim,
    }


@torch.no_grad()
def validate(model, dataloader, device, memory_bank=None):
    """Validate the model (quick version for every epoch)."""
    model.eval()

    total_loss = 0
    total_arc_loss = 0
    total_accuracy = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc='Validation'):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass (no memory bank update during validation)
        loss_dict = model.compute_loss(batch, memory_bank=None)

        total_loss += loss_dict['loss'].item()
        total_arc_loss += loss_dict['arc_loss'].item()
        total_accuracy += loss_dict['pixel_accuracy'].item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_arc_loss = total_arc_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return {
        'loss': avg_loss,
        'arc_loss': avg_arc_loss,
        'accuracy': avg_accuracy,
    }


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


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, memory_bank=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if memory_bank is not None and 'memory_bank_state_dict' in checkpoint:
        memory_bank.load_state_dict(checkpoint['memory_bank_state_dict'])
        print('Loaded memory bank from checkpoint')

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
    train_dataset, val_dataset, grid_train_dataset = create_datasets(args)

    # Create dataloaders
    # ARC puzzle dataloader (always use simple collate, no puzzle_id mapping needed)
    arc_train_loader = DataLoader(
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

    # Grid dataloader for instance recognition (if contrastive learning enabled)
    grid_train_loader = None
    if args.use_contrastive and grid_train_dataset is not None:
        grid_train_loader = DataLoader(
            grid_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_grid_batch,
            num_workers=0,
        )
        print(f"ARC Train batches: {len(arc_train_loader)}")
        print(f"Grid Train batches: {len(grid_train_loader)}")
        print(f"Instance-to-ARC ratio: {args.instance_to_arc_ratio}:1")
    else:
        print(f"Train batches: {len(arc_train_loader)}")

    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model, config = create_model(args)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create memory bank for contrastive learning (if enabled)
    memory_bank = None
    if args.use_contrastive:
        print("\nEnabling multi-task learning with contrastive loss...")
        # Get number of unique GRIDS for memory bank sizing
        # GridDataset assigns a unique index to each individual grid
        num_unique_grids = grid_train_dataset.num_unique_grids()
        num_unique_puzzles = grid_train_dataset.num_unique_puzzles()
        print(f"Creating memory bank for {num_unique_grids} unique grids from {num_unique_puzzles} puzzles...")
        print(f"  (Each grid gets its own embedding in the memory bank)")
        print(f"  (ARCPuzzleDataset has {len(train_dataset)} puzzle examples)")
        memory_bank = MemoryBank(
            num_grids=num_unique_grids,
            embedding_dim=128,  # contrastive_embedding_dim
            momentum=args.memory_bank_momentum
        ).to(device)
        print(f"Contrastive loss weight: {args.contrastive_weight}")
        print(f"Temperature: {args.contrastive_temperature}")
        print(f"Num negatives: {args.contrastive_num_negatives}")

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Calculate total steps (including instance recognition batches if enabled)
    if args.use_contrastive and grid_train_loader is not None:
        steps_per_epoch = len(arc_train_loader) * (1 + args.instance_to_arc_ratio)
    else:
        steps_per_epoch = len(arc_train_loader)
    total_steps = steps_per_epoch * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, memory_bank)

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
        train_metrics = train_epoch(
            model, arc_train_loader, grid_train_loader, optimizer, scheduler, device, epoch, args, memory_bank
        )

        # Validate
        val_metrics = validate(model, val_loader, device, memory_bank)

        epoch_time = time.time() - epoch_start_time

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.num_epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        if args.use_contrastive:
            print(f"    ARC Loss: {train_metrics['arc_loss']:.4f}, Contrastive Loss: {train_metrics['contrastive_loss']:.4f}")
            print(f"    Contrastive Acc: {train_metrics['contrastive_accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

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
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
            }
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics,
                'args': vars(args),
            }
            if memory_bank is not None:
                checkpoint['memory_bank_state_dict'] = memory_bank.state_dict()

            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)

            # Also save as latest
            latest_path = checkpoint_dir / 'checkpoint_latest.pt'
            torch.save(checkpoint, latest_path)

            print(f'  Saved checkpoint to {checkpoint_path}')

        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': {
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                },
                'args': vars(args),
            }
            if memory_bank is not None:
                best_checkpoint['memory_bank_state_dict'] = memory_bank.state_dict()
            torch.save(best_checkpoint, checkpoint_dir / 'checkpoint_best.pt')
            print(f"  New best model! Val Acc: {val_metrics['accuracy']:.4f}")

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

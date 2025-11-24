"""
Training script for slot attention with instance recognition.
"""
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SlotInstanceModel
from memory_bank import MemoryBank
from loss import InfoNCELoss
from dataset.arc_dataset import ARCInstanceDataset, collate_fn_pad


def parse_args():
    parser = argparse.ArgumentParser(description='Train Slot Attention with Instance Recognition')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed dataset directory')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use (default: train)')
    parser.add_argument('--subset', type=str, default='all',
                        help='Dataset subset to use (default: all)')

    # Model architecture
    parser.add_argument('--num_slots', type=int, default=7,
                        help='Number of slots (default: 7)')
    parser.add_argument('--slot_dim', type=int, default=32,
                        help='Slot dimension (default: 32)')
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='Number of slot attention iterations (default: 5)')
    parser.add_argument('--encoder_feature_dim', type=int, default=64,
                        help='Encoder feature dimension (default: 64)')
    parser.add_argument('--encoder_hidden_dim', type=int, default=128,
                        help='Encoder hidden dimension (default: 128)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Final embedding dimension (default: 128)')

    # Instance recognition
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='Memory bank momentum (default: 0.99)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='InfoNCE temperature (default: 0.07)')
    parser.add_argument('--num_negatives', type=int, default=2048,
                        help='Number of negative samples (default: 2048)')

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
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available, else cpu)')

    return parser.parse_args()


def train_epoch(model, memory_bank, criterion, dataloader, optimizer, device, num_negatives):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        grid_ids = batch['grid_ids'].to(device)
        grids = batch['grids'].to(device)

        # Forward pass
        embeddings, slots = model(grids)

        # Get stored embeddings from memory bank
        stored_embeddings = memory_bank.get(grid_ids)

        # Sample negative embeddings
        negative_embeddings = memory_bank.sample_negatives(
            num_negatives=num_negatives,
            exclude_ids=grid_ids
        )

        # Compute loss
        loss, metrics = criterion(embeddings, stored_embeddings, negative_embeddings)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update memory bank (no gradients)
        memory_bank.update(grid_ids, embeddings.detach())

        # Track metrics
        total_loss += metrics['loss']
        total_accuracy += metrics['accuracy']
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'acc': f"{metrics['accuracy']:.4f}",
            'pos_sim': f"{metrics['avg_positive_sim']:.4f}",
            'neg_sim': f"{metrics['avg_negative_sim']:.4f}"
        })

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def main():
    args = parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.checkpoint_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("Training Slot Attention with Instance Recognition")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = ARCInstanceDataset(
        data_dir=args.data_dir,
        split=args.split,
        subset=args.subset,
        augment=True,
        max_grid_size=30
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

    # Get number of unique grids for memory bank
    num_unique_grids = len(dataset.puzzle_identifiers)
    print(f"Number of unique grids: {num_unique_grids}")
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
        max_grid_size=30
    ).to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create memory bank
    print("Creating memory bank...")
    memory_bank = MemoryBank(
        num_grids=num_unique_grids,
        embedding_dim=args.embedding_dim,
        momentum=args.momentum
    ).to(args.device)
    print()

    # Create loss function
    criterion = InfoNCELoss(temperature=args.temperature)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    print("Starting training...")
    print("=" * 80)
    print()

    best_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")

        # Train
        avg_loss, avg_accuracy = train_epoch(
            model=model,
            memory_bank=memory_bank,
            criterion=criterion,
            dataloader=dataloader,
            optimizer=optimizer,
            device=args.device,
            num_negatives=args.num_negatives
        )

        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print()

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'memory_bank_state_dict': memory_bank.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'memory_bank_state_dict': memory_bank.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_accuracy,
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

"""
Main application entry point for Slot Attention training with web dashboard.

Usage:
    python app.py

Opens a web dashboard at http://localhost:5004
Configure everything in the browser!
"""
import os
import json
import argparse
import torch
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from torch.utils.data import DataLoader

from model import SlotInstanceModel
from memory_bank import MemoryBank
from loss import InfoNCELoss
from dataset.arc_dataset import ARCInstanceDataset, collate_fn_pad

# ARC Solver imports
from models.arc_solver import ARCSlotSolver, ARCSlotSolverConfig
from dataset.arc_puzzle_dataset import ARCPuzzleDataset, collate_puzzle_batch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'slot-attention-training'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global training state
training_state = {
    'status': 'idle',  # idle, running, paused, stopped, completed
    'current_epoch': 0,
    'total_epochs': 0,
    'current_batch': 0,
    'total_batches': 0,
    'global_step': 0,
    'metrics': {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'positive_similarity': [],
        'negative_similarity': [],
    },
    'config': {},
    'start_time': None,
    # Use None so JSON serialization never chokes on inf/NaN
    'best_loss': None,
    'config_saved': False,  # Track if configuration has been saved
    'attention_viz_paused': False,  # Track if attention visualizations are paused
}

# Saved configuration (separate from training_state to persist across sessions)
saved_config = None

# Training components (initialized when user clicks start)
training_components = {
    'model': None,
    'memory_bank': None,
    'criterion': None,
    'optimizer': None,
    'dataloader': None,
    'device': None,
    'config': None,
}


def _sanitize_metric(value, default=0.0):
    """
    Ensure metrics we send over Socket.IO / JSON are finite numbers.

    Socket/JSON transports can choke on NaN/inf, which would make the
    frontend stop receiving updates after the first bad value.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not math.isfinite(v):
        return float(default)
    return v


def generate_attention_visualizations(grids, attn_weights, original_shapes, num_samples=3):
    """
    Generate attention mask visualizations for a batch of grids.

    Args:
        grids: [B, H, W] - Input grids (padded)
        attn_weights: [B, num_slots, H, W] - Attention masks (padded)
        original_shapes: List of (H, W) tuples for original (non-padded) sizes
        num_samples: Number of samples to visualize from batch

    Returns:
        List of base64-encoded PNG images
    """
    # ARC color palette (0-9)
    ARC_COLORS = [
        '#000000',  # 0: black
        '#0074D9',  # 1: blue
        '#FF4136',  # 2: red
        '#2ECC40',  # 3: green
        '#FFDC00',  # 4: yellow
        '#AAAAAA',  # 5: gray
        '#F012BE',  # 6: magenta
        '#FF851B',  # 7: orange
        '#7FDBFF',  # 8: sky blue
        '#870C25',  # 9: dark red
    ]

    visualizations = []
    B, num_slots, H, W = attn_weights.shape

    # # Debug: Check input types and values
    # print(f"\nVisualization Debug:")
    # print(f"  grids.shape={grids.shape}, grids.dtype={grids.dtype}")
    # print(f"  grids.min()={grids.min()}, grids.max()={grids.max()}")
    # print(f"  attn_weights.shape={attn_weights.shape}")
    # print(f"  First grid sample (before numpy conversion):")
    # print(f"    {grids[0, :5, :10]}")

    # Convert to numpy and move to CPU
    grids_np = grids.detach().cpu().numpy()
    attn_np = attn_weights.detach().cpu().numpy()

    # Take only num_samples from the batch
    num_samples = min(num_samples, B)

    for batch_idx in range(num_samples):
        grid = grids_np[batch_idx]  # [H_padded, W_padded]
        attn = attn_np[batch_idx]   # [num_slots, H_padded, W_padded]

        # Get original (non-padded) dimensions
        orig_H, orig_W = original_shapes[batch_idx]

        # Crop to original size (remove padding)
        grid = grid[:orig_H, :orig_W]
        attn = attn[:, :orig_H, :orig_W]

        H, W = orig_H, orig_W

        # Debug: Print grid statistics
        unique_colors = np.unique(grid)
        # print(f"  Sample {batch_idx}: shape={grid.shape}, "
        #       f"unique_colors={unique_colors}, num_colors={len(unique_colors)}")
        # print(f"    Grid values (first 5 rows): \n{grid[:min(5, H), :min(10, W)]}")
        # print(f"    BEFORE cropping - full padded grid shape: {grids_np[batch_idx].shape}")
        # print(f"    BEFORE cropping - unique colors: {np.unique(grids_np[batch_idx])}")
        # print(f"    Original shape to crop to: {original_shapes[batch_idx]}")

        # Create figure with grid and all slot attention masks
        fig, axes = plt.subplots(1, num_slots + 1, figsize=(3 * (num_slots + 1), 3))

        # Plot original grid
        grid_img = np.zeros((H, W, 3))
        for i in range(H):
            for j in range(W):
                color_idx = int(grid[i, j])
                color_hex = ARC_COLORS[color_idx]
                # Convert hex to RGB
                r = int(color_hex[1:3], 16) / 255.0
                g = int(color_hex[3:5], 16) / 255.0
                b = int(color_hex[5:7], 16) / 255.0
                grid_img[i, j] = [r, g, b]

        axes[0].imshow(grid_img, interpolation='nearest')
        axes[0].set_title('Input Grid', fontsize=10, color='white')
        axes[0].axis('off')

        # Plot each slot's attention mask
        for slot_idx in range(num_slots):
            mask = attn[slot_idx]  # [H, W]

            # Show pure attention heatmap (no grid overlay)
            im = axes[slot_idx + 1].imshow(mask, cmap='hot', interpolation='nearest')
            axes[slot_idx + 1].set_title(f'Slot {slot_idx}', fontsize=10, color='white')
            axes[slot_idx + 1].axis('off')

        # Set dark background
        fig.patch.set_facecolor('#1a1a2e')

        # Convert to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        visualizations.append(img_base64)

    return visualizations


def generate_reconstruction_visualizations(input_grids, pred_grids, target_grids, original_shapes, num_samples=2):
    """
    Generate reconstruction visualizations showing input, prediction, and target grids.

    Args:
        input_grids: [B, H, W] - Input test grids (padded)
        pred_grids: [B, H, W] - Predicted output grids (padded)
        target_grids: [B, H, W] - Ground truth output grids (padded)
        original_shapes: List of (H, W) tuples for original (non-padded) sizes
        num_samples: Number of samples to visualize from batch

    Returns:
        List of base64-encoded PNG images
    """
    # ARC color palette (0-9)
    ARC_COLORS = [
        '#000000',  # 0: black
        '#0074D9',  # 1: blue
        '#FF4136',  # 2: red
        '#2ECC40',  # 3: green
        '#FFDC00',  # 4: yellow
        '#AAAAAA',  # 5: gray
        '#F012BE',  # 6: magenta
        '#FF851B',  # 7: orange
        '#7FDBFF',  # 8: sky blue
        '#870C25',  # 9: dark red
    ]

    visualizations = []
    B = input_grids.shape[0]

    # Convert to numpy and move to CPU
    input_np = input_grids.detach().cpu().numpy()
    pred_np = pred_grids.detach().cpu().numpy()
    target_np = target_grids.detach().cpu().numpy()

    # Take only num_samples from the batch
    num_samples = min(num_samples, B)

    for batch_idx in range(num_samples):
        input_grid = input_np[batch_idx]
        pred_grid = pred_np[batch_idx]
        target_grid = target_np[batch_idx]

        # Get original (non-padded) dimensions
        orig_H, orig_W = original_shapes[batch_idx]

        # Crop to original size (remove padding)
        input_grid = input_grid[:orig_H, :orig_W]
        pred_grid = pred_grid[:orig_H, :orig_W]
        target_grid = target_grid[:orig_H, :orig_W]

        H, W = orig_H, orig_W

        # Create figure with 3 columns: input, prediction, target
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Helper function to render grid
        def render_grid(grid, ax, title):
            grid_img = np.zeros((H, W, 3))
            for i in range(H):
                for j in range(W):
                    color_idx = int(grid[i, j])
                    # Clamp to valid range [0, 9]
                    color_idx = max(0, min(9, color_idx))
                    color_hex = ARC_COLORS[color_idx]
                    # Convert hex to RGB
                    r = int(color_hex[1:3], 16) / 255.0
                    g = int(color_hex[3:5], 16) / 255.0
                    b = int(color_hex[5:7], 16) / 255.0
                    grid_img[i, j] = [r, g, b]

            ax.imshow(grid_img, interpolation='nearest')
            ax.set_title(title, fontsize=12, color='white', fontweight='bold')
            ax.axis('off')

        # Plot all three grids
        render_grid(input_grid, axes[0], 'Input')
        render_grid(pred_grid, axes[1], 'Prediction')
        render_grid(target_grid, axes[2], 'Target')

        # Calculate accuracy for this sample
        correct_pixels = np.sum(pred_grid == target_grid)
        total_pixels = H * W
        accuracy = correct_pixels / total_pixels * 100

        # Add accuracy as super title
        fig.suptitle(f'Sample {batch_idx + 1} - Accuracy: {accuracy:.1f}%',
                     fontsize=14, color='white', fontweight='bold')

        # Set dark background
        fig.patch.set_facecolor('#1a1a2e')

        # Convert to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        visualizations.append(img_base64)

    return visualizations


def initialize_training(config):
    """Initialize all training components from config."""
    print("Initializing training components...")
    print(f"Config: {json.dumps(config, indent=2)}")

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    print(f"Loading dataset from {config['data_dir']}...")
    dataset = ARCInstanceDataset(
        data_dir=config['data_dir'],
        split=config.get('split', 'train'),
        subset=config.get('subset', 'all'),
        augment=True,
        max_grid_size=30,
        max_puzzles=config.get('max_puzzles', None)
    )

    # Validate that we have enough puzzles for the number of negative samples
    num_unique_grids = len(dataset.puzzle_identifiers)
    num_negatives = config['num_negatives']
    batch_size = config['batch_size']

    # In the worst case, a batch could contain batch_size unique grids
    # After excluding them, we need at least num_negatives grids left
    min_required_grids = num_negatives + batch_size

    if num_unique_grids < min_required_grids:
        raise ValueError(
            f"Number of unique puzzles ({num_unique_grids}) must be at least "
            f"num_negatives + batch_size = {num_negatives} + {batch_size} = {min_required_grids}. "
            f"This ensures there are enough grids to sample negatives from after excluding the batch. "
            f"Please either:\n"
            f"  - Increase max_puzzles to at least {min_required_grids}\n"
            f"  - Decrease num_negatives (currently {num_negatives})\n"
            f"  - Decrease batch_size (currently {batch_size})"
        )

    print(f"✓ Validation passed: {num_unique_grids} puzzles >= {num_negatives} negatives + {batch_size} batch_size")

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
        collate_fn=collate_fn_pad,
        pin_memory=True if device == 'cuda' else False
    )

    # Create model
    print("Creating model...")
    model = SlotInstanceModel(
        num_colors=10,
        encoder_feature_dim=config['encoder_feature_dim'],
        encoder_hidden_dim=config['encoder_hidden_dim'],
        num_slots=config['num_slots'],
        slot_dim=config['slot_dim'],
        num_iterations=config['num_iterations'],
        embedding_dim=config['embedding_dim'],
        max_grid_size=30
    ).to(device)

    # Create memory bank
    print("Creating memory bank...")
    num_unique_grids = len(dataset.puzzle_identifiers)
    memory_bank = MemoryBank(
        num_grids=num_unique_grids,
        embedding_dim=config['embedding_dim'],
        momentum=config['momentum']
    ).to(device)

    # Loss and optimizer
    criterion = InfoNCELoss(temperature=config['temperature'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.0)
    )

    # Store components
    training_components.update({
        'model': model,
        'memory_bank': memory_bank,
        'criterion': criterion,
        'optimizer': optimizer,
        'dataloader': dataloader,
        'device': device,
        'config': config,
    })

    # Update state
    training_state['total_epochs'] = config['num_epochs']
    training_state['total_batches'] = len(dataloader)
    training_state['global_step'] = 0
    state_config = {
        'data_dir': config['data_dir'],
        'dataset_size': len(dataset),
        'unique_grids': num_unique_grids,
        'num_slots': config['num_slots'],
        'slot_dim': config['slot_dim'],
        'embedding_dim': config['embedding_dim'],
        'batch_size': config['batch_size'],
        'num_epochs': config['num_epochs'],
        'learning_rate': config['lr'],
        'momentum': config['momentum'],
        'temperature': config['temperature'],
        'num_negatives': config['num_negatives'],
        'batch_log_interval': config.get('batch_log_interval', 10),
        'device': device,
    }

    # Add max_puzzles if it was specified
    if 'max_puzzles' in config:
        state_config['max_puzzles'] = config['max_puzzles']

    training_state['config'] = state_config

    print(f"✓ Dataset loaded: {len(dataset)} examples")
    print(f"✓ Unique grids: {num_unique_grids}")
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    return True


def train_epoch(epoch):
    """Train for one epoch."""
    model = training_components['model']
    memory_bank = training_components['memory_bank']
    criterion = training_components['criterion']
    optimizer = training_components['optimizer']
    dataloader = training_components['dataloader']
    device = training_components['device']
    config = training_components['config']

    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    num_batches = 0

    # Calculate batch_log_interval once before the loop
    batch_log_interval = max(1, int(config.get('batch_log_interval', 10)))
    print(f"Epoch {epoch}: Will emit batch updates every {batch_log_interval} batches")
    emission_count = 0

    # Track metrics for averaging over the step interval
    step_loss = 0.0
    step_accuracy = 0.0
    step_pos_sim = 0.0
    step_neg_sim = 0.0
    step_batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        # Check status
        if training_state['status'] == 'stopped':
            return None

        while training_state['status'] == 'paused':
            time.sleep(0.1)

        training_state['current_batch'] = batch_idx + 1

        grid_ids = batch['grid_ids'].to(device)
        grids = batch['grids'].to(device)

        # Forward pass
        embeddings, slots = model(grids)
        stored_embeddings = memory_bank.get(grid_ids)
        negative_embeddings = memory_bank.sample_negatives(
            num_negatives=config['num_negatives'],
            exclude_ids=grid_ids
        )

        # Compute loss
        loss, metrics = criterion(embeddings, stored_embeddings, negative_embeddings)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update memory bank
        memory_bank.update(grid_ids, embeddings.detach())

        # Track metrics (keep raw values for epoch averages)
        total_loss += metrics['loss']
        total_accuracy += metrics['accuracy']
        total_pos_sim += metrics['avg_positive_sim']
        total_neg_sim += metrics['avg_negative_sim']
        num_batches += 1

        # Accumulate metrics for the current step
        step_loss += metrics['loss']
        step_accuracy += metrics['accuracy']
        step_pos_sim += metrics['avg_positive_sim']
        step_neg_sim += metrics['avg_negative_sim']
        step_batch_count += 1

        # Send batch update every N batches (configurable, 1-based)
        # Step number = how many complete intervals we've finished
        if batch_idx == 0 or (batch_idx + 1) % batch_log_interval == 0:
            # Calculate step number (1-indexed: step 1 = batches 1-10, step 2 = batches 11-20, etc.)
            current_step = ((batch_idx + 1) // batch_log_interval) + (epoch - 1) * ((len(dataloader) + batch_log_interval - 1) // batch_log_interval)

            # Average the metrics over the step interval
            avg_step_loss = step_loss / step_batch_count if step_batch_count > 0 else 0.0
            avg_step_accuracy = step_accuracy / step_batch_count if step_batch_count > 0 else 0.0
            avg_step_pos_sim = step_pos_sim / step_batch_count if step_batch_count > 0 else 0.0
            avg_step_neg_sim = step_neg_sim / step_batch_count if step_batch_count > 0 else 0.0

            # Prepare JSON-safe metrics
            safe_loss = _sanitize_metric(avg_step_loss)
            safe_acc = _sanitize_metric(avg_step_accuracy)
            safe_pos_sim = _sanitize_metric(avg_step_pos_sim)
            safe_neg_sim = _sanitize_metric(avg_step_neg_sim)

            emission_count += 1
            print(f"Emitting batch_update #{emission_count}: epoch={epoch}, batch={batch_idx + 1}/{len(dataloader)}, step={current_step}, loss={safe_loss:.4f}")
            try:
                socketio.emit('batch_update', {
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'step': current_step,
                    'total_batches': training_state['total_batches'],
                    'total_epochs': training_state['total_epochs'],
                    'loss': safe_loss,
                    'accuracy': safe_acc,
                    'avg_positive_sim': safe_pos_sim,
                    'avg_negative_sim': safe_neg_sim,
                })
                socketio.sleep(0.001)  # Small yield to allow the message to be sent
            except Exception as e:
                print(f"ERROR: Failed to emit batch_update: {e}")

            # Reset step metrics for next interval
            step_loss = 0.0
            step_accuracy = 0.0
            step_pos_sim = 0.0
            step_neg_sim = 0.0
            step_batch_count = 0

        # Generate and send attention visualizations every 20 batches
        if batch_idx == 0 or (batch_idx + 1) % 20 == 0:
            # Only generate and send if not paused
            if not training_state['attention_viz_paused']:
                try:
                    print(f"Generating attention visualizations at batch {batch_idx + 1}...")
                    # Forward pass with attention weights
                    with torch.no_grad():
                        _, _, attn_weights = model(grids, return_attn=True)

                    # Get original shapes from batch
                    original_shapes = batch.get('original_shapes', [(grids.shape[1], grids.shape[2])] * grids.shape[0])

                    # Generate visualizations (3 samples from batch)
                    vis_images = generate_attention_visualizations(grids, attn_weights, original_shapes, num_samples=3)

                    # Calculate current step for visualization
                    viz_step = ((batch_idx + 1) // batch_log_interval) + (epoch - 1) * ((len(dataloader) + batch_log_interval - 1) // batch_log_interval)

                    print(f"Emitting {len(vis_images)} attention visualizations...")
                    socketio.emit('attention_update', {
                        'epoch': epoch,
                        'batch': batch_idx + 1,
                        'step': viz_step,
                        'images': vis_images
                    })
                    socketio.sleep(0.001)
                    print(f"✓ Attention visualizations sent")
                except Exception as e:
                    print(f"ERROR: Failed to generate/emit attention visualizations: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Attention visualizations paused at batch {batch_idx + 1}")

    # Compute epoch metrics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_pos_sim = total_pos_sim / num_batches
    avg_neg_sim = total_neg_sim / num_batches

    print(f"Epoch {epoch} complete: Emitted {emission_count} batch updates out of {num_batches} total batches")

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'avg_positive_sim': avg_pos_sim,
        'avg_negative_sim': avg_neg_sim,
    }


def training_loop():
    """Main training loop (runs in background thread)."""
    config = training_components['config']

    print("\n" + "="*60)
    print("TRAINING LOOP STARTED")
    print("="*60)
    training_state['start_time'] = datetime.now().isoformat()

    for epoch in range(1, config['num_epochs'] + 1):
        if training_state['status'] == 'stopped':
            print("Training stopped by user")
            break

        training_state['current_epoch'] = epoch
        print(f"\n--- Starting Epoch {epoch}/{config['num_epochs']} ---")

        # Train epoch
        metrics = train_epoch(epoch)
        if metrics is None:
            print("Training stopped during epoch")
            break

        # Update metrics
        training_state['metrics']['epochs'].append(epoch)
        training_state['metrics']['train_loss'].append(metrics['loss'])
        training_state['metrics']['train_accuracy'].append(metrics['accuracy'])
        training_state['metrics']['positive_similarity'].append(metrics['avg_positive_sim'])
        training_state['metrics']['negative_similarity'].append(metrics['avg_negative_sim'])

        # Broadcast epoch complete
        print(f"Emitting 'epoch_complete' event for epoch {epoch}")
        socketio.emit('epoch_complete', {
            'epoch': epoch,
            'metrics': metrics,
            'state': get_training_state()
        })

        print(f"✓ Epoch {epoch}/{config['num_epochs']} - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")

        # Save checkpoint
        save_every = config.get('save_every', 10)
        if epoch % save_every == 0 or epoch == config['num_epochs']:
            save_checkpoint(epoch, metrics)

        # Save best model
        current_best = training_state.get('best_loss')
        if current_best is None or metrics['loss'] < current_best:
            training_state['best_loss'] = metrics['loss']
            save_checkpoint(epoch, metrics, is_best=True)

    if training_state['status'] == 'running':
        training_state['status'] = 'completed'
        print("\n" + "="*60)
        print("TRAINING COMPLETED - Emitting event")
        print("="*60)
        socketio.emit('training_completed', get_training_state())
        print("✓ Training completed event emitted")


def save_checkpoint(epoch, metrics, is_best=False):
    """Save model checkpoint."""
    config = training_components['config']
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
    path = os.path.join(checkpoint_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': training_components['model'].state_dict(),
        'memory_bank_state_dict': training_components['memory_bank'].state_dict(),
        'optimizer_state_dict': training_components['optimizer'].state_dict(),
        'metrics': metrics,
        'config': config,
    }, path)

    print(f"  Saved: {path}")

    # Clean up old checkpoints (keep only the N most recent, plus best_model.pt)
    if not is_best:
        keep_n_checkpoints = config.get('keep_n_checkpoints', 3)
        cleanup_old_checkpoints(checkpoint_dir, keep_n_checkpoints)


def cleanup_old_checkpoints(checkpoint_dir, keep_n=3):
    """Remove old checkpoint files, keeping only the N most recent."""
    import glob

    # Find all checkpoint files (excluding best_model.pt)
    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')
    checkpoints = glob.glob(pattern)

    if len(checkpoints) <= keep_n:
        return

    # Sort by modification time (oldest first)
    checkpoints.sort(key=os.path.getmtime)

    # Remove oldest checkpoints
    to_remove = checkpoints[:-keep_n]
    for ckpt_path in to_remove:
        try:
            os.remove(ckpt_path)
            print(f"  Removed old checkpoint: {os.path.basename(ckpt_path)}")
        except Exception as e:
            print(f"  Warning: Failed to remove {ckpt_path}: {e}")


def get_training_state():
    """Get current training state."""
    state = training_state.copy()
    if state['start_time']:
        elapsed = (datetime.now() - datetime.fromisoformat(state['start_time'])).total_seconds()
        state['elapsed_time'] = elapsed

    # Sanitize metrics for JSON serialization
    if 'metrics' in state:
        state['metrics'] = {
            'epochs': state['metrics']['epochs'][:],  # Copy the list
            'train_loss': [_sanitize_metric(v) for v in state['metrics']['train_loss']],
            'train_accuracy': [_sanitize_metric(v) for v in state['metrics']['train_accuracy']],
            'positive_similarity': [_sanitize_metric(v) for v in state['metrics']['positive_similarity']],
            'negative_similarity': [_sanitize_metric(v) for v in state['metrics']['negative_similarity']],
        }

    return state


# Flask routes
@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/state')
def api_state():
    return jsonify(get_training_state())


@app.route('/api/save_config', methods=['POST'])
def api_save_config():
    """Save configuration without starting training."""
    global saved_config

    config = request.json or {}
    print(f"\n{'='*60}")
    print(f"API /save_config called")
    print(f"Received config: {config}")
    print(f"{'='*60}")

    # Validate required fields
    if 'data_dir' not in config:
        print("ERROR: data_dir not in config")
        return jsonify({'error': 'data_dir is required'}), 400

    # Save configuration
    saved_config = config
    training_state['config_saved'] = True

    print(f"✓ Configuration saved successfully")
    return jsonify({
        'status': 'saved',
        'config': saved_config
    })


@app.route('/api/get_config', methods=['GET'])
def api_get_config():
    """Get the currently saved configuration."""
    global saved_config

    return jsonify({
        'config': saved_config,
        'config_saved': training_state['config_saved']
    })


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start training with saved configuration."""
    global saved_config

    print(f"\n{'='*60}")
    print(f"API /start called - Current status: {training_state['status']}")
    print(f"{'='*60}")

    if training_state['status'] in ['idle', 'stopped', 'completed']:
        # Check if configuration has been saved
        if not saved_config:
            print("ERROR: No configuration saved")
            return jsonify({'error': 'Please save configuration first'}), 400

        # Reset state for new training session
        if training_state['status'] in ['stopped', 'completed']:
            print("Resetting training state for new session...")
            training_state['status'] = 'idle'
            training_state['current_epoch'] = 0
            training_state['total_epochs'] = 0
            training_state['current_batch'] = 0
            training_state['total_batches'] = 0
            training_state['global_step'] = 0
            training_state['metrics'] = {
                'epochs': [],
                'train_loss': [],
                'train_accuracy': [],
                'positive_similarity': [],
                'negative_similarity': [],
            }
            training_state['start_time'] = None
            training_state['best_loss'] = None

        # Use saved configuration
        config = saved_config
        print(f"Using saved config: {config}")

        # Validate required fields
        if 'data_dir' not in config:
            print("ERROR: data_dir not in config")
            return jsonify({'error': 'data_dir is required in saved configuration'}), 400

        # Initialize training components
        try:
            print("Initializing training components...")
            initialize_training(config)
            print("✓ Training components initialized successfully")
        except Exception as e:
            print(f"ERROR during initialization: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Initialization failed: {str(e)}'}), 500

        training_state['status'] = 'running'
        print(f"Status changed to: {training_state['status']}")

        # Start training in background background task managed by SocketIO
        # This ensures emits work correctly with the chosen async mode.
        print("Starting training background task...")
        socketio.start_background_task(training_loop)
        print("✓ Training background task started")

        print("Emitting 'status_changed' event...")
        socketio.emit('status_changed', get_training_state())
        print("✓ Event emitted")

        return jsonify({'status': 'started'})

    elif training_state['status'] == 'paused':
        print("Resuming training...")
        training_state['status'] = 'running'
        socketio.emit('status_changed', get_training_state())
        return jsonify({'status': 'resumed'})
    else:
        print(f"ERROR: Cannot start, status is {training_state['status']}")
        return jsonify({'error': 'Training already running'}), 400


@app.route('/api/pause', methods=['POST'])
def api_pause():
    """Pause training."""
    if training_state['status'] == 'running':
        training_state['status'] = 'paused'
        socketio.emit('status_changed', get_training_state())
        return jsonify({'status': 'paused'})
    return jsonify({'error': 'Training not running'}), 400


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop training."""
    if training_state['status'] in ['running', 'paused']:
        training_state['status'] = 'stopped'
        socketio.emit('status_changed', get_training_state())
        return jsonify({'status': 'stopped'})
    return jsonify({'error': 'Training not active'}), 400


@app.route('/api/toggle_attention_viz', methods=['POST'])
def api_toggle_attention_viz():
    """Toggle attention visualization pause state."""
    training_state['attention_viz_paused'] = not training_state['attention_viz_paused']
    new_state = training_state['attention_viz_paused']
    print(f"Attention visualization {'paused' if new_state else 'resumed'}")
    socketio.emit('attention_viz_state_changed', {
        'paused': new_state
    })
    return jsonify({
        'paused': new_state,
        'message': 'Attention visualizations ' + ('paused' if new_state else 'resumed')
    })


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("\n" + "="*60)
    print("CLIENT CONNECTED")
    print("="*60)
    state = get_training_state()
    print(f"Sending initial state: status={state['status']}, epoch={state['current_epoch']}/{state['total_epochs']}")
    socketio.emit('initial_state', state)
    print("✓ Initial state sent")
    print("="*60 + "\n")


# ============================================================================
# ARC SOLVER TRAINING INTEGRATION
# ============================================================================

# Create separate namespace for ARC solver
arc_solver_namespace = '/arc_solver'

# ARC Solver training state
arc_solver_state = {
    'status': 'idle',
    'current_epoch': 0,
    'total_epochs': 0,
    'current_batch': 0,
    'total_batches': 0,
    'metrics': {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    },
    'config': {},
    'start_time': None,
    'best_val_accuracy': None,
    'config_saved': False,
}

arc_solver_saved_config = None

arc_solver_components = {
    'model': None,
    'optimizer': None,
    'scheduler': None,
    'train_loader': None,
    'val_loader': None,
    'device': None,
    'config': None,
}


def initialize_arc_solver_training(config):
    """Initialize ARC solver training components."""
    print("\n" + "="*70)
    print("Initializing ARC Solver Training")
    print("="*70)
    print(f"Config: {json.dumps(config, indent=2)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = ARCPuzzleDataset(
        data_dir=config['data_dir'],
        split='train',
        arc_version=config.get('arc_version', 'agi1'),
        max_train_examples=config['max_train_examples'],
        subset_size=config.get('subset_size', None),
        augment=config.get('augment', True),  # Enable augmentations by default
        augmentations_per_puzzle=config.get('augmentations_per_puzzle', 50),  # 50× augmentations
    )

    val_dataset = ARCPuzzleDataset(
        data_dir=config['data_dir'],
        split='eval',
        arc_version=config.get('arc_version', 'agi1'),
        max_train_examples=config['max_train_examples'],
        subset_size=config.get('subset_size', None) // 4 if config.get('subset_size') else None,
        augment=False,  # No augmentations for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_puzzle_batch,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_puzzle_batch,
        num_workers=0,
    )

    print(f"Train dataset: {len(train_dataset)} puzzles, {len(train_loader)} batches")
    print(f"Val dataset: {len(val_dataset)} puzzles, {len(val_loader)} batches")

    # Create model
    model_config = ARCSlotSolverConfig(
        grid_channels=1,
        cnn_hidden_dim=64,
        slot_dim=config['slot_dim'],
        num_slots_per_grid=config['num_slots'],
        slot_iterations=config['slot_iterations'],
        slot_mlp_hidden=128,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        expansion=4.0,
        max_train_examples=config['max_train_examples'],
        max_grid_size=30,
        decoder_hidden_dim=64,
        output_channels=10,
        forward_dtype="float32",
        use_rope=True,
    )

    model = ARCSlotSolver(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=0.01
    )

    total_steps = len(train_loader) * config['num_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Store components
    arc_solver_components.update({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'device': device,
        'config': config,
    })

    # Update state
    arc_solver_state['total_epochs'] = config['num_epochs']
    arc_solver_state['total_batches'] = len(train_loader)
    arc_solver_state['config'] = {
        'data_dir': config['data_dir'],
        'arc_version': config.get('arc_version', 'agi1'),
        'num_epochs': config['num_epochs'],
        'batch_size': config['batch_size'],
        'learning_rate': config['lr'],
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'num_slots': config['num_slots'],
        'device': str(device),
    }

    print("✓ ARC Solver training initialized")
    print("="*70 + "\n")
    return True


def train_arc_solver_epoch(epoch):
    """Train ARC solver for one epoch."""
    model = arc_solver_components['model']
    optimizer = arc_solver_components['optimizer']
    scheduler = arc_solver_components['scheduler']
    train_loader = arc_solver_components['train_loader']
    device = arc_solver_components['device']
    config = arc_solver_components['config']

    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    # Calculate batch_log_interval (default 10 batches per step)
    batch_log_interval = max(1, int(config.get('batch_log_interval', 10)))
    print(f"Epoch {epoch}: Will emit batch updates every {batch_log_interval} batches")
    emission_count = 0

    # Track metrics for averaging over the step interval
    step_loss = 0.0
    step_accuracy = 0.0
    step_batch_count = 0

    for batch_idx, batch in enumerate(train_loader):
        # Check status
        if arc_solver_state['status'] == 'stopped':
            return None

        while arc_solver_state['status'] == 'paused':
            time.sleep(0.1)

        arc_solver_state['current_batch'] = batch_idx + 1

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1

        # Accumulate metrics for the current step
        step_loss += loss.item()
        step_accuracy += accuracy.item()
        step_batch_count += 1

        # Send batch update every N batches
        if batch_idx == 0 or (batch_idx + 1) % batch_log_interval == 0:
            # Calculate step number
            current_step = ((batch_idx + 1) // batch_log_interval) + (epoch - 1) * ((len(train_loader) + batch_log_interval - 1) // batch_log_interval)

            # Average the metrics over the step interval
            avg_step_loss = step_loss / step_batch_count if step_batch_count > 0 else 0.0
            avg_step_accuracy = step_accuracy / step_batch_count if step_batch_count > 0 else 0.0

            emission_count += 1
            print(f"Emitting batch_update #{emission_count}: epoch={epoch}, batch={batch_idx + 1}/{len(train_loader)}, step={current_step}, loss={avg_step_loss:.4f}")

            try:
                socketio.emit('batch_update', {
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'step': current_step,
                    'total_batches': arc_solver_state['total_batches'],
                    'total_epochs': arc_solver_state['total_epochs'],
                    'train_loss': avg_step_loss,
                    'train_accuracy': avg_step_accuracy,
                }, namespace=arc_solver_namespace)
                socketio.sleep(0.001)
            except Exception as e:
                print(f"ERROR: Failed to emit batch_update: {e}")

            # Reset step metrics for next interval
            step_loss = 0.0
            step_accuracy = 0.0
            step_batch_count = 0

        # Generate attention visualizations every 10 batches
        if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
            try:
                with torch.no_grad():
                    # Get attention weights from slot encoder
                    # We'll visualize the test input grid
                    test_inputs = batch['test_inputs'][:, 0]  # [B, H, W]
                    B = test_inputs.shape[0]

                    # Generate visualizations for up to 2 samples
                    num_samples = min(2, B)
                    vis_images = []

                    for sample_idx in range(num_samples):
                        grid = test_inputs[sample_idx:sample_idx+1]  # [1, H, W]

                        # Get original grid shape
                        test_shape = batch['test_input_shapes'][sample_idx][0]  # (H, W)
                        orig_H, orig_W = test_shape

                        # Get features and attention from the model
                        features = model.cnn_encoder(grid)  # [B, spatial_dim, C] or [B, C, H', W']
                        slots, attn_weights = model.slot_encoder(features, return_attn=True)

                        # Handle different attention weight shapes
                        if attn_weights.ndim == 4:
                            # Already in [B, num_slots, H', W'] format
                            attn_reshaped = attn_weights
                            # If dimensions don't match original, interpolate
                            if attn_reshaped.shape[2:] != (orig_H, orig_W):
                                attn_reshaped = torch.nn.functional.interpolate(
                                    attn_reshaped,
                                    size=(orig_H, orig_W),
                                    mode='bilinear',
                                    align_corners=False
                                )
                        elif attn_weights.ndim == 3:
                            # Shape is [B, num_slots, spatial_dim]
                            B, num_slots, spatial_dim = attn_weights.shape

                            # Try to reshape based on original dimensions
                            if spatial_dim == orig_H * orig_W:
                                # Perfect! Attention maps directly to original grid
                                attn_reshaped = attn_weights.reshape(B, num_slots, orig_H, orig_W)
                            else:
                                # Find best square-ish dimensions that match spatial_dim
                                # Try aspect ratio from original grid first
                                aspect_ratio = orig_W / orig_H
                                feat_H = int((spatial_dim / aspect_ratio) ** 0.5)
                                feat_W = int(spatial_dim / feat_H)

                                # Verify the dimensions work
                                if feat_H * feat_W != spatial_dim:
                                    # Fall back to square assumption
                                    feat_size = int(spatial_dim ** 0.5)
                                    feat_H = feat_W = feat_size
                                    # If still doesn't match, find closest factors
                                    if feat_H * feat_W != spatial_dim:
                                        # Find factors of spatial_dim
                                        for h in range(int(spatial_dim ** 0.5), 0, -1):
                                            if spatial_dim % h == 0:
                                                feat_H = h
                                                feat_W = spatial_dim // h
                                                break

                                try:
                                    attn_reshaped = attn_weights.reshape(B, num_slots, feat_H, feat_W)
                                    # Upsample to match original grid size
                                    attn_reshaped = torch.nn.functional.interpolate(
                                        attn_reshaped,
                                        size=(orig_H, orig_W),
                                        mode='bilinear',
                                        align_corners=False
                                    )
                                except RuntimeError as e:
                                    print(f"  WARNING: Could not reshape attention {attn_weights.shape} to {feat_H}x{feat_W}: {e}")
                                    continue
                        else:
                            print(f"  ERROR: Unexpected attention dimensions: {attn_weights.ndim}")
                            continue

                        # Create visualization
                        vis = generate_attention_visualizations(
                            grid,
                            attn_reshaped,
                            [test_shape],
                            num_samples=1
                        )
                        vis_images.extend(vis)

                    # Calculate current step for visualization
                    viz_step = ((batch_idx + 1) // batch_log_interval) + (epoch - 1) * ((len(train_loader) + batch_log_interval - 1) // batch_log_interval)

                    # Emit visualization
                    socketio.emit('attention_update', {
                        'epoch': epoch,
                        'batch': batch_idx + 1,
                        'step': viz_step,
                        'images': vis_images
                    }, namespace=arc_solver_namespace)

            except Exception as e:
                print(f"Error generating attention viz: {e}")

        # Generate reconstruction visualizations every 30 batches (skip first batch)
        # Only run after training has started to avoid interference
        if batch_idx > 0 and (batch_idx + 1) % 30 == 0:
            try:
                # Save current training state
                was_training = model.training
                model.eval()

                with torch.no_grad():
                    # Get test inputs and targets (detach to avoid any gradient tracking)
                    test_inputs = batch['test_inputs'][:, 0].detach()  # [B, H, W] - first test example
                    test_outputs = batch['test_outputs'][:, 0].detach()  # [B, H, W] - corresponding output
                    test_shapes = [shape[0] for shape in batch['test_output_shapes']]  # List of (H, W)

                    # Create a clean batch copy for visualization (avoid modifying training batch)
                    viz_batch = {k: v.detach() if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}

                    # Generate predictions
                    output = model.forward(viz_batch)

                    # Handle dict output (model returns dict with 'predicted_grids' key)
                    if isinstance(output, dict):
                        # Try different possible keys
                        predictions = output.get('predicted_grids',
                                                output.get('predictions',
                                                          output.get('logits', None)))
                        if predictions is None:
                            print(f"  WARNING: Could not find predictions in output dict. Keys: {output.keys()}")
                            if was_training:
                                model.train()
                            continue
                    else:
                        predictions = output

                    # Detach predictions to avoid any gradient tracking
                    predictions = predictions.detach()

                    # Handle different output formats
                    if predictions.ndim == 4:
                        if predictions.shape[-1] == 10:
                            # [B, H, W, 10] format
                            pred_classes = torch.argmax(predictions, dim=-1)  # [B, H, W]
                        else:
                            # [B, 10, H, W] format
                            pred_classes = torch.argmax(predictions, dim=1)  # [B, H, W]
                    elif predictions.ndim == 3:
                        # [B, H, W] - already class predictions
                        pred_classes = predictions
                    else:
                        print(f"  WARNING: Unexpected prediction shape: {predictions.shape}")
                        if was_training:
                            model.train()
                        continue

                    # Generate visualizations (2 samples from batch)
                    vis_images = generate_reconstruction_visualizations(
                        test_inputs,
                        pred_classes,
                        test_outputs,
                        test_shapes,
                        num_samples=2
                    )

                # Restore training state
                if was_training:
                    model.train()

                # Calculate current step for visualization
                viz_step = ((batch_idx + 1) // batch_log_interval) + (epoch - 1) * ((len(train_loader) + batch_log_interval - 1) // batch_log_interval)

                print(f"Emitting {len(vis_images)} reconstruction visualizations...")
                socketio.emit('reconstruction_update', {
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'step': viz_step,
                    'images': vis_images
                }, namespace=arc_solver_namespace)
                socketio.sleep(0.001)
                print(f"✓ Reconstruction visualizations sent")

            except Exception as e:
                print(f"Error generating reconstruction viz: {e}")
                import traceback
                traceback.print_exc()
                # Make sure to restore training mode even on error
                if 'was_training' in locals() and was_training:
                    model.train()

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return {
        'train_loss': avg_loss,
        'train_accuracy': avg_accuracy,
    }


@torch.no_grad()
def validate_arc_solver():
    """Validate the ARC solver."""
    model = arc_solver_components['model']
    val_loader = arc_solver_components['val_loader']
    device = arc_solver_components['device']

    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        loss_dict = model.compute_loss(batch)
        total_loss += loss_dict['loss'].item()
        total_accuracy += loss_dict['pixel_accuracy'].item()
        num_batches += 1

    return {
        'val_loss': total_loss / num_batches,
        'val_accuracy': total_accuracy / num_batches,
    }


def arc_solver_training_loop():
    """Main training loop for ARC solver."""
    config = arc_solver_components['config']

    print("\n" + "="*70)
    print("ARC SOLVER TRAINING STARTED")
    print("="*70)
    arc_solver_state['start_time'] = datetime.now().isoformat()

    for epoch in range(config['num_epochs']):
        if arc_solver_state['status'] == 'stopped':
            break

        arc_solver_state['current_epoch'] = epoch + 1
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        # Train
        train_metrics = train_arc_solver_epoch(epoch + 1)
        if train_metrics is None:
            break

        # Validate
        val_metrics = validate_arc_solver()

        # Combine metrics
        metrics = {**train_metrics, **val_metrics}

        # Update state
        arc_solver_state['metrics']['epochs'].append(epoch + 1)
        arc_solver_state['metrics']['train_loss'].append(metrics['train_loss'])
        arc_solver_state['metrics']['train_accuracy'].append(metrics['train_accuracy'])
        arc_solver_state['metrics']['val_loss'].append(metrics['val_loss'])
        arc_solver_state['metrics']['val_accuracy'].append(metrics['val_accuracy'])

        # Emit epoch complete
        socketio.emit('epoch_complete', {
            'epoch': epoch + 1,
            'metrics': metrics,
            'state': get_arc_solver_state()
        }, namespace=arc_solver_namespace)

        print(f"✓ Epoch {epoch + 1} - Train Loss: {metrics['train_loss']:.4f}, "
              f"Train Acc: {metrics['train_accuracy']:.4f}, "
              f"Val Loss: {metrics['val_loss']:.4f}, "
              f"Val Acc: {metrics['val_accuracy']:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0:
            save_arc_solver_checkpoint(epoch + 1, metrics)

        # Track best model
        if (arc_solver_state['best_val_accuracy'] is None or
            metrics['val_accuracy'] > arc_solver_state['best_val_accuracy']):
            arc_solver_state['best_val_accuracy'] = metrics['val_accuracy']
            save_arc_solver_checkpoint(epoch + 1, metrics, is_best=True)

    if arc_solver_state['status'] == 'running':
        arc_solver_state['status'] = 'completed'
        socketio.emit('training_completed', get_arc_solver_state(), namespace=arc_solver_namespace)


def save_arc_solver_checkpoint(epoch, metrics, is_best=False):
    """Save ARC solver checkpoint."""
    config = arc_solver_components['config']
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints_arc_solver'))
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    filename = 'checkpoint_best.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
    path = checkpoint_dir / filename

    torch.save({
        'epoch': epoch,
        'model_state_dict': arc_solver_components['model'].state_dict(),
        'optimizer_state_dict': arc_solver_components['optimizer'].state_dict(),
        'scheduler_state_dict': arc_solver_components['scheduler'].state_dict(),
        'metrics': metrics,
        'config': config,
    }, path)

    print(f"  Saved: {path}")


def get_arc_solver_state():
    """Get current ARC solver training state."""
    state = arc_solver_state.copy()
    if state['start_time']:
        elapsed = (datetime.now() - datetime.fromisoformat(state['start_time'])).total_seconds()
        state['elapsed_time'] = elapsed
    return state


# ARC Solver Flask routes
@app.route('/arc_solver')
def arc_solver_dashboard():
    return render_template('arc_solver_dashboard.html')


@app.route('/arc_solver/api/state')
def arc_solver_api_state():
    return jsonify(get_arc_solver_state())


@app.route('/arc_solver/api/save_config', methods=['POST'])
def arc_solver_api_save_config():
    global arc_solver_saved_config

    config = request.json or {}
    print(f"\n{'='*60}")
    print(f"ARC Solver - Save config called")
    print(f"Config: {config}")
    print(f"{'='*60}")

    if 'data_dir' not in config:
        return jsonify({'error': 'data_dir is required'}), 400

    arc_solver_saved_config = config
    arc_solver_state['config_saved'] = True

    print(f"✓ ARC Solver config saved")
    return jsonify({'status': 'saved', 'config': arc_solver_saved_config})


@app.route('/arc_solver/api/start', methods=['POST'])
def arc_solver_api_start():
    global arc_solver_saved_config

    print(f"\n{'='*60}")
    print(f"ARC Solver - Start training")
    print(f"{'='*60}")

    if arc_solver_state['status'] in ['idle', 'stopped', 'completed']:
        if not arc_solver_saved_config:
            return jsonify({'error': 'Please save configuration first'}), 400

        # Reset state
        if arc_solver_state['status'] in ['stopped', 'completed']:
            arc_solver_state['status'] = 'idle'
            arc_solver_state['current_epoch'] = 0
            arc_solver_state['total_epochs'] = 0
            arc_solver_state['current_batch'] = 0
            arc_solver_state['total_batches'] = 0
            arc_solver_state['metrics'] = {
                'epochs': [],
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': [],
            }
            arc_solver_state['start_time'] = None
            arc_solver_state['best_val_accuracy'] = None

        # Initialize training
        try:
            initialize_arc_solver_training(arc_solver_saved_config)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Initialization failed: {str(e)}'}), 500

        arc_solver_state['status'] = 'running'
        socketio.start_background_task(arc_solver_training_loop)
        socketio.emit('status_changed', get_arc_solver_state(), namespace=arc_solver_namespace)

        return jsonify({'status': 'started'})

    elif arc_solver_state['status'] == 'paused':
        arc_solver_state['status'] = 'running'
        socketio.emit('status_changed', get_arc_solver_state(), namespace=arc_solver_namespace)
        return jsonify({'status': 'resumed'})
    else:
        return jsonify({'error': 'Training already running'}), 400


@app.route('/arc_solver/api/pause', methods=['POST'])
def arc_solver_api_pause():
    if arc_solver_state['status'] == 'running':
        arc_solver_state['status'] = 'paused'
        socketio.emit('status_changed', get_arc_solver_state(), namespace=arc_solver_namespace)
        return jsonify({'status': 'paused'})
    return jsonify({'error': 'Training not running'}), 400


@app.route('/arc_solver/api/stop', methods=['POST'])
def arc_solver_api_stop():
    if arc_solver_state['status'] in ['running', 'paused']:
        arc_solver_state['status'] = 'stopped'
        socketio.emit('status_changed', get_arc_solver_state(), namespace=arc_solver_namespace)
        return jsonify({'status': 'stopped'})
    return jsonify({'error': 'Training not active'}), 400


@socketio.on('connect', namespace=arc_solver_namespace)
def arc_solver_handle_connect():
    print("\n" + "="*60)
    print("ARC SOLVER CLIENT CONNECTED")
    print("="*60)
    state = get_arc_solver_state()
    socketio.emit('initial_state', state, namespace=arc_solver_namespace)
    print("✓ Initial state sent")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Slot Attention Training Dashboard')
    parser.add_argument('--port', type=int, default=5004,
                        help='Web server port (default: 5004)')
    args = parser.parse_args()

    print("=" * 80)
    print("🎯 Slot Attention Training Dashboard")
    print("=" * 80)
    print()
    print(f"🌐 Dashboard starting at http://localhost:{args.port}")
    print(f"   - Instance Recognition: http://localhost:{args.port}")
    print(f"   - ARC Puzzle Solver: http://localhost:{args.port}/arc_solver")
    print("=" * 80)
    print()
    print("📝 Configure training settings in your browser")
    print("🚀 Click 'Start Training' when ready!")
    print()
    print("Press Ctrl+C to exit")
    print()

    # Run server
    socketio.run(app, host='0.0.0.0', port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()

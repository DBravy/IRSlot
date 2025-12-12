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

from model import SlotInstanceModel, HierarchicalSlotModel, ColorAwareSpatialSlotModel
from memory_bank import MemoryBank
from loss import InfoNCELoss, ColorEntropyLoss
from dataset.arc_dataset import ARCInstanceDataset, collate_fn_pad

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

# Store batch-level history for chart data (allows late-joining clients to see full history)
batch_history = {
    'steps': [],
    'loss': [],
    'accuracy': [],
    'pos_sim': [],
    'neg_sim': [],
    'color_entropy': [],
}

# Saved configuration (separate from training_state to persist across sessions)
saved_config = None

# Training components (initialized when user clicks start)
training_components = {
    'model': None,
    'memory_bank': None,
    'criterion': None,
    'color_entropy_loss': None,
    'color_entropy_weight': 0.1,
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


# ARC color palette (0-9) - global for use by multiple visualization functions
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

ARC_COLOR_NAMES = [
    'Black', 'Blue', 'Red', 'Green', 'Yellow',
    'Gray', 'Magenta', 'Orange', 'Sky Blue', 'Dark Red'
]


def generate_attention_visualizations(grids, attn_weights, original_shapes, num_samples=3):
    """
    Generate attention mask visualizations for a batch of grids (standard model).

    Args:
        grids: [B, H, W] - Input grids (padded)
        attn_weights: [B, num_slots, H, W] - Attention masks (padded)
        original_shapes: List of (H, W) tuples for original (non-padded) sizes
        num_samples: Number of samples to visualize from batch

    Returns:
        List of base64-encoded PNG images
    """
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


def generate_hierarchical_visualizations(grids, attn_weights, original_shapes, num_samples=3):
    """
    Generate visualizations for HierarchicalSlotModel showing both layers:
    - Row 1: Original grid + spatial slot attention (projected from color attention)
    - Row 2: Slot-to-color attention heatmap

    Args:
        grids: [B, H, W] - Input grids (padded)
        attn_weights: [B, num_slots, num_colors] - Slot attention over colors (3D!)
        original_shapes: List of (H, W) tuples for original (non-padded) sizes
        num_samples: Number of samples to visualize from batch

    Returns:
        List of base64-encoded PNG images
    """
    visualizations = []
    B, num_slots, num_colors = attn_weights.shape

    # Convert to numpy and move to CPU
    grids_np = grids.detach().cpu().numpy()
    attn_np = attn_weights.detach().cpu().numpy()

    # Take only num_samples from the batch
    num_samples = min(num_samples, B)

    for batch_idx in range(num_samples):
        grid = grids_np[batch_idx]  # [H_padded, W_padded]
        slot_color_attn = attn_np[batch_idx]  # [num_slots, num_colors]

        # Get original (non-padded) dimensions
        orig_H, orig_W = original_shapes[batch_idx]

        # Crop to original size (remove padding)
        grid = grid[:orig_H, :orig_W]
        H, W = orig_H, orig_W

        # Find which colors are actually present in this grid
        present_colors = np.unique(grid).astype(int)

        # ============= PROJECT LAYER 2 ATTENTION BACK TO SPATIAL =============
        # For each slot, compute spatial attention by looking up color attention
        # spatial_attn[slot, i, j] = slot_color_attn[slot, grid[i,j]]
        spatial_attn = np.zeros((num_slots, H, W))
        for slot_idx in range(num_slots):
            for i in range(H):
                for j in range(W):
                    color = int(grid[i, j])
                    spatial_attn[slot_idx, i, j] = slot_color_attn[slot_idx, color]

        # Create figure with 2 rows:
        # Row 1: Original grid + spatial slot attention masks
        # Row 2: Slot-to-color attention heatmap
        fig = plt.figure(figsize=(3 * (num_slots + 1), 6))

        # Create grid spec for two rows
        gs = fig.add_gridspec(2, num_slots + 1, hspace=0.3)

        # ============= ROW 1: Spatial Slot Attention (Layer 2 projected) =============
        # Plot original grid
        ax_grid = fig.add_subplot(gs[0, 0])
        grid_img = np.zeros((H, W, 3))
        for i in range(H):
            for j in range(W):
                color_idx = int(grid[i, j])
                color_hex = ARC_COLORS[color_idx]
                r = int(color_hex[1:3], 16) / 255.0
                g = int(color_hex[3:5], 16) / 255.0
                b = int(color_hex[5:7], 16) / 255.0
                grid_img[i, j] = [r, g, b]

        ax_grid.imshow(grid_img, interpolation='nearest')
        ax_grid.set_title('Input Grid', fontsize=9, color='white')
        ax_grid.axis('off')

        # Plot spatial attention for each slot
        for slot_idx in range(num_slots):
            ax_slot = fig.add_subplot(gs[0, slot_idx + 1])
            mask = spatial_attn[slot_idx]  # [H, W]

            # Show as heatmap
            im = ax_slot.imshow(mask, cmap='hot', interpolation='nearest', vmin=0, vmax=spatial_attn.max())
            ax_slot.set_title(f'Slot {slot_idx}', fontsize=9, color='white')
            ax_slot.axis('off')

        # ============= ROW 2: Slot-to-Color Attention (Layer 2 raw) =============
        # Create a heatmap showing which colors each slot attends to
        ax_heatmap = fig.add_subplot(gs[1, :])

        # Only show attention for colors that are present in the grid
        attn_present = slot_color_attn[:, present_colors]  # [num_slots, num_present]

        # Create heatmap (use 'hot' colormap like spatial attention)
        im = ax_heatmap.imshow(attn_present, cmap='hot', aspect='auto', vmin=0, vmax=attn_present.max())

        # Labels
        ax_heatmap.set_yticks(range(num_slots))
        ax_heatmap.set_yticklabels([f'Slot {i}' for i in range(num_slots)], fontsize=8, color='white')
        ax_heatmap.set_xticks(range(len(present_colors)))
        ax_heatmap.set_xticklabels([ARC_COLOR_NAMES[c] for c in present_colors], fontsize=8, color='white', rotation=45, ha='right')
        ax_heatmap.set_title('Layer 2: Slot → Color Attention', fontsize=10, color='white')
        ax_heatmap.tick_params(colors='white')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.02, pad=0.02)
        cbar.ax.tick_params(colors='white', labelsize=7)

        # Add attention values as text annotations
        for i in range(num_slots):
            for j in range(len(present_colors)):
                val = attn_present[i, j]
                text_color = 'white' if val > attn_present.max() * 0.5 else 'black'
                ax_heatmap.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=7, color=text_color)

        # Set dark background
        fig.patch.set_facecolor('#1a1a2e')

        # Convert to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight', dpi=100)
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
    puzzle_filter = config.get('puzzle_filter', None)
    if puzzle_filter:
        print(f"Loading single puzzle '{puzzle_filter}' from raw JSON...")
    else:
        print(f"Loading dataset from {config['data_dir']}...")

    dataset = ARCInstanceDataset(
        data_dir=config['data_dir'],
        split=config.get('split', 'train'),
        subset=config.get('subset', 'all'),
        augment=True,
        max_grid_size=30,
        max_puzzles=config.get('max_puzzles', None),
        puzzle_filter=puzzle_filter,
        arc_version=config.get('arc_version', None),
        num_augmentations=config.get('num_augmentations', 200),
        raw_data_dir='kaggle/combined'
    )

    # Validate that we have enough puzzles for the number of negative samples
    num_unique_grids = len(dataset.puzzle_identifiers)
    num_negatives = config['num_negatives']
    batch_size = config['batch_size']

    # In the worst case, a batch could contain batch_size unique grids
    # After excluding them, we need at least num_negatives grids left
    min_required_grids = num_negatives + batch_size

    if num_unique_grids < min_required_grids:
        error_msg = (
            f"Number of unique puzzles ({num_unique_grids}) must be at least "
            f"num_negatives + batch_size = {num_negatives} + {batch_size} = {min_required_grids}. "
            f"This ensures there are enough grids to sample negatives from after excluding the batch. "
        )
        if config.get('puzzle_filter'):
            error_msg += (
                f"\n\nYou are filtering to puzzle '{config['puzzle_filter']}' which generated "
                f"{num_unique_grids} variants. Consider:\n"
                f"  - Increasing num_augmentations (currently {config.get('num_augmentations', 200)})\n"
                f"  - Reducing num_negatives to at most {num_unique_grids - batch_size} (currently {num_negatives})\n"
                f"  - Reducing batch_size (currently {batch_size})"
            )
        else:
            error_msg += (
                f"Please either:\n"
                f"  - Increase max_puzzles to at least {min_required_grids}\n"
                f"  - Decrease num_negatives (currently {num_negatives})\n"
                f"  - Decrease batch_size (currently {batch_size})"
            )
        raise ValueError(error_msg)

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
    model_type = config.get('model_type', 'standard')
    hard_attention = config.get('hard_attention', False)
    gumbel_temperature = config.get('gumbel_temperature', 1.0)

    if hard_attention:
        print(f"  Using HARD attention (Gumbel-Softmax) with temperature={gumbel_temperature}")
    else:
        print(f"  Using soft attention (standard)")

    if model_type == 'hierarchical':
        print(f"  Using HierarchicalSlotModel (hard-coded color segmentation)")
        model = HierarchicalSlotModel(
            num_colors=10,
            encoder_feature_dim=config['encoder_feature_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            num_slots=config['num_slots'],
            slot_dim=config['slot_dim'],
            num_iterations=config['num_iterations'],
            embedding_dim=config['embedding_dim'],
            max_grid_size=30,
            hard_attention=hard_attention,
            gumbel_temperature=gumbel_temperature
        ).to(device)
    elif model_type == 'color_aware':
        print(f"  Using ColorAwareSpatialSlotModel (spatial attention + learned color embeddings)")
        model = ColorAwareSpatialSlotModel(
            num_colors=10,
            encoder_feature_dim=config['encoder_feature_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            color_embed_dim=config.get('color_embed_dim', 32),
            num_slots=config['num_slots'],
            slot_dim=config['slot_dim'],
            num_iterations=config['num_iterations'],
            embedding_dim=config['embedding_dim'],
            max_grid_size=30,
            hard_attention=hard_attention,
            gumbel_temperature=gumbel_temperature
        ).to(device)
    else:
        print(f"  Using SlotInstanceModel (standard)")
        model = SlotInstanceModel(
            num_colors=10,
            encoder_feature_dim=config['encoder_feature_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            num_slots=config['num_slots'],
            slot_dim=config['slot_dim'],
            num_iterations=config['num_iterations'],
            embedding_dim=config['embedding_dim'],
            max_grid_size=30,
            hard_attention=hard_attention,
            gumbel_temperature=gumbel_temperature
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
    color_entropy_loss = ColorEntropyLoss(num_colors=10)
    color_entropy_weight = config.get('color_entropy_weight', 0.1)

    # Check if we need separate learning rate for color embeddings
    color_embed_lr = config.get('color_embed_lr', None)
    main_lr = config['lr']

    if model_type == 'color_aware' and color_embed_lr is not None:
        # Use parameter groups: separate LR for color embeddings
        color_embed_params = list(model.color_embedding.parameters())
        color_embed_param_ids = set(id(p) for p in color_embed_params)
        other_params = [p for p in model.parameters() if id(p) not in color_embed_param_ids]

        print(f"  Using separate LR for color embeddings: {color_embed_lr} (main LR: {main_lr})")
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': main_lr},
            {'params': color_embed_params, 'lr': color_embed_lr}
        ], weight_decay=config.get('weight_decay', 0.0))
    else:
        # Use single learning rate for all parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=main_lr,
            weight_decay=config.get('weight_decay', 0.0)
        )

    # Store components
    training_components.update({
        'model': model,
        'memory_bank': memory_bank,
        'criterion': criterion,
        'color_entropy_loss': color_entropy_loss,
        'color_entropy_weight': color_entropy_weight,
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
        'color_entropy_weight': color_entropy_weight,
        'device': device,
    }

    # Add max_puzzles if it was specified
    if 'max_puzzles' in config:
        state_config['max_puzzles'] = config['max_puzzles']

    # Add color_embed_lr if it was specified
    if color_embed_lr is not None:
        state_config['color_embed_lr'] = color_embed_lr

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
    color_entropy_criterion = training_components['color_entropy_loss']
    color_entropy_weight = training_components['color_entropy_weight']
    optimizer = training_components['optimizer']
    dataloader = training_components['dataloader']
    device = training_components['device']
    config = training_components['config']

    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_color_entropy = 0.0
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
    step_color_entropy = 0.0
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

        # Check model type for appropriate loss computation
        model_type = config.get('model_type', 'standard')

        # Forward pass
        if model_type == 'hierarchical':
            # Hierarchical model: color entropy loss not applicable (color segmentation is hard-coded)
            embeddings, slots = model(grids, return_attn=False)
            entropy_loss = torch.tensor(0.0, device=device)
            entropy_metrics = {'color_entropy': 0.0}
        else:
            # Standard model: use attention weights for color entropy
            embeddings, slots, attn_weights = model(grids, return_attn=True)
            entropy_loss, entropy_metrics = color_entropy_criterion(attn_weights, grids)

        stored_embeddings = memory_bank.get(grid_ids)
        negative_embeddings = memory_bank.sample_negatives(
            num_negatives=config['num_negatives'],
            exclude_ids=grid_ids
        )

        # Compute contrastive loss
        contrastive_loss, metrics = criterion(embeddings, stored_embeddings, negative_embeddings)

        # Combined loss (entropy loss is 0 for hierarchical model)
        loss = contrastive_loss + color_entropy_weight * entropy_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update memory bank
        memory_bank.update(grid_ids, embeddings.detach())

        # Calculate weighted entropy contribution to loss
        weighted_entropy_contribution = color_entropy_weight * entropy_metrics['color_entropy']

        # Track metrics (keep raw values for epoch averages)
        total_loss += loss.item()
        total_accuracy += metrics['accuracy']
        total_pos_sim += metrics['avg_positive_sim']
        total_neg_sim += metrics['avg_negative_sim']
        total_color_entropy += entropy_metrics['color_entropy']
        num_batches += 1

        # Accumulate metrics for the current step
        step_loss += loss.item()
        step_accuracy += metrics['accuracy']
        step_pos_sim += metrics['avg_positive_sim']
        step_neg_sim += metrics['avg_negative_sim']
        step_color_entropy += weighted_entropy_contribution  # Track weighted contribution
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
            avg_step_color_entropy = step_color_entropy / step_batch_count if step_batch_count > 0 else 0.0

            # Prepare JSON-safe metrics
            safe_loss = _sanitize_metric(avg_step_loss)
            safe_acc = _sanitize_metric(avg_step_accuracy)
            safe_pos_sim = _sanitize_metric(avg_step_pos_sim)
            safe_neg_sim = _sanitize_metric(avg_step_neg_sim)
            safe_color_entropy = _sanitize_metric(avg_step_color_entropy)

            emission_count += 1
            print(f"Emitting batch_update #{emission_count}: epoch={epoch}, batch={batch_idx + 1}/{len(dataloader)}, step={current_step}, loss={safe_loss:.4f}")

            # Store in batch history for late-joining clients
            batch_history['steps'].append(current_step)
            batch_history['loss'].append(safe_loss)
            batch_history['accuracy'].append(safe_acc)
            batch_history['pos_sim'].append(safe_pos_sim)
            batch_history['neg_sim'].append(safe_neg_sim)
            batch_history['color_entropy'].append(safe_color_entropy)

            # Apply mild downsampling to prevent unbounded memory growth
            # Keep last 2000 points at full resolution, downsample older data
            if len(batch_history['steps']) > 2000:
                # Downsample: keep every 5th point from the older data
                for key in batch_history.keys():
                    old_part = batch_history[key][:-2000]  # Everything except last 2000
                    recent_part = batch_history[key][-2000:]  # Last 2000 at full resolution

                    # Only keep every 5th point from old data (mild downsampling)
                    downsampled = old_part[::5]

                    # Reconstruct: downsampled old + full resolution recent
                    batch_history[key] = downsampled + recent_part

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
                    'color_entropy': safe_color_entropy,
                })
                socketio.sleep(0.001)  # Small yield to allow the message to be sent
            except Exception as e:
                print(f"ERROR: Failed to emit batch_update: {e}")

            # Reset step metrics for next interval
            step_loss = 0.0
            step_accuracy = 0.0
            step_pos_sim = 0.0
            step_neg_sim = 0.0
            step_color_entropy = 0.0
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

                    # Use appropriate visualization function based on model type
                    if model_type == 'hierarchical':
                        # Hierarchical model: attn_weights is [B, num_slots, num_colors] (3D)
                        vis_images = generate_hierarchical_visualizations(grids, attn_weights, original_shapes, num_samples=3)
                    else:
                        # Standard model: attn_weights is [B, num_slots, H, W] (4D)
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
    avg_color_entropy = total_color_entropy / num_batches

    print(f"Epoch {epoch} complete: Emitted {emission_count} batch updates out of {num_batches} total batches")
    print(f"  Color entropy: {avg_color_entropy:.4f}")

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'avg_positive_sim': avg_pos_sim,
        'avg_negative_sim': avg_neg_sim,
        'color_entropy': avg_color_entropy,
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
            # Clear batch history for new training
            batch_history['steps'] = []
            batch_history['loss'] = []
            batch_history['accuracy'] = []
            batch_history['pos_sim'] = []
            batch_history['neg_sim'] = []
            batch_history['color_entropy'] = []

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

    # Send batch history so late-joining clients can see the full chart
    if batch_history['steps']:
        print(f"Sending batch history: {len(batch_history['steps'])} data points")
        socketio.emit('batch_history', batch_history)
        print("✓ Batch history sent")

    print("="*60 + "\n")


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
    print(f"🌐 Dashboard: http://localhost:{args.port}")
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

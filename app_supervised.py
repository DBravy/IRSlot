"""
Supervised training app for Slot Attention with web dashboard.

Uses ground truth mask supervision (connectivity-based segmentation)
instead of contrastive learning. Designed for ARC-AGI-1 dataset.

Usage:
    python app_supervised.py

Opens a web dashboard at http://localhost:5005
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

from model import SlotInstanceModel, GNNSlotInstanceModel
from mask_supervision import MaskSupervisionLoss, extract_object_masks
from dataset.arc_dataset import ARCInstanceDataset, collate_fn_pad

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'slot-attention-supervised'
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
        'bg_loss': [],
        'fg_loss': [],
        'avg_matched': [],
    },
    'config': {},
    'start_time': None,
    'best_loss': None,
    'config_saved': False,
    'attention_viz_paused': False,
}

# Store batch-level history for chart data
batch_history = {
    'steps': [],
    'loss': [],
    'bg_loss': [],
    'fg_loss': [],
    'avg_matched': [],
}

# Saved configuration
saved_config = None

# Training components
training_components = {
    'model': None,
    'criterion': None,
    'optimizer': None,
    'dataloader': None,
    'device': None,
    'config': None,
}


def _sanitize_metric(value, default=0.0):
    """Ensure metrics are finite numbers for JSON serialization."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not math.isfinite(v):
        return float(default)
    return v


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

ARC_COLOR_NAMES = [
    'Black', 'Blue', 'Red', 'Green', 'Yellow',
    'Gray', 'Magenta', 'Orange', 'Sky Blue', 'Dark Red'
]


def generate_supervised_visualizations(grids, pred_attn, gt_masks, original_shapes, num_samples=3):
    """
    Generate visualizations comparing predicted attention to ground truth masks.

    Args:
        grids: [B, H, W] - Input grids (padded)
        pred_attn: [B, num_slots, H, W] - Predicted attention masks
        gt_masks: [B, num_slots, H, W] - Ground truth object masks
        original_shapes: List of (H, W) tuples for original sizes
        num_samples: Number of samples to visualize

    Returns:
        List of base64-encoded PNG images
    """
    visualizations = []
    B, num_slots, H, W = pred_attn.shape

    grids_np = grids.detach().cpu().numpy()
    pred_np = pred_attn.detach().cpu().numpy()
    gt_np = gt_masks.detach().cpu().numpy()

    num_samples = min(num_samples, B)

    for batch_idx in range(num_samples):
        grid = grids_np[batch_idx]
        pred = pred_np[batch_idx]
        gt = gt_np[batch_idx]

        orig_H, orig_W = original_shapes[batch_idx]

        # Crop to original size
        grid = grid[:orig_H, :orig_W]
        pred = pred[:, :orig_H, :orig_W]
        gt = gt[:, :orig_H, :orig_W]

        H, W = orig_H, orig_W

        # Create figure: Grid + Predicted slots + GT slots
        # Row 1: Input Grid + Predicted attention (num_slots)
        # Row 2: GT masks (num_slots)
        fig = plt.figure(figsize=(3 * (num_slots + 1), 6))
        gs = fig.add_gridspec(2, num_slots + 1, hspace=0.3)

        # === Row 1: Input Grid + Predicted Attention ===
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

        # Plot predicted attention
        for slot_idx in range(num_slots):
            ax = fig.add_subplot(gs[0, slot_idx + 1])
            mask = pred[slot_idx]
            ax.imshow(mask, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
            slot_name = 'BG' if slot_idx == 0 else f'Slot {slot_idx}'
            ax.set_title(f'Pred {slot_name}', fontsize=9, color='white')
            ax.axis('off')

        # === Row 2: Ground Truth Masks ===
        ax_label = fig.add_subplot(gs[1, 0])
        ax_label.text(0.5, 0.5, 'Ground\nTruth', fontsize=12, color='white',
                     ha='center', va='center', transform=ax_label.transAxes)
        ax_label.axis('off')

        for slot_idx in range(num_slots):
            ax = fig.add_subplot(gs[1, slot_idx + 1])
            mask = gt[slot_idx]
            ax.imshow(mask, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
            slot_name = 'BG' if slot_idx == 0 else f'Obj {slot_idx}'
            ax.set_title(f'GT {slot_name}', fontsize=9, color='white')
            ax.axis('off')

        fig.patch.set_facecolor('#1a1a2e')

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
    print("Initializing supervised training components...")
    print(f"Config: {json.dumps(config, indent=2)}")

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset with mask generation enabled
    print(f"Loading ARC-AGI-1 dataset from {config['data_dir']}...")

    dataset = ARCInstanceDataset(
        data_dir=config['data_dir'],
        split=config.get('split', 'train'),
        subset=config.get('subset', 'all'),
        augment=True,
        max_grid_size=30,
        max_puzzles=config.get('max_puzzles', None),
        num_slots=config['num_slots'],
        return_masks=True  # Enable ground truth mask generation
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_pad,
        pin_memory=True if device == 'cuda' else False
    )

    # Create model based on encoder type
    encoder_type = config.get('encoder_type', 'cnn')
    print(f"Creating model with {encoder_type.upper()} encoder...")

    if encoder_type == 'cnn':
        model = SlotInstanceModel(
            num_colors=10,
            encoder_feature_dim=config['encoder_feature_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            num_slots=config['num_slots'],
            slot_dim=config['slot_dim'],
            num_iterations=config['num_iterations'],
            embedding_dim=config['embedding_dim'],
            max_grid_size=30,
            hard_attention=config.get('hard_attention', False),
            gumbel_temperature=config.get('gumbel_temperature', 1.0)
        ).to(device)

    elif encoder_type == 'gnn':
        model = GNNSlotInstanceModel(
            num_colors=10,
            encoder_feature_dim=config['encoder_feature_dim'],
            encoder_hidden_dim=config['encoder_hidden_dim'],
            num_slots=config['num_slots'],
            slot_dim=config['slot_dim'],
            num_iterations=config['num_iterations'],
            embedding_dim=config['embedding_dim'],
            max_grid_size=30,
            hard_attention=config.get('hard_attention', False),
            gumbel_temperature=config.get('gumbel_temperature', 1.0),
            # GNN-specific parameters
            gnn_num_layers=config.get('gnn_num_layers', 4),
            gnn_edge_connectivity=config.get('gnn_edge_connectivity', 4),
            gnn_use_position=config.get('gnn_use_position', True),
            gnn_dropout=config.get('gnn_dropout', 0.0)
        ).to(device)

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'cnn' or 'gnn'.")

    # Create loss function
    criterion = MaskSupervisionLoss(
        bg_weight=config.get('bg_weight', 1.0),
        fg_weight=config.get('fg_weight', 1.0)
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.0)
    )

    # Store components
    training_components.update({
        'model': model,
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

    # Build config for display
    display_config = {
        'data_dir': config['data_dir'],
        'dataset_size': len(dataset),
        'unique_grids': len(dataset.puzzle_identifiers),
        'encoder_type': encoder_type,
        'num_slots': config['num_slots'],
        'slot_dim': config['slot_dim'],
        'embedding_dim': config['embedding_dim'],
        'batch_size': config['batch_size'],
        'num_epochs': config['num_epochs'],
        'learning_rate': config['lr'],
        'bg_weight': config.get('bg_weight', 1.0),
        'fg_weight': config.get('fg_weight', 1.0),
        'batch_log_interval': config.get('batch_log_interval', 10),
        'device': device,
    }

    # Add GNN-specific config if using GNN encoder
    if encoder_type == 'gnn':
        display_config.update({
            'gnn_num_layers': config.get('gnn_num_layers', 4),
            'gnn_edge_connectivity': config.get('gnn_edge_connectivity', 4),
            'gnn_use_position': config.get('gnn_use_position', True),
            'gnn_dropout': config.get('gnn_dropout', 0.0),
        })

    training_state['config'] = display_config

    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Unique grids: {len(dataset.puzzle_identifiers)}")
    print(f"Encoder: {encoder_type.upper()}")
    if encoder_type == 'gnn':
        print(f"  GNN layers: {config.get('gnn_num_layers', 4)}")
        print(f"  Edge connectivity: {config.get('gnn_edge_connectivity', 4)}-connected")
        print(f"  Position features: {config.get('gnn_use_position', True)}")
        print(f"  Dropout: {config.get('gnn_dropout', 0.0)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    return True


def train_epoch(epoch):
    """Train for one epoch with mask supervision."""
    model = training_components['model']
    criterion = training_components['criterion']
    optimizer = training_components['optimizer']
    dataloader = training_components['dataloader']
    device = training_components['device']
    config = training_components['config']

    model.train()
    total_loss = 0.0
    total_bg_loss = 0.0
    total_fg_loss = 0.0
    total_matched = 0
    num_batches = 0

    batch_log_interval = max(1, int(config.get('batch_log_interval', 10)))
    print(f"Epoch {epoch}: Will emit batch updates every {batch_log_interval} batches")
    emission_count = 0

    # Track metrics for averaging
    step_loss = 0.0
    step_bg_loss = 0.0
    step_fg_loss = 0.0
    step_matched = 0
    step_batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if training_state['status'] == 'stopped':
            return None

        while training_state['status'] == 'paused':
            time.sleep(0.1)

        training_state['current_batch'] = batch_idx + 1

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

        step_loss += metrics['loss']
        step_bg_loss += metrics['bg_loss']
        step_fg_loss += metrics['fg_loss']
        step_matched += metrics['num_matched']
        step_batch_count += 1

        # Send batch update
        if batch_idx == 0 or (batch_idx + 1) % batch_log_interval == 0:
            current_step = ((batch_idx + 1) // batch_log_interval) + (epoch - 1) * ((len(dataloader) + batch_log_interval - 1) // batch_log_interval)

            avg_step_loss = step_loss / step_batch_count if step_batch_count > 0 else 0.0
            avg_step_bg_loss = step_bg_loss / step_batch_count if step_batch_count > 0 else 0.0
            avg_step_fg_loss = step_fg_loss / step_batch_count if step_batch_count > 0 else 0.0
            avg_step_matched = step_matched / step_batch_count if step_batch_count > 0 else 0.0

            safe_loss = _sanitize_metric(avg_step_loss)
            safe_bg_loss = _sanitize_metric(avg_step_bg_loss)
            safe_fg_loss = _sanitize_metric(avg_step_fg_loss)
            safe_matched = _sanitize_metric(avg_step_matched)

            emission_count += 1
            print(f"Emitting batch_update #{emission_count}: epoch={epoch}, batch={batch_idx + 1}/{len(dataloader)}, loss={safe_loss:.4f}")

            batch_history['steps'].append(current_step)
            batch_history['loss'].append(safe_loss)
            batch_history['bg_loss'].append(safe_bg_loss)
            batch_history['fg_loss'].append(safe_fg_loss)
            batch_history['avg_matched'].append(safe_matched)

            # Mild downsampling for memory
            if len(batch_history['steps']) > 2000:
                for key in batch_history.keys():
                    old_part = batch_history[key][:-2000]
                    recent_part = batch_history[key][-2000:]
                    downsampled = old_part[::5]
                    batch_history[key] = downsampled + recent_part

            try:
                socketio.emit('batch_update', {
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'step': current_step,
                    'total_batches': training_state['total_batches'],
                    'total_epochs': training_state['total_epochs'],
                    'loss': safe_loss,
                    'bg_loss': safe_bg_loss,
                    'fg_loss': safe_fg_loss,
                    'avg_matched': safe_matched,
                })
                socketio.sleep(0.001)
            except Exception as e:
                print(f"ERROR: Failed to emit batch_update: {e}")

            step_loss = 0.0
            step_bg_loss = 0.0
            step_fg_loss = 0.0
            step_matched = 0
            step_batch_count = 0

        # Generate attention visualizations
        if (batch_idx == 0 or (batch_idx + 1) % 20 == 0):
            if not training_state['attention_viz_paused']:
                try:
                    print(f"Generating supervised visualizations at batch {batch_idx + 1}...")

                    original_shapes = batch.get('original_shapes', [(grids.shape[1], grids.shape[2])] * grids.shape[0])

                    vis_images = generate_supervised_visualizations(
                        grids, attn_weights, gt_masks, original_shapes, num_samples=3
                    )

                    viz_step = ((batch_idx + 1) // batch_log_interval) + (epoch - 1) * ((len(dataloader) + batch_log_interval - 1) // batch_log_interval)

                    print(f"Emitting {len(vis_images)} visualizations...")
                    socketio.emit('attention_update', {
                        'epoch': epoch,
                        'batch': batch_idx + 1,
                        'step': viz_step,
                        'images': vis_images
                    })
                    socketio.sleep(0.001)
                except Exception as e:
                    print(f"ERROR: Failed to generate visualizations: {e}")
                    import traceback
                    traceback.print_exc()

    avg_loss = total_loss / num_batches
    avg_bg_loss = total_bg_loss / num_batches
    avg_fg_loss = total_fg_loss / num_batches
    avg_matched = total_matched / num_batches

    print(f"Epoch {epoch} complete: {emission_count} batch updates")

    return {
        'loss': avg_loss,
        'bg_loss': avg_bg_loss,
        'fg_loss': avg_fg_loss,
        'avg_matched': avg_matched,
    }


def training_loop():
    """Main training loop."""
    config = training_components['config']

    print("\n" + "="*60)
    print("SUPERVISED TRAINING LOOP STARTED")
    print("="*60)
    training_state['start_time'] = datetime.now().isoformat()

    for epoch in range(1, config['num_epochs'] + 1):
        if training_state['status'] == 'stopped':
            print("Training stopped by user")
            break

        training_state['current_epoch'] = epoch
        print(f"\n--- Starting Epoch {epoch}/{config['num_epochs']} ---")

        metrics = train_epoch(epoch)
        if metrics is None:
            print("Training stopped during epoch")
            break

        # Update metrics
        training_state['metrics']['epochs'].append(epoch)
        training_state['metrics']['train_loss'].append(metrics['loss'])
        training_state['metrics']['bg_loss'].append(metrics['bg_loss'])
        training_state['metrics']['fg_loss'].append(metrics['fg_loss'])
        training_state['metrics']['avg_matched'].append(metrics['avg_matched'])

        print(f"Emitting 'epoch_complete' for epoch {epoch}")
        socketio.emit('epoch_complete', {
            'epoch': epoch,
            'metrics': metrics,
            'state': get_training_state()
        })

        print(f"Epoch {epoch}/{config['num_epochs']} - Loss: {metrics['loss']:.4f}, BG: {metrics['bg_loss']:.4f}, FG: {metrics['fg_loss']:.4f}")

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
        print("TRAINING COMPLETED")
        print("="*60)
        socketio.emit('training_completed', get_training_state())


def save_checkpoint(epoch, metrics, is_best=False):
    """Save model checkpoint."""
    config = training_components['config']
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints_supervised')
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
    path = os.path.join(checkpoint_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': training_components['model'].state_dict(),
        'optimizer_state_dict': training_components['optimizer'].state_dict(),
        'metrics': metrics,
        'config': config,
    }, path)

    print(f"  Saved: {path}")

    if not is_best:
        keep_n = config.get('keep_n_checkpoints', 3)
        cleanup_old_checkpoints(checkpoint_dir, keep_n)


def cleanup_old_checkpoints(checkpoint_dir, keep_n=3):
    """Remove old checkpoint files."""
    import glob

    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')
    checkpoints = glob.glob(pattern)

    if len(checkpoints) <= keep_n:
        return

    checkpoints.sort(key=os.path.getmtime)
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

    if 'metrics' in state:
        state['metrics'] = {
            'epochs': state['metrics']['epochs'][:],
            'train_loss': [_sanitize_metric(v) for v in state['metrics']['train_loss']],
            'bg_loss': [_sanitize_metric(v) for v in state['metrics']['bg_loss']],
            'fg_loss': [_sanitize_metric(v) for v in state['metrics']['fg_loss']],
            'avg_matched': [_sanitize_metric(v) for v in state['metrics']['avg_matched']],
        }

    return state


# Flask routes
@app.route('/')
def index():
    return render_template('dashboard_supervised.html')


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
    print(f"{'='*60}")

    if 'data_dir' not in config:
        return jsonify({'error': 'data_dir is required'}), 400

    saved_config = config
    training_state['config_saved'] = True

    print(f"Configuration saved successfully")
    return jsonify({
        'status': 'saved',
        'config': saved_config
    })


@app.route('/api/get_config', methods=['GET'])
def api_get_config():
    """Get the currently saved configuration."""
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
        if not saved_config:
            return jsonify({'error': 'Please save configuration first'}), 400

        # Reset state for new training
        if training_state['status'] in ['stopped', 'completed']:
            training_state['status'] = 'idle'
            training_state['current_epoch'] = 0
            training_state['total_epochs'] = 0
            training_state['current_batch'] = 0
            training_state['total_batches'] = 0
            training_state['global_step'] = 0
            training_state['metrics'] = {
                'epochs': [],
                'train_loss': [],
                'bg_loss': [],
                'fg_loss': [],
                'avg_matched': [],
            }
            training_state['start_time'] = None
            training_state['best_loss'] = None
            batch_history['steps'] = []
            batch_history['loss'] = []
            batch_history['bg_loss'] = []
            batch_history['fg_loss'] = []
            batch_history['avg_matched'] = []

        config = saved_config

        try:
            initialize_training(config)
        except Exception as e:
            print(f"ERROR during initialization: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Initialization failed: {str(e)}'}), 500

        training_state['status'] = 'running'

        socketio.start_background_task(training_loop)
        socketio.emit('status_changed', get_training_state())

        return jsonify({'status': 'started'})

    elif training_state['status'] == 'paused':
        training_state['status'] = 'running'
        socketio.emit('status_changed', get_training_state())
        return jsonify({'status': 'resumed'})
    else:
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
    socketio.emit('attention_viz_state_changed', {'paused': new_state})
    return jsonify({
        'paused': new_state,
        'message': 'Visualizations ' + ('paused' if new_state else 'resumed')
    })


@app.route('/api/save_checkpoint_now', methods=['POST'])
def api_save_checkpoint_now():
    """Save a checkpoint immediately."""
    if training_state['status'] not in ['running', 'paused']:
        return jsonify({'error': 'Training is not active'}), 400

    if training_components['model'] is None:
        return jsonify({'error': 'No model to save'}), 400

    try:
        epoch = training_state['current_epoch']
        config = training_components['config']
        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints_supervised')
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'manual_checkpoint_epoch_{epoch}_{timestamp}.pt'
        path = os.path.join(checkpoint_dir, filename)

        metrics = {}
        if training_state['metrics']['train_loss']:
            idx = len(training_state['metrics']['train_loss']) - 1
            metrics = {
                'loss': training_state['metrics']['train_loss'][idx],
                'bg_loss': training_state['metrics']['bg_loss'][idx],
                'fg_loss': training_state['metrics']['fg_loss'][idx],
                'avg_matched': training_state['metrics']['avg_matched'][idx],
            }

        torch.save({
            'epoch': epoch,
            'model_state_dict': training_components['model'].state_dict(),
            'optimizer_state_dict': training_components['optimizer'].state_dict(),
            'metrics': metrics,
            'config': config,
            'manual_save': True,
            'timestamp': timestamp,
        }, path)

        socketio.emit('checkpoint_saved', {
            'path': path,
            'epoch': epoch,
            'timestamp': timestamp
        })

        return jsonify({
            'status': 'saved',
            'path': path,
            'epoch': epoch,
            'timestamp': timestamp
        })

    except Exception as e:
        return jsonify({'error': f'Failed to save checkpoint: {str(e)}'}), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("\n" + "="*60)
    print("CLIENT CONNECTED")
    print("="*60)
    state = get_training_state()
    socketio.emit('initial_state', state)

    if batch_history['steps']:
        socketio.emit('batch_history', batch_history)


# Main entry point
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Supervised Slot Attention Training Dashboard')
    parser.add_argument('--port', type=int, default=5005,
                        help='Web server port (default: 5005)')
    args = parser.parse_args()

    print("=" * 80)
    print("Supervised Slot Attention Training Dashboard")
    print("=" * 80)
    print()
    print(f"Dashboard: http://localhost:{args.port}")
    print("=" * 80)
    print()
    print("Configure training settings in your browser")
    print("Uses mask supervision with ground truth from connectivity analysis")
    print()
    print("Press Ctrl+C to exit")
    print()

    socketio.run(app, host='0.0.0.0', port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()

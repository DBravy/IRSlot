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
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from torch.utils.data import DataLoader

from model import SlotInstanceModel
from memory_bank import MemoryBank
from loss import InfoNCELoss
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
        max_grid_size=30
    )

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
    training_state['config'] = {
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
    batch_log_interval = max(1, int(config.get('batch_log_interval', 1)))
    print(f"Epoch {epoch}: Will emit batch updates every {batch_log_interval} batches")
    emission_count = 0

    for batch_idx, batch in enumerate(dataloader):
        # Check status
        if training_state['status'] == 'stopped':
            return None

        while training_state['status'] == 'paused':
            time.sleep(0.1)

        training_state['current_batch'] = batch_idx + 1
        training_state['global_step'] += 1

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

        # Prepare JSON-safe metrics for live updates
        safe_loss = _sanitize_metric(metrics['loss'])
        safe_acc = _sanitize_metric(metrics['accuracy'])
        safe_pos_sim = _sanitize_metric(metrics['avg_positive_sim'])
        safe_neg_sim = _sanitize_metric(metrics['avg_negative_sim'])

        # Send batch update every N batches (configurable, 1-based)
        if batch_idx == 0 or (batch_idx + 1) % batch_log_interval == 0:
            emission_count += 1
            print(f"Emitting batch_update #{emission_count}: epoch={epoch}, batch={batch_idx + 1}/{len(dataloader)}, step={training_state['global_step']}, loss={safe_loss:.4f}")
            try:
                socketio.emit('batch_update', {
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'step': training_state['global_step'],
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

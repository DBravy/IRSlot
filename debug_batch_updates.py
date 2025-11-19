"""
Debug script to monitor batch update emissions and identify graphing issues.

This script will:
1. Check if batch updates are being emitted correctly
2. Monitor the emission frequency
3. Track the data being sent in batch_update events
"""
import os
import json
import time
from datetime import datetime

# Mock a simple training loop to check the emission logic
def test_emission_logic():
    """Test the batch emission logic with different configurations."""

    print("="*80)
    print("TESTING BATCH EMISSION LOGIC")
    print("="*80)
    print()

    test_cases = [
        {"batch_log_interval": 1, "total_batches": 10, "name": "Every batch"},
        {"batch_log_interval": 5, "total_batches": 20, "name": "Every 5 batches"},
        {"batch_log_interval": 10, "total_batches": 50, "name": "Every 10 batches"},
    ]

    for test in test_cases:
        batch_log_interval = test['batch_log_interval']
        total_batches = test['total_batches']

        print(f"\nTest: {test['name']} (interval={batch_log_interval}, batches={total_batches})")
        print("-" * 80)

        emissions = []
        for batch_idx in range(total_batches):
            # This is the same logic from app.py line 242
            if batch_idx == 0 or (batch_idx + 1) % batch_log_interval == 0:
                emissions.append(batch_idx + 1)  # batch number (1-based)

        print(f"Would emit at batch numbers: {emissions}")
        print(f"Total emissions: {len(emissions)}/{total_batches}")
        print()


def check_config_file():
    """Check if there's a saved config that might affect batch_log_interval."""

    print("="*80)
    print("CHECKING CONFIGURATION FILES")
    print("="*80)
    print()

    config_paths = [
        'checkpoints/args.json',
        'checkpoints_agi1/args.json',
        'checkpoints_agi2/args.json',
    ]

    for path in config_paths:
        if os.path.exists(path):
            print(f"\n✓ Found: {path}")
            with open(path, 'r') as f:
                config = json.load(f)

            # Check relevant settings
            relevant_keys = ['batch_log_interval', 'batch_size', 'num_epochs', 'data_dir']
            for key in relevant_keys:
                if key in config:
                    print(f"  {key}: {config[key]}")
        else:
            print(f"✗ Not found: {path}")
    print()


def analyze_frontend_logic():
    """Analyze the frontend batch_update handler logic."""

    print("="*80)
    print("ANALYZING FRONTEND LOGIC")
    print("="*80)
    print()

    print("Frontend batch_update handler (dashboard.html:754):")
    print("- Receives batch update events")
    print("- Should log: '⚙️ Batch X/Y (epoch Z) - Loss: ...'")
    print("- Adds data to trainingData arrays")
    print("- Calls updateCharts() to redraw")
    print()

    print("Potential issues:")
    print("1. Socket events not being received (check browser console)")
    print("2. Chart update function failing silently")
    print("3. Data not being appended correctly to trainingData")
    print("4. Plotly.react() not updating the chart")
    print()


def simulate_batch_updates():
    """Simulate what batch updates should look like."""

    print("="*80)
    print("SIMULATING BATCH UPDATES")
    print("="*80)
    print()

    # Simulate 3 batches with interval=1
    batch_log_interval = 10
    total_batches = 30

    print(f"Simulating epoch 1 with {total_batches} batches, interval={batch_log_interval}")
    print("-" * 80)

    for batch_idx in range(total_batches):
        if batch_idx == 0 or (batch_idx + 1) % batch_log_interval == 0:
            batch_num = batch_idx + 1
            global_step = batch_idx + 1

            update_data = {
                'epoch': 1,
                'batch': batch_num,
                'step': global_step,
                'total_batches': total_batches,
                'total_epochs': 100,
                'loss': 0.5 - (batch_idx * 0.01),  # Simulated decreasing loss
                'accuracy': 0.5 + (batch_idx * 0.01),  # Simulated increasing accuracy
                'avg_positive_sim': 0.7,
                'avg_negative_sim': 0.3,
            }

            print(f"  Batch {batch_num}/{total_batches}: step={global_step}, loss={update_data['loss']:.4f}")

    print()
    print(f"Expected {len([i for i in range(total_batches) if i == 0 or (i + 1) % batch_log_interval == 0])} emissions")
    print()


def check_socketio_mode():
    """Check the SocketIO configuration."""

    print("="*80)
    print("CHECKING SOCKETIO CONFIGURATION")
    print("="*80)
    print()

    print("From app.py:")
    print("- Line 29: socketio = SocketIO(app, cors_allowed_origins='*')")
    print("- Line 498: socketio.run(app, ..., allow_unsafe_werkzeug=True)")
    print()
    print("Recommendations:")
    print("1. Check browser console for 'Socket connected!' message")
    print("2. Check for any CORS errors")
    print("3. Verify socketio.emit() calls are executing")
    print("4. Add debug prints before socketio.emit() calls")
    print()


def main():
    """Run all diagnostic tests."""

    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "BATCH UPDATE DEBUGGING TOOL" + " "*31 + "║")
    print("╚" + "="*78 + "╝")
    print()

    test_emission_logic()
    check_config_file()
    simulate_batch_updates()
    analyze_frontend_logic()
    check_socketio_mode()

    print("="*80)
    print("DEBUGGING RECOMMENDATIONS")
    print("="*80)
    print()
    print("1. Check browser console for:")
    print("   - '✓ Socket connected!' message")
    print("   - '⚙️ Batch X/Y (epoch Z)...' logs")
    print("   - Any JavaScript errors")
    print()
    print("2. Add debug print in app.py before socketio.emit('batch_update', ...):")
    print("   print(f'DEBUG: Emitting batch_update for batch {batch_idx + 1}')")
    print()
    print("3. Check if training loop is actually running multiple batches:")
    print("   - Look for 'Starting Epoch X/Y' messages")
    print("   - Monitor total_batches in training_state")
    print()
    print("4. Common issues:")
    print("   - Frontend not receiving socket events (connection issue)")
    print("   - Training stopping after 1 batch (data/model error)")
    print("   - Chart not updating due to JavaScript error")
    print("   - socketio.emit() not being called (logic error)")
    print()
    print("="*80)
    print()


if __name__ == '__main__':
    main()

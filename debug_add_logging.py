"""
Script to add debug logging to app.py to track batch update emissions.

This will help identify:
1. How many batches are being processed
2. Which batch numbers trigger emissions
3. Whether socketio.emit() is being called
4. Whether the emission logic is working correctly
"""

import sys

def add_debug_logging():
    """Read app.py and suggest debug additions."""

    print("="*80)
    print("DEBUG LOGGING ADDITIONS FOR app.py")
    print("="*80)
    print()

    print("Add these debug print statements to app.py:")
    print()

    print("1. At line 240 (before batch_log_interval calculation):")
    print("-" * 80)
    print("""
        # Debug: Track batch processing
        print(f"DEBUG: Processing batch {batch_idx + 1}/{len(dataloader)}, global_step={training_state['global_step']}")
""")
    print()

    print("2. At line 243 (inside the if statement, before socketio.emit):")
    print("-" * 80)
    print("""
            # Debug: Track emissions
            print(f"DEBUG: Emitting batch_update for batch {batch_idx + 1}, batch_log_interval={batch_log_interval}")
""")
    print()

    print("3. At line 256 (after socketio.emit):")
    print("-" * 80)
    print("""
            print(f"DEBUG: Emission complete for batch {batch_idx + 1}")
""")
    print()

    print("="*80)
    print()

    print("Alternatively, run this to create a patched version:")
    print()
    print("  python debug_add_logging.py patch")
    print()


def create_patched_version():
    """Create a patched version of app.py with debug logging."""

    print("Creating patched version of app.py...")

    with open('app.py', 'r') as f:
        lines = f.readlines()

    patched_lines = []
    for i, line in enumerate(lines):
        patched_lines.append(line)

        # Add debug after line with "for batch_idx, batch in enumerate(dataloader):"
        if 'for batch_idx, batch in enumerate(dataloader):' in line:
            indent = '    ' * 2  # Two levels of indentation
            patched_lines.append(f'{indent}# DEBUG: Track batch processing\n')
            patched_lines.append(f'{indent}print(f"DEBUG: Processing batch {{batch_idx + 1}}/{{len(dataloader)}}, step={{training_state[\'global_step\'] + 1}}")\n')

        # Add debug before socketio.emit('batch_update'
        if "socketio.emit('batch_update'" in line:
            indent = '    ' * 3  # Three levels of indentation
            patched_lines.insert(-1, f'{indent}# DEBUG: About to emit batch_update\n')
            patched_lines.insert(-1, f'{indent}print(f"DEBUG: Emitting batch_update for batch {{batch_idx + 1}}, interval={{batch_log_interval}}")\n')

    # Write patched version
    with open('app_debug.py', 'w') as f:
        f.writelines(patched_lines)

    print("✓ Created app_debug.py with debug logging")
    print()
    print("To use:")
    print("  1. Backup your current app.py: cp app.py app.py.backup")
    print("  2. Use the debug version: python app_debug.py")
    print("  3. Watch the console output while training")
    print()


def check_common_issues():
    """Check for common issues that could cause this problem."""

    print("="*80)
    print("COMMON ISSUES CHECKLIST")
    print("="*80)
    print()

    print("✓ Check #1: Is batch_log_interval being set correctly?")
    print("  - Default in app.py:241 is 1")
    print("  - Default in dashboard.html:335 is 10")
    print("  - Value is from: config.get('batch_log_interval', 1)")
    print()

    print("✓ Check #2: Is the training loop processing multiple batches?")
    print("  - Check console for batch processing messages")
    print("  - Verify total_batches > 1 in training_state")
    print()

    print("✓ Check #3: Are socket events being received in browser?")
    print("  - Open browser DevTools Console (F12)")
    print("  - Look for: '✓ Socket connected!'")
    print("  - Look for: '⚙️ Batch X/Y (epoch Z)...'")
    print()

    print("✓ Check #4: Is socketio.sleep(0) causing issues?")
    print("  - Line 255 has: socketio.sleep(0)")
    print("  - This should yield to allow emission")
    print("  - Try removing it or increasing to socketio.sleep(0.01)")
    print()

    print("✓ Check #5: Chart update frequency")
    print("  - Frontend calls updateCharts() on every batch_update")
    print("  - Check if Plotly.react() is failing")
    print("  - Open browser console to see errors")
    print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'patch':
        create_patched_version()
    else:
        add_debug_logging()
        check_common_issues()

# Web Dashboard for Training Monitoring

A beautiful, real-time web dashboard for **monitoring and controlling** your Slot Attention training!

## Features

⚙️ **Zero-Config Startup**
- Just run `python app.py` - no command-line arguments needed!
- Configure everything through the web interface
- Sensible defaults with full customization

✨ **Real-time Graphs**
- Loss and Accuracy curves updated live
- Positive vs Negative similarity tracking
- Interactive Plotly charts with zoom/pan

📊 **Live Metrics**
- Current epoch progress with visual bar
- Latest loss, accuracy, and similarity scores
- Training configuration display

🎮 **Training Controls**
- **Start/Stop/Pause** buttons right in the dashboard!
- Training waits for you to click "Start" - full control!
- Safe stop with automatic checkpoint saving

🎨 **Modern UI**
- Dark theme optimized for long viewing sessions
- Color-coded metrics (green for good, red for concerning)
- Responsive design works on any screen size

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard

**One simple command:**

```bash
python app.py
```

That's it! The dashboard starts at **http://localhost:5000**

Optional: Use `--port` to change the port:
```bash
python app.py --port 8080
```

### 3. Configure and Control Training from Browser

1. Open browser to `http://localhost:5000`
2. **Fill in the configuration form** with your training settings:
   - Data directory (e.g., `data/processed_agi1`)
   - Model parameters (slots, dimensions, iterations)
   - Training hyperparameters (epochs, batch size, learning rate)
   - Loss parameters (temperature, momentum, negatives)
3. Click **"▶ Start Training"** when ready!
4. Use **"⏸ Pause"** to temporarily pause
5. Use **"⏹ Stop"** to end early (checkpoint auto-saved)

## Command Line Options

The web dashboard only needs one optional argument:

```bash
python app.py [--port PORT]
```

- `--port`: Web server port (default: 5000)

**All training configuration is done through the web interface!** No need for command-line arguments.

## Dashboard Overview

### Header Section
- **Training Status**: Shows whether training is Running, Paused, or Stopped
- **Current Progress**: Displays current epoch and progress bar

### Metrics Cards

1. **Training Progress**
   - Current epoch / Total epochs
   - Visual progress bar

2. **Current Metrics**
   - Loss (lower is better)
   - Accuracy (higher is better, shown as %)
   - Positive Similarity (should be high)
   - Negative Similarity (should be lower than positive)

### Live Graphs

1. **Loss & Accuracy Chart**
   - Red line: Training loss
   - Green line: Training accuracy
   - Dual Y-axes for different scales

2. **Similarity Scores Chart**
   - Green line: Positive similarities (same grid, different augmentations)
   - Purple line: Negative similarities (different grids)
   - Positive should be consistently higher than negative

### Configuration Panel

**Before Training (Status: Idle)**
- Interactive form to configure all training parameters
- Includes sensible defaults for quick starts
- Configurable parameters:
  - **Data**: Directory path (required)
  - **Model**: Number of slots, slot dimension, embedding dimension
  - **Encoder**: Feature dimension, hidden dimension, iterations
  - **Training**: Epochs, batch size, learning rate
  - **Loss**: Temperature, momentum, number of negatives
  - **Checkpointing**: Directory and save frequency

**During/After Training**
- Shows all configured hyperparameters:
  - Data directory and dataset statistics
  - Model architecture (slots, dimensions)
  - Learning rate and batch size
  - Temperature, momentum, etc.

## What to Look For

### Good Training Signs ✅
- **Loss decreasing** steadily over time
- **Accuracy increasing** toward 100%
- **Positive similarity > Negative similarity** with a clear gap
- Smooth curves without sudden jumps

### Warning Signs ⚠️
- Loss increasing or oscillating wildly
- Accuracy stuck at low values
- Positive and negative similarities converging
- NaN or infinite values

### Typical Training Behavior

**Epochs 1-20:**
- Loss drops rapidly
- Accuracy climbs quickly
- Similarities separate

**Epochs 20-50:**
- Loss continues decreasing but slower
- Accuracy reaches 80-95%
- Stable similarity gap

**Epochs 50-100:**
- Fine-tuning phase
- Metrics plateau
- Small incremental improvements

## Advanced Features

### Real-time Updates
The dashboard uses WebSockets (Socket.IO) for instant updates with no page refresh needed. Metrics appear as soon as each epoch completes!

### Interactive Charts
- **Zoom**: Click and drag on charts
- **Pan**: Double-click to reset view
- **Hover**: See exact values at any point
- **Legend**: Click to show/hide lines

### Multiple Viewers
You can open the dashboard in multiple browser windows/tabs simultaneously. All viewers will see the same real-time updates!

### Mobile Friendly
The dashboard is responsive and works on tablets and phones, so you can monitor training on the go!

## Troubleshooting

### Dashboard won't load

**Problem**: `Connection refused` or can't access `localhost:5000`

**Solutions**:
1. Make sure training script is running
2. Check if port 5000 is available: `lsof -i :5000`
3. Try a different port: `./train_with_web.sh agi1 8080`
4. Check firewall settings

### No graphs showing

**Problem**: Dashboard loads but shows no data

**Solutions**:
1. Wait for first epoch to complete
2. Check browser console for errors (F12)
3. Refresh the page
4. Clear browser cache

### Graphs not updating

**Problem**: Dashboard shows initial data but doesn't update

**Solutions**:
1. Check if training is actually running (look at terminal)
2. Refresh the page
3. Check WebSocket connection in browser console

### Port already in use

**Problem**: `Address already in use` error

**Solutions**:
```bash
# Find what's using the port
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or use a different port
./train_with_web.sh agi1 8080
```

## Architecture

```
train_web.py (Main training script)
    ↓
web_app.py (Flask server + SocketIO)
    ↓
templates/dashboard.html (Frontend)
    ↓
Browser (You!)
```

### How It Works

1. **Training Loop**: Runs in main thread
2. **Web Server**: Runs in background thread
3. **Metrics Update**: After each epoch, training emits metrics
4. **WebSocket Broadcast**: Server pushes to all connected clients
5. **Chart Update**: Browser receives data and redraws charts

## Tips

- **Keep dashboard open** during long training runs to spot issues early
- **Monitor similarity gap** - it's often more informative than raw accuracy
- **Use progress bar** to estimate time remaining
- **Screenshot interesting patterns** in the graphs for analysis
- **Check metrics regularly** but don't over-optimize for single epochs

## Access from Other Devices

To access the dashboard from other devices on your network:

1. Find your local IP:
   ```bash
   # On Mac/Linux
   ifconfig | grep "inet "

   # On Windows
   ipconfig
   ```

2. Launch with host binding:
   ```bash
   python train_web.py --web_port 5000 --data_dir data/processed_agi1 ...
   ```

3. Access from other device:
   ```
   http://YOUR_LOCAL_IP:5000
   ```

## Example Screenshots

The dashboard features:
- **Purple gradient header** with status indicator
- **Dark theme** cards with metrics
- **Smooth animations** on updates
- **Color-coded values** (green=good, red=bad)
- **Interactive Plotly graphs** with professional styling

Enjoy monitoring your training! 🚀

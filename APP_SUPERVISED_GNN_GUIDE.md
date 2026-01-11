# Using GNN Encoder in app_supervised.py

## Overview

The supervised training web app now supports both CNN and GNN encoders. You can easily switch between them in the browser interface.

## Quick Start

1. **Launch the web app:**
   ```bash
   python app_supervised.py
   ```

2. **Open the dashboard:**
   - Navigate to `http://localhost:5005` in your browser

3. **Configure GNN encoder:**
   - Scroll to the **"Encoder Selection"** section
   - Select **"GNN (Graph Neural Network)"** from the encoder type dropdown
   - GNN-specific parameters will appear below

4. **Configure GNN parameters:**
   - **GNN Layers**: Number of graph convolution layers (default: 4)
   - **Edge Connectivity**: 4-connected or 8-connected (default: 4)
   - **Position Features**: Include spatial position features (default: enabled)
   - **GNN Dropout**: Dropout rate (default: 0.0)

5. **Save configuration:**
   - Click **"Save Configuration"** button

6. **Start training:**
   - Click **"Start Training"** button
   - Monitor real-time metrics and visualizations

## Web Interface Features

### Encoder Selection Section

The new encoder selection section includes:

- **Encoder Type Dropdown**: Choose between CNN and GNN
- **GNN Parameters Panel**: Appears automatically when GNN is selected
  - Highlighted with a dark background for visibility
  - All GNN-specific settings in one place

### Configuration Display

When training is active, you'll see:
- **Encoder Type**: CNN or GNN
- **GNN-specific config** (if using GNN):
  - Number of GNN layers
  - Edge connectivity (4 or 8)
  - Position features enabled/disabled
  - Dropout rate

## Recommended Settings

### Starting Configuration (GNN)
```
Encoder Type: GNN
GNN Layers: 4
Edge Connectivity: 4-connected
Position Features: Enabled
GNN Dropout: 0.0
Batch Size: 16 (smaller than CNN due to memory)
```

### For Faster Training
```
Encoder Type: GNN
GNN Layers: 2
Edge Connectivity: 4-connected
Position Features: Enabled
Batch Size: 24
```

### For Maximum Expressiveness
```
Encoder Type: GNN
GNN Layers: 6
Edge Connectivity: 8-connected
Position Features: Enabled
GNN Dropout: 0.1
Batch Size: 8
```

## Comparing CNN vs GNN

### Side-by-Side Comparison

1. **Train CNN baseline:**
   - Set Encoder Type to "CNN"
   - Save config and start training
   - Note the checkpoint directory (default: `checkpoints_supervised`)

2. **Train GNN variant:**
   - Stop training if running
   - Change Encoder Type to "GNN"
   - Update checkpoint directory to `checkpoints_supervised_gnn`
   - Save config and start training

3. **Compare results:**
   - Monitor loss curves in real-time
   - Check visualization quality
   - Compare final metrics

### Expected Differences

| Metric | CNN | GNN |
|--------|-----|-----|
| **Training Speed** | Faster | ~1.5-2× slower |
| **Memory Usage** | Lower | ~1.5× higher |
| **Batch Size** | 32-64 | 16-24 recommended |
| **Connectivity Modeling** | Implicit | Explicit |

## Visualization Differences

The attention visualizations show how well each encoder segments objects:

- **Good segmentation**: Clear, sharp boundaries between slots
- **Poor segmentation**: Blurry or overlapping slot attention

Compare CNN vs GNN visualizations to see which better captures object boundaries.

## Troubleshooting

### "GNN encoder not available" error

**Solution**: Install PyTorch Geometric:
```bash
pip install torch-geometric torch-scatter torch-sparse
```

### "CUDA out of memory" error

**Solution**: Reduce batch size:
- Try 16, then 8, then 4 if needed
- GNN uses more memory than CNN

### Training very slow

**Solution**:
- Use 4-connectivity instead of 8-connectivity
- Reduce GNN layers (try 2 or 3)
- Consider using CPU if GPU is slow (set device in config)

### Loss not decreasing

**Solution**:
- Ensure position features are enabled
- Try lower learning rate (e.g., 5e-5)
- Add dropout (0.1) to prevent overfitting
- Check that ground truth masks are being generated

## Advanced: Config via Python API

You can also configure the app programmatically:

```python
import requests

config = {
    'data_dir': 'data/processed_agi1',
    'num_epochs': 100,
    'batch_size': 16,
    'lr': 0.0001,
    'num_slots': 7,
    'encoder_type': 'gnn',
    'gnn_num_layers': 4,
    'gnn_edge_connectivity': 4,
    'gnn_use_position': True,
    'gnn_dropout': 0.0,
    # ... other parameters
}

# Save config
requests.post('http://localhost:5005/api/save_config', json=config)

# Start training
requests.post('http://localhost:5005/api/start')
```

## Real-time Monitoring

The dashboard provides real-time updates:

- **Loss Chart**: Shows total loss, background loss, foreground loss
- **Matched Objects**: Number of objects successfully assigned to slots
- **Attention Visualizations**: Side-by-side comparison of predictions vs ground truth
  - Pause visualizations with the toggle button to reduce overhead

## Checkpoints

Checkpoints are saved automatically:
- Every N epochs (configurable via "Save Every N Epochs")
- Best model (lowest loss)
- Manual save available via "Save Checkpoint Now" button

GNN checkpoints include all GNN-specific parameters, so you can resume training later.

## Tips for Best Results

1. **Start with defaults**: Use 4 layers, 4-connectivity, position features enabled
2. **Monitor memory**: Watch GPU memory usage, reduce batch size if needed
3. **Compare objectively**: Run both CNN and GNN with same hyperparameters
4. **Check visualizations**: Better segmentation = better model
5. **Be patient**: GNN trains slower but may learn better connectivity patterns

## Questions?

See [GNN_USAGE.md](GNN_USAGE.md) for detailed information about the GNN implementation, or check the main implementation in:
- `encoder_gnn.py` - GNN encoder
- `model.py` - GNNSlotInstanceModel
- `app_supervised.py` - Web app integration

Happy training! 🚀

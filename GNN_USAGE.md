# GNN Encoder Usage Guide

## Overview

The IRSlot codebase now supports **Graph Neural Network (GNN)** encoders as an alternative to CNN encoders. This allows you to test whether explicit graph-based connectivity modeling improves learning for ARC grid segmentation tasks.

## Architecture Comparison

| Component | CNN Encoder | GNN Encoder |
|-----------|-------------|-------------|
| **Grid Representation** | 2D spatial grid | Graph (pixels as nodes) |
| **Inductive Bias** | Local spatial patterns (convolutions) | Connectivity patterns (message passing) |
| **Receptive Field** | 3×3 kernels, stacked layers | K-hop neighborhoods (K = num layers) |
| **Edge Definition** | N/A | 4-connectivity or 8-connectivity |
| **Position Encoding** | Implicit in convolutions | Explicit node features (optional) |
| **Output** | `[B, H*W, 64]` features | `[B, H*W, 64]` features (identical) |

Both encoders feed into the same SlotAttention mechanism, enabling direct comparison.

---

## Installation

### 1. Install PyTorch Geometric

The GNN encoder requires PyTorch Geometric. Installation depends on your PyTorch and CUDA versions:

```bash
# For PyTorch 2.0+ with CUDA 11.8
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# For PyTorch 2.0+ with CPU only
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Or install from requirements.txt (may need adjustment for your system)
pip install torch-geometric torch-scatter torch-sparse
```

See [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

### 2. Verify Installation

```python
import torch
from torch_geometric.nn import GCNConv
print("PyTorch Geometric installed successfully!")
```

---

## Usage

### Basic Training with GNN Encoder

To train with the GNN encoder instead of CNN, simply add `--encoder_type gnn`:

```bash
python train_supervised.py \
  --encoder_type gnn \
  --data_dir data/processed \
  --num_slots 7 \
  --batch_size 16 \
  --num_epochs 100 \
  --checkpoint_dir checkpoints_gnn
```

### GNN-Specific Arguments

When using `--encoder_type gnn`, you can configure:

| Argument | Default | Description |
|----------|---------|-------------|
| `--gnn_num_layers` | 4 | Number of GCN layers (depth) |
| `--gnn_edge_connectivity` | 4 | Edge connectivity: 4 or 8 |
| `--gnn_use_position` | True | Include spatial position features |
| `--gnn_dropout` | 0.0 | Dropout rate for GNN layers |

### Example: 8-Connectivity GNN

```bash
python train_supervised.py \
  --encoder_type gnn \
  --gnn_edge_connectivity 8 \
  --data_dir data/processed \
  --checkpoint_dir checkpoints_gnn_8conn
```

### Example: GNN Without Position Features

```bash
python train_supervised.py \
  --encoder_type gnn \
  --gnn_use_position False \
  --data_dir data/processed \
  --checkpoint_dir checkpoints_gnn_no_pos
```

### Example: Deeper GNN

```bash
python train_supervised.py \
  --encoder_type gnn \
  --gnn_num_layers 6 \
  --data_dir data/processed \
  --checkpoint_dir checkpoints_gnn_deep
```

---

## Comparing CNN vs GNN

### Training Both Models

**CNN Baseline:**
```bash
python train_supervised.py \
  --encoder_type cnn \
  --data_dir data/processed \
  --batch_size 32 \
  --num_epochs 100 \
  --checkpoint_dir checkpoints_cnn
```

**GNN Alternative:**
```bash
python train_supervised.py \
  --encoder_type gnn \
  --data_dir data/processed \
  --batch_size 16 \
  --num_epochs 100 \
  --checkpoint_dir checkpoints_gnn
```

*Note: GNN may require smaller batch sizes due to memory usage.*

### Expected Differences

| Metric | CNN | GNN |
|--------|-----|-----|
| **Training Speed** | Faster (optimized convolutions) | ~1.5-2× slower (scatter/gather) |
| **Memory Usage** | Lower | ~1.5× higher (graph structures) |
| **Parameters** | ~100K | ~100K (similar capacity) |
| **Convergence** | Fast | May vary (different inductive bias) |

---

## How It Works

### Graph Construction

For each grid `[H, W]`:

1. **Nodes**: Each pixel becomes a node (H×W nodes total)
2. **Node Features**:
   - One-hot color encoding (10 dimensions)
   - Normalized (x, y) position (2 dimensions, if enabled)
   - Total: 12 dimensions per node

3. **Edges**: Spatial neighbors are connected
   - **4-connectivity**: Up, down, left, right (max 4 edges per node)
   - **8-connectivity**: All 8 neighbors including diagonals (max 8 edges per node)

### Message Passing

The GNN encoder applies 4 layers of graph convolution (GCN):

```
Input: Node features [H×W, 12]
  ↓
Layer 1: GCN(12 → 128) + LayerNorm + ReLU + Residual
  ↓
Layer 2: GCN(128 → 128) + LayerNorm + ReLU + Residual
  ↓
Layer 3: GCN(128 → 128) + LayerNorm + ReLU + Residual
  ↓
Layer 4: GCN(128 → 128) + LayerNorm + ReLU + Residual
  ↓
Output Projection: Linear(128 → 64)
  ↓
Output: Features [H×W, 64]
```

Each GCN layer aggregates information from neighboring nodes, allowing connectivity patterns to propagate through the graph.

### Receptive Field

- **1 layer**: Immediate neighbors (1-hop)
- **4 layers**: 4-hop neighborhood
  - With 4-connectivity: Roughly a 9×9 spatial region
  - With 8-connectivity: Larger neighborhood

This explicit graph structure may help the model learn object boundaries defined by color connectivity.

---

## Ablation Studies

### 1. Edge Connectivity

**Research Question**: Does 8-connectivity (including diagonals) improve segmentation?

```bash
# 4-connectivity
python train_supervised.py --encoder_type gnn --gnn_edge_connectivity 4 \
  --checkpoint_dir ablations/gnn_4conn

# 8-connectivity
python train_supervised.py --encoder_type gnn --gnn_edge_connectivity 8 \
  --checkpoint_dir ablations/gnn_8conn
```

### 2. Position Features

**Research Question**: Are explicit position features necessary for GNNs?

```bash
# With position features (default)
python train_supervised.py --encoder_type gnn --gnn_use_position \
  --checkpoint_dir ablations/gnn_with_pos

# Without position features
python train_supervised.py --encoder_type gnn --gnn_use_position False \
  --checkpoint_dir ablations/gnn_no_pos
```

### 3. Network Depth

**Research Question**: How many GNN layers are optimal?

```bash
# 2 layers (shallow)
python train_supervised.py --encoder_type gnn --gnn_num_layers 2 \
  --checkpoint_dir ablations/gnn_2layers

# 4 layers (default)
python train_supervised.py --encoder_type gnn --gnn_num_layers 4 \
  --checkpoint_dir ablations/gnn_4layers

# 6 layers (deep)
python train_supervised.py --encoder_type gnn --gnn_num_layers 6 \
  --checkpoint_dir ablations/gnn_6layers
```

---

## Implementation Details

### Files Modified/Added

**New Files:**
- `encoder_gnn.py` - GNN encoder implementation
- `utils/graph_utils.py` - Graph construction utilities
- `GNN_USAGE.md` - This usage guide

**Modified Files:**
- `model.py` - Added `GNNSlotInstanceModel` class
- `train_supervised.py` - Added encoder selection and GNN arguments
- `requirements.txt` - Added PyTorch Geometric dependencies

### Key Design Decisions

1. **Interface Compatibility**: GNN encoder matches CNN encoder interface exactly
   - Input: `[B, H, W]` integer tensor
   - Output: `[B, H*W, 64]` features
   - Drop-in replacement with SlotAttention unchanged

2. **Edge Caching**: Edge indices are cached for reuse
   - Pre-computed once per (H, W, connectivity) combination
   - Stored in global cache to avoid recomputation
   - Minimal overhead after first batch

3. **Batch Processing**: Each sample processed separately through GNN
   - Simpler than PyG's disjoint union batching
   - Works because dataloader pads grids to same size

4. **Residual Connections**: Added to prevent over-smoothing
   - Critical for deep GNNs (4+ layers)
   - Maintains feature diversity across layers

---

## Troubleshooting

### ImportError: PyTorch Geometric not installed

**Solution**: Install PyG following the installation section above. Make sure your PyTorch and CUDA versions match.

### CUDA out of memory

**Solution**: Reduce batch size. GNNs use more memory than CNNs.
```bash
python train_supervised.py --encoder_type gnn --batch_size 8
```

### Training is very slow

**Solution**: This is expected. GNNs are ~1.5-2× slower than CNNs due to scatter/gather operations. Consider:
- Using GPU (if not already)
- Reducing `--gnn_num_layers` (try 2 or 3 instead of 4)
- Using 4-connectivity instead of 8-connectivity

### Model fails to converge

**Solution**: Try:
- Lower learning rate: `--lr 5e-5`
- Add dropout: `--gnn_dropout 0.1`
- Reduce depth: `--gnn_num_layers 3`
- Enable position features: `--gnn_use_position` (should be default)

---

## Evaluation Metrics

Track these metrics to compare CNN vs GNN:

| Metric | What It Measures |
|--------|-----------------|
| **Mask Dice Loss** | Segmentation quality (lower is better) |
| **Background Loss** | How well slot 0 captures background |
| **Foreground Loss** | How well slots 1-6 capture objects |
| **Avg Matched Objects** | Number of objects successfully assigned to slots |
| **Training Time/Epoch** | Computational efficiency |
| **Parameters** | Model capacity |

View metrics in training output and checkpoint files.

---

## Research Questions

The GNN encoder enables investigating:

1. **Does explicit connectivity help?** GNNs model connectivity directly via graph edges, while CNNs learn it implicitly. Which is better for ARC grids?

2. **What's the right receptive field?** CNNs use small kernels (3×3), GNNs can reach farther with fewer layers. Does this help?

3. **Are position features necessary?** CNNs have positional bias built-in. Do GNNs need explicit position encoding?

4. **4-conn vs 8-conn**: ARC objects are defined by color connectivity. Does including diagonals (8-conn) improve segmentation?

5. **Generalization**: Do GNNs generalize better to grids with different connectivity patterns?

---

## Next Steps

After training both CNN and GNN models:

1. **Compare losses**: Which achieves lower mask Dice loss?
2. **Visualize attention**: Do GNN slots segment objects more cleanly?
3. **Check boundary quality**: Do GNN models produce crisper object boundaries?
4. **Test generalization**: Evaluate on held-out ARC puzzles
5. **Try hybrids**: Could you combine CNN (local features) + GNN (connectivity)?

---

## Citation

If you use this GNN implementation, consider citing:

- **PyTorch Geometric**: Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric", ICLR 2019
- **GCN**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017
- **Slot Attention**: Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020

---

## Questions or Issues?

If you encounter problems or have questions about the GNN implementation:

1. Check this guide's Troubleshooting section
2. Verify PyTorch Geometric installation
3. Try with smaller grids/batch sizes first
4. Compare outputs with CNN encoder to ensure interface compatibility

Good luck with your experiments! 🚀

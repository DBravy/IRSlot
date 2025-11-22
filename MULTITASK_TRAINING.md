# Multi-Task Learning: ARC Solving + Contrastive Learning

This implementation combines two training objectives:
1. **ARC Puzzle Solving Loss**: Cross-entropy loss for predicting output grids
2. **Contrastive Learning Loss**: InfoNCE loss for learning object-centric representations via slot attention

## How It Works

### Architecture
```
Input Grid
    ↓
CNN Encoder → Features
    ↓
Slot Attention → Slots
    ↓         ↘
    ↓          [Pool + Project] → Embeddings → Contrastive Loss (InfoNCE)
    ↓
Transformer → Decoder → Predicted Grid → ARC Loss (Cross-Entropy)
```

### Gradient Flow
- **All slot attention parameters** receive gradients from BOTH losses
- Contrastive loss provides strong signal for object segmentation
- ARC loss provides signal for task-relevant reasoning
- Both losses backpropagate through the shared slot attention mechanism

## Usage

### Training with Multi-Task Learning

Enable contrastive learning with the `--use_contrastive` flag:

```bash
python train_arc_solver.py \
    --data_dir kaggle/combined \
    --use_contrastive \
    --contrastive_weight 1.0 \
    --contrastive_temperature 0.07 \
    --contrastive_num_negatives 512 \
    --memory_bank_momentum 0.5 \
    --batch_size 4 \
    --num_epochs 100 \
    --lr 1e-4
```

### Training without Contrastive Learning (ARC Only)

Simply omit the `--use_contrastive` flag:

```bash
python train_arc_solver.py \
    --data_dir kaggle/combined \
    --batch_size 4 \
    --num_epochs 100 \
    --lr 1e-4
```

## Key Parameters

### Contrastive Learning Parameters

- `--use_contrastive`: Enable multi-task learning (default: False)
- `--contrastive_weight`: Weight for contrastive loss in total loss (default: 1.0)
  - Total loss = ARC loss + contrastive_weight * contrastive loss
- `--contrastive_temperature`: Temperature for InfoNCE loss (default: 0.07)
  - Lower = sharper similarity distribution
- `--contrastive_num_negatives`: Number of negative samples (default: 512)
  - More negatives = harder discrimination task
- `--memory_bank_momentum`: Momentum for memory bank updates (default: 0.5)
  - 0.0 = full replacement, 1.0 = no update

### Typical Values
- **Balanced multi-task**: `--contrastive_weight 1.0`
- **Emphasize ARC solving**: `--contrastive_weight 0.1`
- **Emphasize object detection**: `--contrastive_weight 10.0`

## Training Metrics

With multi-task learning enabled, you'll see:

```
Epoch 1/100:
  Train Loss: 2.3456, Train Acc: 0.1234
    ARC Loss: 2.2000, Contrastive Loss: 0.1456
    Contrastive Acc: 0.3456
  Val Loss: 2.4567, Val Acc: 0.1123
```

### Metrics Explanation
- **Train Loss**: Total combined loss (ARC + weighted contrastive)
- **ARC Loss**: Cross-entropy loss for puzzle solving
- **Contrastive Loss**: InfoNCE loss for instance recognition
- **Contrastive Acc**: Accuracy at identifying positive vs negative samples
  - High accuracy (>0.5) means slot attention is learning good object representations

## Implementation Details

### Memory Bank
- Stores embeddings for each unique puzzle's test input
- Size: Number of puzzles in training set
- Updated with momentum during training (no gradients)
- Used to sample positive and negative pairs for contrastive learning

### What Gets Trained on What
- **Slot Attention (Q, K, V, GRU, MLP)**: Both losses ✓✓
- **CNN Encoder**: Both losses ✓✓
- **Contrastive Projection Head**: Contrastive loss only ✓
- **Transformer**: ARC loss only ✓
- **Decoder**: ARC loss only ✓

### Checkpointing
Checkpoints include memory bank state when using contrastive learning:
```python
checkpoint = {
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'memory_bank_state_dict': ...,  # Only when --use_contrastive
    'metrics': ...
}
```

## Expected Benefits

1. **Better Object Segmentation**: Contrastive loss encourages slot attention to discover distinct objects
2. **Faster Convergence**: Strong learning signal for slot attention from the start
3. **More Robust Representations**: Slots learn both object identity (contrastive) and task relevance (ARC)

## Troubleshooting

### Contrastive accuracy stuck at ~0.0
- Memory bank might need more time to stabilize
- Try reducing `--contrastive_num_negatives` (fewer negatives = easier task)
- Try increasing `--contrastive_temperature` (softer similarities)

### ARC loss not improving
- Contrastive loss might be dominating
- Try reducing `--contrastive_weight` to 0.1 or 0.01
- Or disable contrastive learning initially, then resume with it enabled

### Memory errors
- Memory bank stores embeddings for all puzzles
- Reduce `--contrastive_num_negatives` to lower memory usage
- Or reduce `--batch_size`

## Curriculum Learning (Optional)

You can start without contrastive learning and add it later:

1. Train for N epochs without contrastive:
```bash
python train_arc_solver.py --data_dir kaggle/combined --num_epochs 20
```

2. Resume with contrastive learning enabled:
```bash
python train_arc_solver.py \
    --data_dir kaggle/combined \
    --resume checkpoints_arc_solver/checkpoint_latest.pt \
    --use_contrastive \
    --num_epochs 100
```

## Code References

- Multi-task loss computation: [models/arc_solver.py:326](models/arc_solver.py#L326) (`compute_loss`)
- Contrastive projection head: [models/arc_solver.py:198](models/arc_solver.py#L198) (`__init__`)
- Training loop with memory bank: [train_arc_solver.py:152](train_arc_solver.py#L152) (`train_epoch`)

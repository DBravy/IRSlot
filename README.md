# Slot Attention for ARC with Instance Recognition

This project implements Slot Attention trained via instance recognition (contrastive learning) for the ARC (Abstraction and Reasoning Corpus) dataset.

## Overview

The model learns to:
1. **Decompose** ARC grids into object-centric slots using Slot Attention
2. **Create consistent representations** across different augmentations of the same grid
3. **Distinguish** between different grids using contrastive learning (InfoNCE loss)

## Architecture

```
ARC Grid [H, W]
    ↓
CNN Encoder → Features [B, H*W, feature_dim]
    ↓
Slot Attention → Slots [B, num_slots, slot_dim]
    ↓
Mean Pooling → Pooled [B, slot_dim]
    ↓
Projection Head → Embedding [B, embedding_dim]
    ↓
Instance Recognition (InfoNCE Loss)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

**Important:** Choose whether to train on ARC-AGI-1 or ARC-AGI-2.

⚠️ **Do NOT mix both versions!** ARC-AGI-2 training data contains some ARC-AGI-1 evaluation puzzles, which would cause data leakage.

### Option 1: Quick preparation script

```bash
# For ARC-AGI-1 (recommended for beginners)
./prepare_data.sh agi1

# For ARC-AGI-2
./prepare_data.sh agi2
```

### Option 2: Manual preparation

```bash
# For ARC-AGI-1
python build_arc_dataset.py \
    --input-file-prefix kaggle/combined/arc-agi \
    --output-dir data/processed_agi1 \
    --arc-version agi1 \
    --num-aug 100 \
    --seed 42

# For ARC-AGI-2
python build_arc_dataset.py \
    --input-file-prefix kaggle/combined/arc-agi \
    --output-dir data/processed_agi2 \
    --arc-version agi2 \
    --num-aug 100 \
    --seed 42
```

This will:
- Load ARC puzzles from the `kaggle/combined/` directory
- **AGI-1**: Use `training` and `evaluation` subsets only (evaluate on `evaluation`)
- **AGI-2**: Use **ALL** data (`training`, `training2`, `concept`, `evaluation`, `evaluation2`) for training, evaluate on `evaluation2`
- Apply augmentations (rotations, reflections, color permutations)
- Create train/test splits
- Save processed data to `data/processed_agi1/` or `data/processed_agi2/`

**Note:** For AGI-2, we can safely use ARC-AGI-1's data (including `evaluation`) for training since we only evaluate on `evaluation2`.

## Training

Train the slot attention model with instance recognition:

```bash
python train.py \
    --data_dir data/processed \
    --num_slots 7 \
    --slot_dim 64 \
    --embedding_dim 128 \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --momentum 0.5 \
    --temperature 0.07 \
    --num_negatives 2048 \
    --checkpoint_dir checkpoints \
    --save_every 10
```

### Key Hyperparameters

**Model Architecture:**
- `--num_slots`: Number of slots (default: 7) - how many objects to detect
- `--slot_dim`: Slot dimension (default: 64)
- `--num_iterations`: Slot attention iterations (default: 3)
- `--embedding_dim`: Final embedding dimension (default: 128)

**Instance Recognition:**
- `--momentum`: Memory bank momentum (default: 0.5) - how fast embeddings update
- `--temperature`: InfoNCE temperature (default: 0.07) - lower = sharper similarities
- `--num_negatives`: Number of negative samples (default: 2048)

**Training:**
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)

## How Instance Recognition Works

### Memory Bank
- Stores one embedding vector per unique grid in the dataset
- Updated with momentum: `bank[id] = 0.5 * old_emb + 0.5 * new_emb`
- Provides negative samples for contrastive learning

### Training Process
1. Sample a batch of grids (with augmentations)
2. Forward pass → get current embeddings
3. Retrieve stored embeddings from memory bank (positive pairs)
4. Sample random embeddings as negatives
5. Compute InfoNCE loss:
   - Maximize similarity between current and stored embeddings (same grid)
   - Minimize similarity with negative samples (different grids)
6. Update memory bank with new embeddings

### Loss Function (InfoNCE)

```
L = -log(exp(sim(current, stored) / τ) / (exp(sim(current, stored) / τ) + Σ exp(sim(current, neg_i) / τ)))
```

Where:
- `current`: Current batch embeddings
- `stored`: Stored embeddings from memory bank (positive)
- `neg_i`: Negative samples from memory bank
- `τ`: Temperature parameter

## Project Structure

```
.
├── slot.py                  # Slot Attention implementation
├── encoder.py               # CNN encoder for ARC grids
├── model.py                 # Full model (Encoder + Slots + Projection)
├── memory_bank.py           # Memory bank for instance recognition
├── loss.py                  # InfoNCE loss function
├── augmentation.py          # Data augmentation utilities
├── train.py                 # Training script
├── build_arc_dataset.py     # Dataset preprocessing
├── dataset/
│   ├── common.py           # Data structures and transforms
│   └── arc_dataset.py      # PyTorch Dataset class
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Checkpoints

Checkpoints are saved in the `--checkpoint_dir` directory:
- `checkpoint_epoch_N.pt`: Saved every `--save_every` epochs
- `best_model.pt`: Best model based on training loss
- `args.json`: Training arguments

### Loading a Checkpoint

```python
import torch
from model import SlotInstanceModel

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Create model
model = SlotInstanceModel(**checkpoint['args'])
model.load_state_dict(checkpoint['model_state_dict'])

# Use model
embeddings, slots = model(grids)
```

## Expected Results

During training, you should see:
- **Loss** decreasing over time
- **Accuracy** increasing (positive pairs ranked highest)
- **Positive similarity** higher than negative similarity
- Stable convergence after 50-100 epochs

## Citation

This implementation is based on:

```
@inproceedings{locatello2020object,
  title={Object-centric learning with slot attention},
  author={Locatello, Francesco and Weissenborn, Dirk and Unterthiner, Thomas and Mahendran, Aravindh and Heigold, Georg and Uszkoreit, Jakob and Dosovitskiy, Alexey and Kipf, Thomas},
  booktitle={NeurIPS},
  year={2020}
}
```

Instance Recognition methodology inspired by:
```
@inproceedings{wu2018unsupervised,
  title={Unsupervised feature learning via non-parametric instance discrimination},
  author={Wu, Zhirong and Xiong, Yuanjun and Yu, Stella X and Lin, Dahua},
  booktitle={CVPR},
  year={2018}
}
```

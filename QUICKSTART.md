# Quick Start Guide

Get up and running with Slot Attention training in 3 steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare the Dataset

**Choose your ARC version:**
- `agi1` - Train/eval on ARC-AGI-1 only
- `agi2` - Train on ALL data, evaluate on ARC-AGI-2 (maximum training data!)

Make sure your ARC dataset JSON files are in `kaggle/combined/`, then run:

### For ARC-AGI-1 (default):
```bash
./prepare_data.sh agi1
```

### For ARC-AGI-2:
```bash
./prepare_data.sh agi2
```

Or manually:

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

This will create augmented training data in `data/processed_agi1/` or `data/processed_agi2/`.

**Note:** AGI-2 uses ALL available ARC data (including AGI-1 training + evaluation) to maximize training data, but only evaluates on `evaluation2`.

## Step 3: Train the Model

```bash
# For ARC-AGI-1
./quick_train.sh agi1

# For ARC-AGI-2
./quick_train.sh agi2
```

Or manually:

```bash
python train.py \
    --data_dir data/processed_agi1 \
    --num_slots 7 \
    --slot_dim 64 \
    --embedding_dim 128 \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints_agi1
```

Training progress will be displayed with a progress bar showing:
- Loss
- Accuracy (how often positive pairs are ranked highest)
- Positive similarity (should be high)
- Negative similarity (should be low)

Checkpoints will be saved to `checkpoints/`.

### 🌐 Recommended: Train with Web Dashboard

For a beautiful real-time monitoring experience with **full training control**:

```bash
python app.py
```

That's it! Then open **http://localhost:5000** in your browser to:
- ⚙️ **Configure everything in the browser** - no command-line arguments needed!
- ✨ Live updating graphs (Loss, Accuracy, Similarities)
- 🎮 **Start/Stop/Pause controls** - training waits for you!
- 📊 Real-time metrics display
- 🎨 Modern dark theme UI
- 📈 Interactive Plotly charts

Fill in the configuration form and click **"▶ Start Training"** when you're ready!

See [WEB_DASHBOARD.md](WEB_DASHBOARD.md) for full details!

## Step 4: Test the Model (Optional)

After training, you can test the model:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data/processed \
    --num_samples 10
```

This will show you:
- Embeddings for each grid
- Slot representations
- Grid previews

## Expected File Structure

```
IRSlot/
├── kaggle/combined/          # Input: Raw ARC JSON files
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   └── ...
├── data/processed/           # Output: Processed dataset
│   ├── train/
│   │   ├── all__inputs.npy
│   │   ├── all__labels.npy
│   │   └── ...
│   └── test/
├── checkpoints/              # Output: Model checkpoints
│   ├── best_model.pt
│   ├── checkpoint_epoch_10.pt
│   └── args.json
└── ...
```

## Troubleshooting

**Import errors**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

**CUDA out of memory**: Reduce `--batch_size` (try 16 or 8)

**Data not found**: Ensure you've run `prepare_data.sh` first

**Slow training**: Enable GPU with `--device cuda` if you have one

## What's Happening During Training?

1. **Encoder** converts each ARC grid to feature maps
2. **Slot Attention** decomposes features into object-centric slots
3. **Pooling** aggregates slots into a single embedding
4. **Memory Bank** stores embeddings for each unique grid
5. **InfoNCE Loss** trains the model to:
   - Make embeddings similar for augmented versions of the same grid
   - Make embeddings different for different grids

The model learns to create consistent, object-centric representations!

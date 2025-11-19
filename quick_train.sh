#!/bin/bash
# Quick start script for training

echo "========================================="
echo "Slot Attention Training (Quick Start)"
echo "========================================="
echo ""

# Default values
ARC_VERSION="${1:-agi1}"  # Default to agi1
DATA_DIR="data/processed_${ARC_VERSION}"
CHECKPOINT_DIR="checkpoints_${ARC_VERSION}"

# Validate ARC_VERSION
if [[ "$ARC_VERSION" != "agi1" && "$ARC_VERSION" != "agi2" && "$ARC_VERSION" != "both" ]]; then
    echo "Error: Invalid ARC version '$ARC_VERSION'"
    echo "Usage: $0 [agi1|agi2|both]"
    echo ""
    echo "  agi1 - Train on ARC-AGI-1"
    echo "  agi2 - Train on ARC-AGI-2"
    echo "  both - Train on both (NOT recommended)"
    exit 1
fi

# Check if data exists
if [ ! -d "$DATA_DIR/train" ]; then
    echo "Error: Training data not found at $DATA_DIR/train"
    echo "Please run './prepare_data.sh $ARC_VERSION' first to prepare the dataset."
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Run training with default hyperparameters
echo "ARC Version: $ARC_VERSION"
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""
echo "Starting training with default hyperparameters..."
echo ""

python train.py \
    --data_dir "$DATA_DIR" \
    --num_slots 7 \
    --slot_dim 64 \
    --embedding_dim 128 \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --momentum 0.5 \
    --temperature 0.07 \
    --num_negatives 2048 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --save_every 10

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Training completed!"
    echo "Checkpoints saved to: $CHECKPOINT_DIR"
    echo "========================================="
else
    echo ""
    echo "Error: Training failed!"
    exit 1
fi

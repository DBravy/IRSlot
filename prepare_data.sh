#!/bin/bash
# Script to prepare the ARC dataset for training

echo "========================================="
echo "ARC Dataset Preparation"
echo "========================================="
echo ""

# Default values
INPUT_PREFIX="kaggle/combined/arc-agi"
OUTPUT_DIR="data/processed"
ARC_VERSION="${1:-agi1}"  # Default to agi1, can pass agi1, agi2, or both as argument
NUM_AUG=100
SEED=42

# Validate ARC_VERSION
if [[ "$ARC_VERSION" != "agi1" && "$ARC_VERSION" != "agi2" && "$ARC_VERSION" != "both" ]]; then
    echo "Error: Invalid ARC version '$ARC_VERSION'"
    echo "Usage: $0 [agi1|agi2|both]"
    echo ""
    echo "  agi1 - Train on ARC-AGI-1 only (training + evaluation)"
    echo "  agi2 - Train on ALL data, evaluate on evaluation2 (recommended for max data)"
    echo "  both - Same as agi2 (kept for compatibility)"
    exit 1
fi

# Check if kaggle directory exists
if [ ! -d "kaggle/combined" ]; then
    echo "Error: kaggle/combined directory not found!"
    echo "Please ensure the ARC dataset JSON files are in kaggle/combined/"
    exit 1
fi

# Adjust output directory to include version
OUTPUT_DIR="${OUTPUT_DIR}_${ARC_VERSION}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run dataset preparation
echo "ARC Version: $ARC_VERSION"
echo "Input prefix: $INPUT_PREFIX"
echo "Output directory: $OUTPUT_DIR"
echo "Number of augmentations: $NUM_AUG"
echo "Random seed: $SEED"
echo ""

python build_arc_dataset.py \
    --input-file-prefix "$INPUT_PREFIX" \
    --output-dir "$OUTPUT_DIR" \
    --arc-version "$ARC_VERSION" \
    --num-aug "$NUM_AUG" \
    --seed "$SEED"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Dataset preparation completed!"
    echo "Data saved to: $OUTPUT_DIR"
    echo "========================================="
else
    echo ""
    echo "Error: Dataset preparation failed!"
    exit 1
fi

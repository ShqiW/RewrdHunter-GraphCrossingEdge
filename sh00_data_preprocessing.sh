#!/bin/bash
# Data Preprocessing Pipeline for RewardHunter
# This script downloads and preprocesses the Rome graph dataset

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_DIR/data"

echo "=========================================="
echo "RewardHunter Data Preprocessing Pipeline"
echo "=========================================="

# Step 1: Create data directory
echo ""
echo "[Step 1/5] Creating data directory..."
mkdir -p "$DATA_DIR"

# Step 2: Download Rome dataset
echo ""
echo "[Step 2/5] Downloading Rome dataset..."
if [ -f "$DATA_DIR/rome.tgz" ]; then
    echo "  -> rome.tgz already exists, skipping download"
else
    wget https://graphdrawing.unipg.it/data/rome-graphml.tgz -O "$DATA_DIR/rome.tgz"
    echo "  -> Downloaded rome.tgz"
fi

# Step 3: Extract dataset
echo ""
echo "[Step 3/5] Extracting dataset..."
if [ -d "$DATA_DIR/rome" ] && [ "$(ls -A "$DATA_DIR/rome")" ]; then
    echo "  -> rome/ directory already exists and is not empty, skipping extraction"
else
    mkdir -p "$DATA_DIR/rome"
    tar -xzf "$DATA_DIR/rome.tgz" -C "$DATA_DIR/rome" --strip-components=1
    echo "  -> Extracted to data/rome/"
fi

# Step 4: Analyze and split dataset (train/test)
echo ""
echo "[Step 4/5] Analyzing and splitting dataset..."
cd "$DATA_DIR"
poetry run python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/scripts')
from importlib import import_module
script = import_module('00_analyze_dataset')

# Analyze with correct path (we're in data/ directory)
nums, file_map = script.analyze_dataset(data_dir='rome')

# Split into train and test
script.split_dataset(nums, file_map, train_cutoff=9999, test_start=10000, test_end=10100)
"
echo "  -> Created train_graph.txt and test_graph.txt"

# Step 5: Preprocess graphs (compute features)
echo ""
echo "[Step 5/5] Preprocessing graphs (computing features)..."
cd "$PROJECT_DIR"
poetry run python -c "
from src.data.rome import RomeDataset

print('Processing train dataset...')
train_ds = RomeDataset(root='data', split='train', force_reload=True)
print(f'Train: {len(train_ds)} graphs')

print('Processing test dataset...')
test_ds = RomeDataset(root='data', split='test', force_reload=True)
print(f'Test: {len(test_ds)} graphs')

# Verify
sample = train_ds[0]
print(f'Sample: {sample.graph_name}, nodes={sample.num_nodes}, tau={sample.tau:.4f}')
"

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "=========================================="
echo "Data structure:"
echo "  data/"
echo "    ├── rome/              # Raw GraphML files"
echo "    ├── processed/"
echo "    │   ├── train/         # Processed train graphs (.pt)"
echo "    │   └── test/          # Processed test graphs (.pt)"
echo "    ├── train_graph.txt    # Train file list"
echo "    └── test_graph.txt     # Test file list"

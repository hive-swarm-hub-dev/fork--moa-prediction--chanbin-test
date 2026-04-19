#!/bin/bash
set -e

echo "=== Preparing MoA Prediction Task ==="

# Install dependencies
pip3 install -r requirements.txt -q

# Download data from HuggingFace
if [ ! -f data/train_features.csv ]; then
    echo "Downloading data from HuggingFace..."
    mkdir -p data
    HF_BASE="https://huggingface.co/datasets/beaniscold/hive-moa-prediction/resolve/main"
    curl -L -o data/train_features.csv "$HF_BASE/train_features.csv"
    curl -L -o data/train_targets.csv "$HF_BASE/train_targets.csv"
    curl -L -o data/test_features.csv "$HF_BASE/test_features.csv"
    echo "Download complete."
else
    echo "Data already exists, skipping download."
fi

echo "=== Setup complete ==="
echo "Train features: data/train_features.csv"
echo "Train targets:  data/train_targets.csv"
echo "Test features:  data/test_features.csv"

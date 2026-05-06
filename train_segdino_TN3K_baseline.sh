#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_SCRIPT="train_segdino_refactored.py"
SPLIT_SCRIPT="prepare_tn3k_split.py"
DATASET_ROOT="../datasets/TN3K"
SEED=42
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_SCRIPT_PATH="$SCRIPT_DIR/$SPLIT_SCRIPT"

if [ ! -f "$SPLIT_SCRIPT_PATH" ]; then
    SPLIT_SCRIPT_PATH="$SCRIPT_DIR/../datasets/TN3K/$SPLIT_SCRIPT"
fi


if [ ! -f "$SPLIT_SCRIPT_PATH" ]; then
    echo "Error: $SPLIT_SCRIPT not found!"
    exit 1
fi

# Check if python script exists
if [ ! -f "$SCRIPT_DIR/$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    echo "Error: torch not found in interpreter: $PYTHON_BIN"
    echo "Tip: set PYTHON_BIN to your training environment Python."
    echo "Example: PYTHON_BIN=/data/xhschulz/.venv/bin/python bash segdino_fork/train_segdino_TN3K_baseline.sh"
    exit 1
fi

cd "$SCRIPT_DIR"

"$PYTHON_BIN" "$SPLIT_SCRIPT_PATH" \
    --dataset_root "$DATASET_ROOT" \
    --train_ratio 0.8 \
    --seed "$SEED" \
    --img_ext .jpg \
    --mask_ext .jpg

"$PYTHON_BIN" "$PYTHON_SCRIPT" \
    --data_dir ../datasets \
    --dataset TN3K \
    --img_ext .jpg \
    --gt_ext .jpg \
    --train_split train \
    --val_split val \
    --epochs 50 \
    --batch_size 4 \
    --seed "$SEED" \
    --input_h 256 \
    --input_w 256 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --repo_dir ../dinov3 \
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --dino_size s \
    --img_dir_name image \
    --gt_dir_name mask
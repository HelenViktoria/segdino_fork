#!/bin/bash

# Parse command line arguments
PYTHON_SCRIPT="test_segdino_wo_tiling.py"

# Check if python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

# Pass all arguments to the python script
python "$PYTHON_SCRIPT" \
    --data_dir ../datasets \
    --dataset TN3K \
    --img_ext .jpg \
    --gt_ext .jpg \
    --test_split test \
    --input_h 256 \
    --input_w 256 \
    --batch_size 1 \
    --num_workers 16 \
    --ckpt ../trainings/segdino_s_TN3K_2026-05-06T17-53-42.746837/ckpts/latest.pth \
    --repo_dir ../dinov3 \
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --dino_size s \
    --img_dir_name test-image \
    --gt_dir_name test-mask
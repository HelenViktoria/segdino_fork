#!/bin/bash

# Parse command line arguments
PYTHON_SCRIPT="test_segdino_refactored.py"

# Check if python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

# Pass all arguments to the python script
python "$PYTHON_SCRIPT" \
    --data_dir ../datasets \
    --dataset dis5k \
    --mask_ext .png \
    --test_split DIS-TE1 \
    --input_h 256 \
    --input_w 256 \
    --batch_size 1 \
    --num_workers 4 \
    --num_classes 1 \
    --ckpt ../trainings/segdino_s_dis5k_2026-01-03T15:43:45.939508/ckpts/best.pth \
    --repo_dir ../dinov3 \
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --dino_size s \
    --img_dir_name im \
    --label_dir_name gt

python "$PYTHON_SCRIPT" \
    --data_dir ../datasets \
    --dataset dis5k \
    --mask_ext .png \
    --test_split DIS-TE2 \
    --input_h 256 \
    --input_w 256 \
    --batch_size 1 \
    --num_workers 4 \
    --num_classes 1 \
    --ckpt ../trainings/segdino_s_dis5k_2026-01-03T15:43:45.939508/ckpts/best.pth \
    --repo_dir ../dinov3 \
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --dino_size s \
    --img_dir_name im \
    --label_dir_name gt

python "$PYTHON_SCRIPT" \
    --data_dir ../datasets \
    --dataset dis5k \
    --mask_ext .png \
    --test_split DIS-TE3 \
    --input_h 256 \
    --input_w 256 \
    --batch_size 1 \
    --num_workers 4 \
    --num_classes 1 \
    --ckpt ../trainings/segdino_s_dis5k_2026-01-03T15:43:45.939508/ckpts/best.pth \
    --repo_dir ../dinov3 \
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --dino_size s \
    --img_dir_name im \
    --label_dir_name gt

python "$PYTHON_SCRIPT" \
    --data_dir ../datasets \
    --dataset dis5k \
    --mask_ext .png \
    --test_split DIS-TE4 \
    --input_h 256 \
    --input_w 256 \
    --batch_size 1 \
    --num_workers 4 \
    --num_classes 1 \
    --ckpt ../trainings/segdino_s_dis5k_2026-01-03T15:43:45.939508/ckpts/best.pth \
    --repo_dir ../dinov3 \
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --dino_size s \
    --img_dir_name im \
    --label_dir_name gt
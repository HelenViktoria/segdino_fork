# Parse command line arguments
$PYTHON_SCRIPT = "train_segdino_refactored.py"

# Check if python script exists
if (-not (Test-Path $PYTHON_SCRIPT)) {
    Write-Error "Error: $PYTHON_SCRIPT not found!"
    exit 1
}

# Pass all arguments to the python script
python $PYTHON_SCRIPT `
    --data_dir ../datasets `
    --dataset dis5k `
    --img_ext .jpg `
    --gt_ext .png `
    --train_split DIS-TR `
    --val_split DIS-VD `
    --epochs 5 `
    --batch_size 4 `
    --seed 42 `
    --input_h 256 `
    --input_w 256 `
    --lr 1e-4 `
    --weight_decay 1e-4 `
    --num_workers 4 `
    --repo_dir ../dinov3 `
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth `
    --dino_size s `
    --img_dir_name im `
    --gt_dir_name gt

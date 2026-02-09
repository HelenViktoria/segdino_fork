# Parse command line arguments
$PYTHON_SCRIPT = "test_segdino_refactored.py"

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
    --mask_ext .png `
    --test_split DIS-TE1 `
    --batch_size 1 `
    --input_h 256 `
    --input_w 256 `
    --num_workers 0 `
    --num_classes 1 `
    --repo_dir ../dinov3 `
    --dino_ckpt ./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth `
    --ckpt ../trainings\segdino_s_dis5k_2026-02-07T11-15-43.939265\ckpts\latest.pth `
    --dino_size s `
    --img_dir_name im `
    --label_dir_name gt

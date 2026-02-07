import argparse
import os
import sys
from datetime import datetime

import torch
import numpy as np

from dpt import DPT
from dino import load_dino_backbone

# Get the parent directory and add to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from spider_utils.dataset import FolderTupleDataset
from spider_utils.transforms import make_random_crop_transform
from spider_utils.train import train_one_epoch, validate, set_seed, plot_train_metrics
from spider_utils.model_utils import load_ckpt, save_ckpt


### Main function that performs model training and validation
def main():

    ### Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--in_ch", type=int, default=1)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, required=True,help="Path to the pretrained DINO checkpoint (.pth). "
                         "Use ViT-B/16 checkpoint for --dino_size b, "
                         "or ViT-S/16 checkpoint for --dino_size s.")
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"],
                        help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--last_layer_idx", type=int, default=-1)
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    args = parser.parse_args()

    ### Set random seed for reproducibility
    set_seed(args.seed)


    ### Create directories for saving results and checkpoints
    iso_time = datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')
    save_root = f"../trainings/segdino_{args.dino_size}_{args.dataset}_{iso_time}"
    os.makedirs(save_root, exist_ok=True)
    
    ckpt_dir = os.path.join(save_root, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_path = os.path.join(ckpt_dir, "latest.pth")
    best_path = os.path.join(ckpt_dir, "best.pth")  


    ### Load DINO backbone and initialize DPT model
    backbone = load_dino_backbone(repo_dir=args.repo_dir, dino_size=args.dino_size, dino_ckpt=args.dino_ckpt)
    model = DPT(nclass=args.num_classes, backbone=backbone)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")
    model = model.to(device)
    

    ### Prepare optimizer + loss function
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fct = torch.nn.BCEWithLogitsLoss() if args.num_classes == 1 else torch.nn.CrossEntropyLoss()


    ### Load segmentation checkpoint if provided
    if args.model_ckpt is not None:
        print(f"[Load segmentation ckpt] {args.model_ckpt}")
        load_ckpt(model, args.model_ckpt, map_location=device)
        print(f"[Info] Successfully loaded segmentation ckpt from {args.model_ckpt}")
    else:
        print("[Info] No segmentation ckpt provided, training from scratch.")
    

    ### Prepare datasets and dataloaders
    root = os.path.join(args.data_dir, args.dataset)
    train_transform = make_random_crop_transform(size=(args.input_h, args.input_w))
    val_transform   = make_random_crop_transform(size=(args.input_h, args.input_w))

    train_dataset = FolderTupleDataset(
        root=root,
        split=args.train_split,
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        transform=train_transform,
    )
    val_dataset = FolderTupleDataset(
        root=root,
        split=args.val_split,
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )


    ### Training and validation loop
    best_val_dice = -1.0
    best_val_dice_epoch = -1
    best_val_iou  = -1.0
    best_val_iou_epoch  = -1

    dice_array = np.zeros(args.epochs)
    iou_array = np.zeros(args.epochs)


    for epoch in range(1, args.epochs + 1):
        _, _, _ = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fct,
            device,
            num_classes=args.num_classes,
            is_logits=True,
            dice_thr=0.5,
            epoch=epoch
        )
        _, val_dice, val_iou = validate(
            model,
            val_loader,
            loss_fct,
            device,
            num_classes=args.num_classes,
            is_logits=True,
            dice_thr=0.5
        )

        # Store metrics
        dice_array[epoch-1] = val_dice
        iou_array[epoch-1] = val_iou

        # Save latest checkpoint
        save_ckpt(model, epoch, val_dice, val_iou, latest_path)
        print(f"[Save] Latest ckpt: {latest_path}")

        # Save best checkpoints based on validation Dice and IoU
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_dice_epoch = epoch
            save_ckpt(model, epoch, val_dice, val_iou, best_path)
            print(f"[Save] New best ckpt: {best_path}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_iou_epoch = epoch


    ### Save + plot metric arrays
    metric_save_dir = os.path.join(save_root, "metrics")
    os.makedirs(metric_save_dir, exist_ok=True)
    
    dice_path = os.path.join(metric_save_dir, "dice_array.npy")
    np.save(dice_path, dice_array)
    print(f"[Save] Saved dice array -> {dice_path}")
    
    iou_path = os.path.join(metric_save_dir, "iou_array.npy")
    np.save(iou_path, iou_array)
    print(f"[Save] Saved iou array -> {iou_path}")

    plot_train_metrics(dice_array, iou_array, metric_save_dir)
    print(f"[Save] Saved metrics plot -> {metric_save_dir}/metrics_plot.png")

    print("=" * 60)
    print(f"[Summary] Best Val Dice = {best_val_dice:.4f} @ epoch {best_val_dice_epoch}")
    print(f"[Summary] Best Val IoU  = {best_val_iou:.4f}  @ epoch {best_val_iou_epoch}")
    print("=" * 60)


if __name__ == "__main__":
    main()
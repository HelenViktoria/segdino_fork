import argparse
import os
import sys
from datetime import datetime

import torch

from dino import load_dino_backbone
from dpt import DPT

# Get the parent directory and add to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from spider_utils.dataset import FolderDataset
from spider_utils.transforms import ResizeAndNormalize
from spider_utils.model_utils import load_ckpt
from spider_utils.test import run_test


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--input_h", type=int, default=1024)
    parser.add_argument("--input_w", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--dice_thr", type=float, default=0.5)

    # Segmentation model checkpoint (DPT + decoder)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the trained segmentation model checkpoint (.pth).")

    parser.add_argument("--img_dir_name", type=str, default="Original")
    parser.add_argument("--label_dir_name", type=str, default="Ground truth")

    # DINO backbone configuration
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"],
                        help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--dino_ckpt", type=str, required=True,
                        help="Path to the pretrained DINO checkpoint (.pth). "
                             "Use ViT-B/16 checkpoint for --dino_size b, or ViT-S/16 for --dino_size s.")
    parser.add_argument("--repo_dir", type=str, default="./dinov3",
                        help="Local path to the DINOv3 torch.hub repo (contains hubconf.py).")

    args = parser.parse_args()


    ### Output directories
    iso_time = datetime.now().isoformat()
    save_root = f"../tests/segdino_{args.dino_size}_{args.dataset}_{args.ckpt}_{args.test_split}_{iso_time}"
    os.makedirs(save_root, exist_ok=True)

    vis_dir   = os.path.join(save_root, "test_vis")
    csv_path  = os.path.join(save_root, "test_metrics.csv")


    # Load DINO backbone depending on size
    backbone = load_dino_backbone(repo_dir=args.repo_dir, dino_size=args.dino_size, dino_ckpt=args.dino_ckpt)
    model = DPT(nclass=args.num_classes, backbone=backbone)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")
    model = model.to(device)

    print(f"[Load segmentation ckpt] {args.ckpt}")
    load_ckpt(model, None, args.ckpt, map_location=device)


    # Dataset and DataLoader
    root = os.path.join(args.data_dir, args.dataset)
    test_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    
    test_dataset = FolderDataset(
        root=root,
        split=args.test_split,
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=test_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


    # Run evaluation
    run_test(
        model,
        test_loader,
        device,
        is_logits=True,
        dice_thr=args.dice_thr,
        vis_dir=vis_dir,
        csv_path=csv_path
    )


if __name__ == "__main__":
    main()
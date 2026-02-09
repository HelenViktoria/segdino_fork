from argparse import ArgumentParser
import os
import sys
from datetime import datetime

from torch.utils.data import DataLoader
import torch

from dino import load_dino_backbone
from dpt import DPT

# Get the parent directory and add to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from spider_utils.dataset import FolderTupleDataset
from spider_utils.transforms import make_tile_normalize_transform
from spider_utils.model_utils import load_ckpt
from spider_utils.test import run_test_with_tiling



def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--img_ext", type=str, default=".png")
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
    iso_time = datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')
    save_root = f"../tests/segdino_{args.dino_size}_{args.dataset}_{args.test_split}_{iso_time}"
    os.makedirs(save_root, exist_ok=True)

    vis_dir   = os.path.join(save_root, "test_vis")
    os.makedirs(vis_dir, exist_ok=True)
    csv_path  = os.path.join(save_root, "test_metrics.csv")


    # Load DINO backbone depending on size
    backbone = load_dino_backbone(repo_dir=args.repo_dir, dino_size=args.dino_size, dino_ckpt=args.dino_ckpt)
    model = DPT(nclass=args.num_classes, backbone=backbone)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")
    model = model.to(device)

    print(f"[Load segmentation ckpt] {args.ckpt}")
    load_ckpt(model, args.ckpt, map_location=device)


    # Dataset and DataLoader
    root = os.path.join(args.data_dir, args.dataset)
    test_transform = make_tile_normalize_transform()
    
    test_dataset = FolderTupleDataset(
        root=root,
        split=args.test_split,
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


    # Run evaluation
    mean_dice, mean_iou = run_test_with_tiling(
        model=model,
        test_loader=test_loader,
        device=device,
        tile_size=(256, 256),
        overlap=(20, 20),
        vis_dir=vis_dir,
        csv_path=csv_path,
        is_logits=True
    )

    print("=" * 60)
    print(f"[Test Summary] Dice={mean_dice:.4f}  IoU={mean_iou:.4f}")
    print("=" * 60)



if __name__ == "__main__":
    main()
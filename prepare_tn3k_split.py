#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val folders from TN3K trainval-image/trainval-mask."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_ext", type=str, default=".jpg")
    parser.add_argument("--mask_ext", type=str, default=".jpg")
    return parser.parse_args()


def collect_pairs(
    image_dir: Path,
    mask_dir: Path,
    img_ext: str,
    mask_ext: str,
) -> list[tuple[Path, Path]]:
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == img_ext.lower()
    )
    pairs: list[tuple[Path, Path]] = []

    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}{mask_ext}"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask file missing for {image_path.name}: {mask_path}")
        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {image_dir} and {mask_dir}")

    return pairs


def reset_split_dir(split_dir: Path) -> None:
    if split_dir.exists():
        shutil.rmtree(split_dir)
    (split_dir / "image").mkdir(parents=True, exist_ok=True)
    (split_dir / "mask").mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs: list[tuple[Path, Path]], split_dir: Path) -> None:
    image_out_dir = split_dir / "image"
    mask_out_dir = split_dir / "mask"

    for image_path, mask_path in pairs:
        shutil.copy2(image_path, image_out_dir / image_path.name)
        shutil.copy2(mask_path, mask_out_dir / mask_path.name)


def main() -> None:
    args = parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    dataset_root = args.dataset_root.resolve()
    trainval_image_dir = dataset_root / "trainval-image"
    trainval_mask_dir = dataset_root / "trainval-mask"

    if not trainval_image_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {trainval_image_dir}")
    if not trainval_mask_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {trainval_mask_dir}")

    pairs = collect_pairs(trainval_image_dir, trainval_mask_dir, args.img_ext, args.mask_ext)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    train_count = int(len(pairs) * args.train_ratio)
    train_count = min(max(train_count, 1), len(pairs) - 1)

    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    reset_split_dir(train_dir)
    reset_split_dir(val_dir)

    copy_pairs(train_pairs, train_dir)
    copy_pairs(val_pairs, val_dir)

    manifest = {
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "img_ext": args.img_ext,
        "mask_ext": args.mask_ext,
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "train_files": [image_path.name for image_path, _ in train_pairs],
        "val_files": [image_path.name for image_path, _ in val_pairs],
    }
    manifest_path = dataset_root / "train_val_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"Prepared TN3K split at {dataset_root}: "
        f"train={len(train_pairs)} val={len(val_pairs)} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
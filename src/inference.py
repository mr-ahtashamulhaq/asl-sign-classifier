"""
src/inference.py
ASL Sign Language Classifier — Inference + Submission Generator

Generates a competition-ready submissions/submission.csv using
Test-Time Augmentation (TTA) for a free accuracy boost.

Usage (Kaggle cell):
    !python src/inference.py \
        --data_dir /kaggle/input/asl-sign-language-dataset \
        --ckpt     /kaggle/working/checkpoints/best_model.pth \
        --output   /kaggle/working/submissions/submission.csv \
        --tta      5

Usage (local):
    python src/inference.py --data_dir ./data --ckpt checkpoints/best_model.pth
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


# ──────────────────────── transforms ─────────────────────────────
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]


def _norm():
    return [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]


TTA_TRANSFORMS = [
    # 1. Clean centre crop (baseline)
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        *_norm()
    ]),
    # 2. Horizontal flip
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        *_norm()
    ]),
    # 3. Slight clockwise rotation
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomRotation((8, 8)),
        *_norm()
    ]),
    # 4. Slight counter-clockwise rotation
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomRotation((-8, -8)),
        *_norm()
    ]),
    # 5. Brightness & contrast shift
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        *_norm()
    ]),
]


# ──────────────────────── dataset ────────────────────────────────
class TestDataset(Dataset):
    """Flat test directory — no labels, just filenames."""

    def __init__(self, test_dir: str, transform):
        paths = sorted(Path(test_dir).glob("*.jpg"))
        if not paths:
            paths = sorted(Path(test_dir).glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No images found in {test_dir}")
        self.paths     = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        return self.transform(img), path.name


# ──────────────────────── inference ──────────────────────────────
@torch.no_grad()
def run_tta(model: nn.Module, test_dir: str,
            idx_to_class: dict, tta_count: int,
            batch_size: int, device: torch.device):
    """
    Run TTA inference.
    Returns: sorted list of (image_id, predicted_label)
    """
    model.eval()
    augmentations = TTA_TRANSFORMS[:tta_count]
    accum: dict[str, torch.Tensor] = {}   # filename → accumulated probabilities

    for aug_idx, aug in enumerate(augmentations):
        print(f"  TTA pass {aug_idx+1}/{len(augmentations)} …")
        ds     = TestDataset(test_dir, aug)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

        for imgs, fnames in tqdm(loader, leave=False):
            imgs  = imgs.to(device, non_blocking=True)
            probs = torch.softmax(model(imgs), dim=-1).cpu()

            for fname, p in zip(fnames, probs):
                if fname not in accum:
                    accum[fname] = torch.zeros_like(p)
                accum[fname] += p

    # Argmax of averaged probabilities
    results = []
    for image_id, avg_probs in sorted(accum.items()):
        pred_label = idx_to_class[avg_probs.argmax().item()]
        results.append((image_id, pred_label))

    return results


# ──────────────────────────── main ───────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--output",      default="submissions/submission.csv")
    p.add_argument("--tta",         type=int, default=5,
                   help="Number of TTA passes (1–5). 1 = no TTA.")
    p.add_argument("--batch_size",  type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load checkpoint ──────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    arch         = ckpt.get("arch", "efficientnet_b3")
    num_classes  = len(class_to_idx)

    print(f"Architecture : {arch}")
    print(f"Classes      : {num_classes}")
    print(f"Best val acc : {ckpt.get('best_val_acc', 'N/A')}")

    model = build_model(arch, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"])

    # ── run inference ────────────────────────────────────────────
    test_dir  = os.path.join(args.data_dir, "test")
    tta_count = max(1, min(args.tta, len(TTA_TRANSFORMS)))
    print(f"\nRunning TTA={tta_count} on: {test_dir}")

    results = run_tta(model, test_dir, idx_to_class,
                      tta_count=tta_count,
                      batch_size=args.batch_size,
                      device=device)

    # ── write submission ─────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        writer.writerows(results)

    # Sanity checks
    labels_used   = sorted({r[1] for r in results})
    valid_labels  = set(class_to_idx.keys())
    invalid       = set(labels_used) - valid_labels

    print(f"\n{'='*50}")
    print(f"Submission   : {args.output}")
    print(f"Total rows   : {len(results)}")
    print(f"Labels used  : {labels_used}")
    if invalid:
        print(f"⚠  INVALID LABELS FOUND: {invalid}")
    else:
        print(f"✓  All labels are valid")
    print(f"{'='*50}")

    # Preview first 5 rows
    print("\nFirst 5 predictions:")
    for image_id, label in results[:5]:
        print(f"  {image_id:20s}  →  {label}")


if __name__ == "__main__":
    main()

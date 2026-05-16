"""
src/train.py
ASL Sign Language Classifier — Training Script

Designed to run on Kaggle (GPU P100/T4) or locally.
Reads config from environment variables or CLI flags.

Usage (Kaggle notebook cell):
    !python src/train.py \
        --data_dir /kaggle/input/asl-sign-language-dataset \
        --save_dir /kaggle/working/checkpoints \
        --arch efficientnet_b3 \
        --epochs 20 \
        --batch_size 64

Usage (local):
    python src/train.py --data_dir ./data --epochs 5 --batch_size 32
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment

# Local import
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, LabelSmoothingCE


# ─────────────────────── configuration ───────────────────────────
IMG_SIZE = 224
SEED     = 42
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]


def seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────── transforms ─────────────────────────────
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0,
                                translate=(0.1, 0.1),
                                scale=(0.85, 1.15),
                                shear=10),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ─────────────────────── train / eval loops ──────────────────────
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += logits.argmax(1).eq(labels).sum().item()
        total      += imgs.size(0)

        if batch_idx % 100 == 0:
            print(f"    [{batch_idx:4d}/{len(loader)}] "
                  f"loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[-1]:.6f}")

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += logits.argmax(1).eq(labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, 100.0 * correct / total


# ──────────────────────────── main ───────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ASL Sign Language Classifier")
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--arch",        default="efficientnet_b3",
                   choices=["efficientnet_b3", "resnet50", "convnext_tiny"])
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--workers",     type=int,   default=2)
    p.add_argument("--resume",      default=None)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*50}")
    print(f"Device   : {device}")
    print(f"Arch     : {args.arch}")
    print(f"Epochs   : {args.epochs}")
    print(f"Batch    : {args.batch_size}")
    print(f"{'='*50}")

    # ── dataset ──────────────────────────────────────────────────
    train_root = Path(args.data_dir) / "train"
    assert train_root.exists(), f"Train folder not found: {train_root}"

    # ImageFolder reads class names from subdirectory names
    full_ds = datasets.ImageFolder(train_root, transform=get_train_transforms())
    print(f"Total training images : {len(full_ds)}")
    print(f"Classes ({len(full_ds.classes)}): {full_ds.classes}")

    val_size   = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Val split uses val transforms (no augmentation)
    val_ds_clean = datasets.ImageFolder(train_root, transform=get_val_transforms())
    # Use same indices but clean transforms
    val_ds.dataset = val_ds_clean

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0)
    )

    # ── model ─────────────────────────────────────────────────────
    num_classes = len(full_ds.classes)
    model       = build_model(args.arch, num_classes=num_classes).to(device)

    # Differential learning rates: backbone trains 10× slower than head
    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith("classifier") and not n.startswith("fc")]
    head_params     = [p for n, p in model.named_parameters()
                       if n.startswith("classifier") or n.startswith("fc")]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr / 10},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    total_steps = args.epochs * len(train_loader)
    scheduler   = OneCycleLR(
        optimizer,
        max_lr=[args.lr / 10, args.lr],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    criterion = LabelSmoothingCE(num_classes, smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── resume ────────────────────────────────────────────────────
    start_epoch  = 0
    best_val_acc = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} | best val acc: {best_val_acc:.2f}%")

    os.makedirs(args.save_dir, exist_ok=True)

    # Save class mapping alongside checkpoints (needed at inference time)
    mapping_path = os.path.join(args.save_dir, "class_to_idx.json")
    with open(mapping_path, "w") as f:
        json.dump(full_ds.class_to_idx, f, indent=2)
    print(f"Class mapping saved: {mapping_path}")

    # ── training loop ─────────────────────────────────────────────
    history = []
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} " + "─" * 35)
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"  train → loss: {tr_loss:.4f}  acc: {tr_acc:.2f}%")
        print(f"  val   → loss: {vl_loss:.4f}  acc: {vl_acc:.2f}%  [{elapsed:.0f}s]")

        history.append({"epoch": epoch+1, "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "vl_loss": vl_loss, "vl_acc": vl_acc})

        checkpoint = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_to_idx": full_ds.class_to_idx,
            "arch":         args.arch,
        }

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ✓ New best → {best_val_acc:.2f}%  [saved best_model.pth]")

        torch.save(checkpoint, os.path.join(args.save_dir, "latest.pth"))

    # Save training history
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training complete. Best val accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint: {args.save_dir}/best_model.pth")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

import os, json, time, csv, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, models, transforms
from torchvision.transforms import RandAugment
from PIL import Image
from tqdm.auto import tqdm

# ───────────────────────  DATASET PATHS ─────────────────────────────────────

COMPETITION_ROOT = "/kaggle/input/competitions/the-silent-gap-asl-sign-language-challenge"
DATA_DIR         = f"{COMPETITION_ROOT}/competition_data/competition_data"
SAMPLE_SUB_PATH  = f"{COMPETITION_ROOT}/sample_submission.csv"   # correct location
SAVE_DIR         = "/kaggle/working/checkpoints"
OUTPUT_CSV       = "/kaggle/working/submission.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────── HYPERPARAMETERS ─────────────────────────────────────
ARCH        = "efficientnet_b3"   # change to "resnet50" if GPU runs out of memory
EPOCHS      = 25                  # FIX: bumped from 20 → 25 for extra accuracy
BATCH_SIZE  = 64
LR          = 3e-4
VAL_SPLIT   = 0.1
IMG_SIZE    = 224
SEED        = 42
TTA_COUNT   = 4                   # FIX: reduced from 5 → 4 (removed H-flip pass)
WORKERS     = 2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─────────────────────── REPRODUCIBILITY ─────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ─────────────────────── TRANSFORMS ──────────────────────────────────────────
# FIX 2: Removed RandomHorizontalFlip — ASL signs are NOT mirror-symmetric.
#         Flipping changes the meaning of letters (J, Z, and hand-chirality signs).
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    # RandomHorizontalFlip REMOVED — breaks ASL sign meaning
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ─────────────────────── MODEL ────────────────────────────────────────────────
def build_model(num_classes):
    weights = models.EfficientNet_B3_Weights.DEFAULT
    model   = models.efficientnet_b3(weights=weights)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_feat, 512),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model

class LabelSmoothingCE(nn.Module):
    def __init__(self, n, smoothing=0.1):
        super().__init__()
        self.n = n
        self.s = smoothing
    def forward(self, pred, target):
        log_p   = nn.functional.log_softmax(pred, dim=-1)
        smooth  = torch.full_like(log_p, self.s / (self.n - 1))
        smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.s)
        return -(smooth * log_p).sum(dim=-1).mean()

# ─────────────────────── DATASET ─────────────────────────────────────────────
train_root = os.path.join(DATA_DIR, "train")
full_ds    = datasets.ImageFolder(train_root, transform=train_tf)
num_classes = len(full_ds.classes)

print(f"\nClasses ({num_classes}): {full_ds.classes}")
print(f"Total training images : {len(full_ds)}")

val_size   = int(len(full_ds) * VAL_SPLIT)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(
    full_ds, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Val set gets clean transforms — use Subset properly to avoid index mismatch
val_clean = datasets.ImageFolder(train_root, transform=val_tf)
val_ds    = Subset(val_clean, val_ds.indices)   # FIX 3: cleaner than reassigning .dataset

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=WORKERS, pin_memory=True)

# ─────────────────────── TRAINING SETUP ──────────────────────────────────────
model     = build_model(num_classes).to(device)
criterion = LabelSmoothingCE(num_classes, smoothing=0.1)

scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
head_params     = list(model.classifier.parameters())

optimizer = optim.AdamW([
    {"params": backbone_params, "lr": LR / 10},
    {"params": head_params,     "lr": LR},
], weight_decay=1e-4)

total_steps = EPOCHS * len(train_loader)
scheduler   = OneCycleLR(optimizer,
                         max_lr=[LR / 10, LR],
                         total_steps=total_steps,
                         pct_start=0.1,
                         anneal_strategy="cos")

# ─────────────────────── TRAIN / EVAL FUNCTIONS ───────────────────────────────
def train_epoch(model, loader):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, 100.0 * correct / total

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False, desc="  val  "):
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, 100.0 * correct / total

# ─────────────────────── TRAINING LOOP ───────────────────────────────────────
best_val_acc = 0.0
history      = []

print("\n" + "="*55)
print(f"  Training {ARCH} for {EPOCHS} epochs")
print("="*55)

for epoch in range(EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader)
    vl_loss, vl_acc = eval_epoch(model, val_loader)
    elapsed = time.time() - t0

    history.append(dict(epoch=epoch+1, tr_acc=tr_acc, vl_acc=vl_acc))
    print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
          f"train {tr_acc:.2f}%  val {vl_acc:.2f}%  "
          f"[{elapsed:.0f}s]", end="")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save({
            "model":        model.state_dict(),
            "class_to_idx": full_ds.class_to_idx,
            "arch":         ARCH,
            "best_val_acc": best_val_acc,
            "epoch":        epoch,
        }, os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"  ← best ✓", end="")
    print()

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

# ─────────────────────── PLOT TRAINING ───────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    epochs_x = [h["epoch"] for h in history]
    plt.figure(figsize=(9, 4))
    plt.plot(epochs_x, [h["tr_acc"] for h in history], "b-o", label="Train")
    plt.plot(epochs_x, [h["vl_acc"] for h in history], "r-o", label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title(f"{ARCH} — best val {best_val_acc:.2f}%")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("/kaggle/working/training_curve.png", dpi=120)
    plt.show()
except Exception as e:
    print(f"Plot skipped: {e}")

# ─────────────────────── INFERENCE + TTA ─────────────────────────────────────
print("\n" + "="*55)
print(f"  Generating submission (TTA={TTA_COUNT})")
print("="*55)

# FIX 4: weights_only=False suppresses PyTorch 2.x deprecation warning
ckpt         = torch.load(os.path.join(SAVE_DIR, "best_model.pth"),
                           map_location=device, weights_only=False)
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
model.load_state_dict(ckpt["model"])
model.eval()

def _n(): return [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]

# FIX 2 (continued): H-flip TTA pass removed — ASL signs are NOT mirror-symmetric
TTA_TRANSFORMS = [
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), *_n()]),
    transforms.Compose([transforms.Resize((IMG_SIZE+20, IMG_SIZE+20)),
                        transforms.CenterCrop(IMG_SIZE),
                        transforms.RandomRotation((8, 8)), *_n()]),
    transforms.Compose([transforms.Resize((IMG_SIZE+20, IMG_SIZE+20)),
                        transforms.CenterCrop(IMG_SIZE),
                        transforms.RandomRotation((-8, -8)), *_n()]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2), *_n()]),
]

class TestDataset(Dataset):
    def __init__(self, test_dir, transform):
        test_path = Path(test_dir)
        exts  = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        paths = []
        for ext in exts:
            paths.extend(test_path.glob(ext))
        self.paths = sorted(set(paths), key=lambda p: p.name)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {test_dir}")
        self.transform = transform

    def __len__(self):  return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        return self.transform(Image.open(p).convert("RGB")), p.name

# Read sample_submission to determine image_id format (with/without extension)
# SAMPLE_SUB_PATH is set at the top — it lives one level above competition_data/competition_data
use_extension = True
if os.path.exists(SAMPLE_SUB_PATH):
    _ss = pd.read_csv(SAMPLE_SUB_PATH)
    print(f"\nSample submission preview:\n{_ss.head(3).to_string(index=False)}\n")
    first_id = str(_ss["image_id"].iloc[0])
    use_extension = "." in first_id

def make_image_id(fname):
    """Return image_id in the same format as sample_submission.csv."""
    return fname if use_extension else Path(fname).stem

test_dir = os.path.join(DATA_DIR, "test")
accum    = {}   # filename → summed probs

for aug_i, aug in enumerate(TTA_TRANSFORMS[:TTA_COUNT]):
    print(f"  TTA pass {aug_i+1}/{TTA_COUNT} ...")
    loader = DataLoader(TestDataset(test_dir, aug),
                        batch_size=128, shuffle=False,
                        num_workers=WORKERS, pin_memory=True)
    with torch.no_grad():
        for imgs, fnames in tqdm(loader, leave=False):
            probs = torch.softmax(model(imgs.to(device)), dim=-1).cpu()
            for fname, p in zip(fnames, probs):
                key = make_image_id(fname)
                if key not in accum:
                    accum[key] = torch.zeros_like(p)
                accum[key] += p

# Write CSV
results = [(img_id, idx_to_class[probs.argmax().item()])
           for img_id, probs in sorted(accum.items())]

with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "label"])
    w.writerows(results)

# ─────────────────────── FINAL CHECKS ────────────────────────────────────────
sub   = pd.read_csv(OUTPUT_CSV)
VALID = set(full_ds.classes)
bad   = sub[~sub["label"].isin(VALID)]

print(f"\n{'='*55}")
print(f"  Submission saved : {OUTPUT_CSV}")
print(f"  Rows             : {len(sub)}")
print(f"  Invalid labels   : {len(bad)}  {'✓ all good' if len(bad)==0 else '⚠ FIX THESE'}")
print(f"  Best val acc     : {best_val_acc:.2f}%")
print(f"  image_id format  : {'with extension' if use_extension else 'no extension'}")
print(f"{'='*55}")
print("\nLabel distribution:")
print(sub["label"].value_counts().sort_index().to_string())
print("\nFirst 10 predictions:")
print(sub.head(10).to_string(index=False))
print(f"\n✓ Download submission.csv from Output panel on the right →")

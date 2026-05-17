<div align="center">

# 🤟 ASL Sign Language Classifier

### XR-Hackathon 3.0 — FCCU × The Silent Gap Challenge

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-1.00000-brightgreen?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/t/9afe513d766c484fa6d50d1cf36587e8)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Top%2010-gold?style=for-the-badge&logo=trophy)](https://www.kaggle.com/t/9afe513d766c484fa6d50d1cf36587e8)
[![Model](https://img.shields.io/badge/Model-EfficientNet--B3-blue?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12-yellow?style=for-the-badge&logo=python)](https://www.python.org/)

*Classifying all 29 American Sign Language hand gestures with perfect accuracy.*

</div>

---

## 🏆 Result

| Metric | Value |
|--------|-------|
| **Competition Score** | **1.00000 (100%)** |
| **Validation Accuracy** | **100.00%** |
| **Architecture** | EfficientNet-B3 (pretrained ImageNet) |
| **Training Platform** | Kaggle — Tesla T4 GPU |
| **Classes** | 29 ASL signs (A–Z + `del`, `space`, `nothing`) |

---

## 🏅 Competition

**XR-Hackathon 3.0** hosted by **Forman Christian College University (FCCU)**

> **The Silent Gap — ASL Sign Language Classification Challenge**
>
> Build a model to classify 87,000 images across 29 American Sign Language hand signs.
> - `train/` — ~70,000 images, organized into 29 subfolders (one per sign)
> - `test/`  — ~17,000 images in a flat folder (no labels)
> - Predict the sign label for every test image and submit a CSV

🔗 **Competition Link:** https://www.kaggle.com/t/9afe513d766c484fa6d50d1cf36587e8

---

## 🧠 Model Architecture

```
EfficientNet-B3  (pretrained on ImageNet)
        │
        ▼
   Backbone (frozen LR = LR/10)
        │
        ▼
   Custom Classifier Head:
   ┌─────────────────────────┐
   │  Dropout(0.4)           │
   │  Linear(1536 → 512)     │
   │  SiLU Activation        │
   │  Dropout(0.2)           │
   │  Linear(512 → 29)       │
   └─────────────────────────┘
```

### Why EfficientNet-B3?
- Excellent accuracy-to-compute ratio — ideal for Kaggle's T4 GPU
- Strong pretrained features transfer well to hand gesture recognition
- Smaller than ResNet-50/101 but consistently outperforms them on image classification tasks

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 25 |
| Batch Size | 64 |
| Image Size | 224 × 224 |
| Optimizer | AdamW (differential LR) |
| Backbone LR | `3e-5` |
| Head LR | `3e-4` |
| Scheduler | OneCycleLR (cosine annealing) |
| Loss | Label Smoothing Cross-Entropy (ε=0.1) |
| AMP | ✅ Mixed Precision (float16) |
| Gradient Clipping | 1.0 |
| Val Split | 10% |

### Data Augmentation (Train)
- Random Crop (256 → 224)
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation, hue)
- Random Affine (translate, scale, shear)
- RandAugment (2 ops, magnitude 9)
- Random Erasing (p=0.2)

> ⚠️ **No Horizontal Flip** — ASL hand signs are NOT mirror-symmetric. Flipping changes the meaning of directional signs (J, Z) and breaks hand chirality.

### Test-Time Augmentation (TTA)
4 inference passes averaged for robustness:
1. Clean resize
2. +8° rotation
3. −8° rotation
4. Color jitter

---

## 📈 Training Progress

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 42.99% | 95.63% |
| 2 | 93.52% | 99.67% |
| 5 | 99.10% | **100.00%** ← best saved |
| 10 | 99.53% | 100.00% |
| 25 | 99.76% | 100.00% |

The model converged to 100% validation accuracy by **Epoch 5** and held it for all remaining epochs.

---

## 🗂️ Project Structure

```
asl-sign-classifier/
│
├── notebooks/
│   └── kaggle_train.py        # Complete training + inference script (Kaggle-ready)
│
├── submissions/               # Submission CSVs stored here (not tracked in git)
│   └── .gitkeep
│
├── Hackathon_Guide.pdf        # Official competition guidelines
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### On Kaggle (Recommended)
1. Create a new Kaggle notebook
2. Add **The Silent Gap — Sign Language Challenge** as competition data input
3. Copy the contents of `notebooks/kaggle_train.py` into a single code cell
4. Enable **GPU (T4)** accelerator
5. Run All — training + inference + `submission.csv` generation is fully automated

### Dependencies
All dependencies are pre-installed in Kaggle's base image. For local use:
```bash
pip install -r requirements.txt
```

```
torch>=2.1.0
torchvision>=0.16.0
tqdm>=4.65.0
Pillow>=9.5.0
matplotlib>=3.7.0
pandas>=2.0.0
```

---

## 👥 Team — AI-Took-My-Job

| Name | LinkedIn |
|------|----------|
| **Muhammad Ahtasham Ul Haq** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/mr-ahtasham-ul-haq/) |
| **Hasnain Ali Asghar** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/hasnain-ali-asghar-2123222a6/) |

---

## 📄 License

This project is open-sourced for educational purposes.  
Competition: XR-Hackathon 3.0 — FCCU × The Silent Gap, May 2026.

---

<div align="center">

Made with 🤟 for XR-Hackathon 3.0 · FCCU · May 2026

</div>
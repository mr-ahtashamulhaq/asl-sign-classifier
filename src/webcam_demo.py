"""
Phase 2: Live Webcam ASL Classifier
Forman CS Club · AI Hackathon 2026

Features:
  - MediaPipe hand detection → crops hand region before inference
  - Confidence threshold + hold timer → commits letters to a word buffer
  - Displays: live camera | detected sign | current word | sentence history
  - del / space / nothing classes are handled automatically

Requirements:
    pip install opencv-python mediapipe torch torchvision

Usage:
    python webcam_demo.py --ckpt checkpoints/best_model.pth

Controls:
    Q        → quit
    SPACE    → manual space (skip hold timer)
    BKSP     → delete last character
    C        → clear sentence
"""

import argparse
import collections
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import build_model


# ─────────────────────────── config ──────────────────────────────
IMG_SIZE         = 224
CONFIDENCE_THRESH = 0.88    # must be this confident to count
HOLD_SECONDS      = 1.5     # must hold same sign for this long to commit
STABILITY_FRAMES  = 8       # last N predictions must agree (smoothing)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ─────────────────────────── helpers ─────────────────────────────
def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    arch         = ckpt.get("arch", "efficientnet_b3")
    model        = build_model(arch, num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded: {arch}  |  classes: {len(class_to_idx)}  |  "
          f"best val: {ckpt.get('best_val_acc', 'N/A')}")
    return model, idx_to_class


@torch.no_grad()
def predict(model, img_rgb: np.ndarray, device: torch.device, idx_to_class: dict):
    """Returns (label, confidence) for a single hand crop."""
    tensor = TRANSFORM(img_rgb).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=-1)[0]
    conf, idx = probs.max(0)
    return idx_to_class[idx.item()], conf.item()


def crop_hand(frame_rgb, hand_landmarks, padding: float = 0.15):
    """
    Crop a padded bounding box around MediaPipe hand landmarks.
    Returns the cropped RGB region, or None if out of bounds.
    """
    h, w = frame_rgb.shape[:2]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min = max(0, int((min(xs) - padding) * w))
    x_max = min(w, int((max(xs) + padding) * w))
    y_min = max(0, int((min(ys) - padding) * h))
    y_max = min(h, int((max(ys) + padding) * h))

    if x_max <= x_min or y_max <= y_min:
        return None, None, None, None

    crop = frame_rgb[y_min:y_max, x_min:x_max]
    return crop, x_min, y_min, x_max, y_max


# ─────────────────────────── UI drawing ──────────────────────────
FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_SIMPLEX
GREEN  = (80, 200, 120)
BLUE   = (100, 180, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GRAY   = (160, 160, 160)
AMBER  = (50, 180, 240)


def draw_overlay(frame, label, conf, committed_letter, sentence,
                 hold_progress: float, bbox=None):
    """Draw all UI elements on frame (in-place)."""
    h, w = frame.shape[:2]

    # ── bounding box ──
    if bbox:
        x1, y1, x2, y2 = bbox
        color = GREEN if conf >= CONFIDENCE_THRESH else GRAY
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ── top bar: current prediction ──
    bar_h = 70
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)

    if label not in ("nothing", None):
        sign_text = label if label in ("space", "del") else label
        cv2.putText(frame, sign_text, (20, 50),
                    FONT, 1.6, WHITE, 2, cv2.LINE_AA)
        conf_txt = f"{conf*100:.0f}%"
        conf_col = GREEN if conf >= CONFIDENCE_THRESH else GRAY
        cv2.putText(frame, conf_txt, (130, 50),
                    FONT, 1.0, conf_col, 1, cv2.LINE_AA)

        # Hold timer bar
        if hold_progress > 0:
            bar_w = int((w - 40) * hold_progress)
            cv2.rectangle(frame, (20, 58), (20 + bar_w, 65), GREEN, -1)
            cv2.rectangle(frame, (20, 58), (w - 20, 65), GRAY, 1)

    # ── bottom bar: sentence ──
    bottom_h = 80
    cv2.rectangle(frame, (0, h - bottom_h), (w, h), (20, 20, 20), -1)

    # Word being built
    display_sentence = sentence[-60:] if len(sentence) > 60 else sentence
    cursor = display_sentence + "▌"
    cv2.putText(frame, cursor, (16, h - bottom_h + 32),
                FONT, 0.75, WHITE, 1, cv2.LINE_AA)

    # Controls hint
    hint = "Q:quit  SPACE:space  BKSP:delete  C:clear"
    cv2.putText(frame, hint, (16, h - 14),
                FONT, 0.38, GRAY, 1, cv2.LINE_AA)

    # Committed flash
    if committed_letter:
        flash_txt = f"→ '{committed_letter}'"
        cv2.putText(frame, flash_txt, (w - 140, 50),
                    FONT, 1.0, AMBER, 2, cv2.LINE_AA)


# ─────────────────────────── main loop ───────────────────────────
def run(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class = load_model(args.ckpt, device)

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sentence        = ""
    recent_preds    = collections.deque(maxlen=STABILITY_FRAMES)
    hold_start      = None
    last_stable     = None
    committed_flash = ""
    flash_until     = 0.0

    print("Webcam started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        label      = "nothing"
        conf       = 0.0
        bbox       = None
        hold_prog  = 0.0

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            crop_result = crop_hand(rgb, landmarks)
            if crop_result[0] is not None:
                crop, x1, y1, x2, y2 = crop_result
                bbox  = (x1, y1, x2, y2)
                label, conf = predict(model, crop, device, idx_to_class)

        # ── stability filter ──
        recent_preds.append(label if conf >= CONFIDENCE_THRESH else "nothing")
        stable = (len(set(recent_preds)) == 1 and
                  list(recent_preds)[0] not in ("nothing",))
        stable_label = list(recent_preds)[0] if stable else None

        # ── hold timer ──
        now = time.time()
        if stable_label and stable_label == last_stable:
            if hold_start is None:
                hold_start = now
            elapsed  = now - hold_start
            hold_prog = min(elapsed / HOLD_SECONDS, 1.0)

            if elapsed >= HOLD_SECONDS:
                # Commit!
                if stable_label == "del":
                    sentence = sentence[:-1]
                    committed_flash = "⌫"
                elif stable_label == "space":
                    sentence += " "
                    committed_flash = "SPACE"
                else:
                    sentence += stable_label
                    committed_flash = stable_label
                flash_until = now + 0.8
                hold_start  = None
                last_stable = None
                recent_preds.clear()
        else:
            hold_start  = now if stable_label else None
            last_stable = stable_label
            hold_prog   = 0.0

        if now > flash_until:
            committed_flash = ""

        # ── draw ──
        draw_overlay(
            frame, label, conf,
            committed_flash if now < flash_until else "",
            sentence, hold_prog, bbox
        )
        cv2.imshow("ASL Sign Language Classifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = ""
        elif key == 8 or key == 127:   # backspace
            sentence = sentence[:-1]
        elif key == 32:                 # spacebar
            sentence += " "

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if sentence.strip():
        print(f"\nFinal sentence: {sentence.strip()}")


# ─────────────────────────── entry point ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best_model.pth")
    args = parser.parse_args()
    run(args)

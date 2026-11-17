"""
training/diagnose_gemini_frames.py
Runs a per-frame diagnostic on gemini_fake.mp4 using best_model_clean.pth
to check per-frame probabilities and detect any subtle artifact signals.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.video_model import get_model

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = PROJECT_ROOT / "weights" / "best_model_clean.pth"
VIDEO_PATH = PROJECT_ROOT / "test_videos" / "external_ai" / "gemini_fake.mp4"
NUM_FRAMES = 10

print(f"Device: {DEVICE}")
print(f"Video: {VIDEO_PATH}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video missing: {VIDEO_PATH}")

# ==========================================
# TRANSFORMS & MODEL
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = get_model()
state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# ==========================================
# EXTRACT FRAMES
# ==========================================
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError("Failed to open video file.")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")

indices = np.unique(
    np.linspace(0, total_frames - 1, min(NUM_FRAMES, total_frames), dtype=int)
)

frames = []
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    if not ret:
        continue
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(frame_rgb))
    frames.append((int(idx), tensor))

cap.release()

# ==========================================
# INFERENCE & DIAGNOSTICS
# ==========================================
print("\n========== FRAME-BY-FRAME BREAKDOWN ==========")
tensors = torch.stack([f[1] for f in frames]).to(DEVICE)

with torch.no_grad():
    outputs = model(tensors)
    probs = torch.softmax(outputs, dim=1)

fake_probs = probs[:, 0].cpu().numpy()
real_probs = probs[:, 1].cpu().numpy()

for pos, (orig_idx, _) in enumerate(frames):
    f_p = float(fake_probs[pos])
    r_p = float(real_probs[pos])
    pred = "REAL" if r_p >= f_p else "FAKE"
    
    print(
        f"Sample {pos + 1:02d} (Frame Index {orig_idx:04d}): "
        f"REAL = {r_p * 100:6.2f}% | "
        f"FAKE = {f_p * 100:6.2f}% | "
        f"Pred = {pred}"
    )

print("\n========== AGGREGATE SUMMARY ==========")
avg_real = float(np.mean(real_probs)) * 100
avg_fake = float(np.mean(fake_probs)) * 100
print(f"Mean REAL Probability: {avg_real:.2f}%")
print(f"Mean FAKE Probability: {avg_fake:.2f}%")
print(f"Final Classification:  {'REAL' if avg_real >= avg_fake else 'FAKE'}")
print("=======================================\n")
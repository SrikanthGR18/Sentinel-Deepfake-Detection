
import cv2
import torch
import numpy as np

from pathlib import Path
from torchvision import transforms
from PIL import Image

from models.video_model import get_model


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    PROJECT_ROOT
    / "weights"
    / "best_model_clean.pth"
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# LOAD MODEL
# =========================================================

if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT}"
    )

model = get_model()

state_dict = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()


# =========================================================
# IMAGE TRANSFORMATION
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================================================
# FRAME EXTRACTION
# =========================================================

def extract_frames(video_path, num_frames=10):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return []

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total <= 0:
        cap.release()
        return []

    # Never request more frames than the video contains
    sample_count = min(
        num_frames,
        total
    )

    # Uniformly sample across the entire video
    indices = np.linspace(
        0,
        total - 1,
        sample_count,
        dtype=int
    )

    # Avoid duplicate frame indices
    indices = np.unique(indices)

    frames = []

    for idx in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx)
        )

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    return frames


# =========================================================
# VIDEO PREDICTION
# =========================================================

def predict_video(video_path):

    frames = extract_frames(video_path)

    # -----------------------------------------------------
    # No frames extracted
    # -----------------------------------------------------

    if len(frames) == 0:

        return {
            "score": 50.0,
            "status": "Error",
            "reason": "No frames extracted.",
            "real_probability": 0,
            "fake_probability": 0,
            "real_votes": 0,
            "fake_votes": 0,
            "frames": 0
        }

    real_probs = []
    fake_probs = []
    frame_predictions = []

    # =====================================================
    # FRAME-LEVEL INFERENCE
    # =====================================================

    with torch.no_grad():

        for frame in frames:

            img = Image.fromarray(frame)

            img = transform(img)

            img = img.unsqueeze(0).to(DEVICE)

            output = model(img)

            prob = torch.softmax(
                output,
                dim=1
            )[0]

            # Class 0 = FAKE
            # Class 1 = REAL

            fake_prob = float(
                prob[0].cpu()
            )

            real_prob = float(
                prob[1].cpu()
            )

            fake_probs.append(fake_prob)
            real_probs.append(real_prob)

            # -------------------------------------------------
            # Frame-level diagnostic
            # -------------------------------------------------

            print(
                f"Frame {len(real_probs):02d}: "
                f"REAL={real_prob * 100:.2f}% | "
                f"FAKE={fake_prob * 100:.2f}%"
            )

            # -------------------------------------------------
            # Frame prediction
            # -------------------------------------------------

            if real_prob >= fake_prob:

                frame_predictions.append("REAL")

            else:

                frame_predictions.append("FAKE")

    # =====================================================
    # VIDEO-LEVEL AGGREGATION
    # =====================================================

    # Average probability across sampled frames

    real_probability = float(
        np.mean(real_probs)
    )

    fake_probability = float(
        np.mean(fake_probs)
    )

    # Count frame-level votes

    real_votes = frame_predictions.count("REAL")

    fake_votes = frame_predictions.count("FAKE")

    # Overall model confidence
    # This is the maximum averaged softmax score,
    # not a calibrated probability of authenticity.

    confidence = max(
        real_probability,
        fake_probability
    )

    # =====================================================
    # MODEL OUTPUT
    # =====================================================

    print("\n========== AI MODEL ==========")

    print(
        f"Frames Analysed : {len(frame_predictions)}"
    )

    print(
        f"REAL Votes      : {real_votes}"
    )

    print(
        f"FAKE Votes      : {fake_votes}"
    )

    print(
        f"REAL Probability: {real_probability * 100:.2f}%"
    )

    print(
        f"FAKE Probability: {fake_probability * 100:.2f}%"
    )

    print(
        f"Model Confidence: {confidence * 100:.2f}%"
    )

    print("==============================")

    # =====================================================
    # FINAL VIDEO CLASSIFICATION
    # =====================================================

    if real_probability >= fake_probability:

        status = "Authentic"

        # Higher score = more REAL

        score = real_probability * 100

    else:

        status = "Deepfake"

        # Lower score = more FAKE
        # Preserves the existing DTFE score convention.

        score = (1 - fake_probability) * 100

    # =====================================================
    # EXPLANATION
    # =====================================================

    reason = (
        f"Analyzed {len(frame_predictions)} sampled frames. "
        f"REAL votes: {real_votes}, "
        f"FAKE votes: {fake_votes}. "
        f"Average confidence: {confidence * 100:.2f}%."
    )

    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {
        "score": round(score, 2),

        "status": status,

        "reason": reason,

        "real_probability": round(
            real_probability * 100, 2
        ),

        "fake_probability": round(
            fake_probability * 100, 2
        ),

        "real_votes": real_votes,

        "fake_votes": fake_votes,

        "frames": len(frame_predictions)
    }

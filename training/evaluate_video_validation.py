
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from models.video_model import get_model


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_ROOT = PROJECT_ROOT / "datasets/CelebDF-v2"
MANIFEST = PROJECT_ROOT / "datasets/sentinel_clean/manifests/val.csv"

CHECKPOINT = PROJECT_ROOT / "weights/best_model_clean.pth"

OUTPUT = PROJECT_ROOT / "results/video_validation_corrected_20261102.csv"

NUM_FRAMES = 10

# Class mapping:
# 0 = FAKE
# 1 = REAL

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# SAFETY CHECKS
# ==========================================

if not MANIFEST.exists():
    raise FileNotFoundError(MANIFEST)

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}\n"
        "Change OUTPUT filename before running again."
    )


# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# LOAD MODEL
# ==========================================

model = get_model()

state = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state)
model.to(DEVICE)
model.eval()


# ==========================================
# READ VALIDATION MANIFEST
# ==========================================

videos = []

with open(
    MANIFEST,
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    required = {"video", "label", "frames_saved"}

    if not required.issubset(reader.fieldnames or []):
        raise ValueError(
            f"Manifest must contain columns: {required}"
        )

    for row in reader:

        label = int(row["label"])

        if label not in (0, 1):
            raise ValueError(f"Unexpected label: {label}")

        video_path = DATASET_ROOT / row["video"]

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        videos.append({
            "path": video_path,
            "relative_path": row["video"],
            "label": label
        })

print("Validation videos:", len(videos))


# ==========================================
# EXTRACT FRAMES
# ==========================================

def extract_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return []

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total <= 0:
        cap.release()
        return []

    indices = np.unique(
        np.linspace(
            0,
            total - 1,
            min(NUM_FRAMES, total),
            dtype=int
        )
    )

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(
            transform(Image.fromarray(frame))
        )

    cap.release()

    return frames


# ==========================================
# VIDEO-LEVEL PREDICTION
# ==========================================

def predict_video(video_path):

    frames = extract_frames(video_path)

    if not frames:
        return None

    batch = torch.stack(frames).to(DEVICE)

    with torch.no_grad():

        outputs = model(batch)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        average = probabilities.mean(dim=0)

    fake_probability = average[0].item()
    real_probability = average[1].item()

    return {
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "frames": len(frames)
    }


# ==========================================
# EVALUATE ALL VALIDATION VIDEOS
# ==========================================

actual = []
fake_probabilities = []
results = []
failed = []

for index, item in enumerate(videos, start=1):

    result = predict_video(item["path"])

    if result is None:
        failed.append(item["relative_path"])
        continue

    actual.append(item["label"])

    fake_probabilities.append(
        result["fake_probability"]
    )

    # CORRECT CLASS MAPPING:
    # Fake probability >= 0.5 -> FAKE (0)
    # Fake probability < 0.5  -> REAL (1)

    predicted = (
        0 if result["fake_probability"] >= 0.5
        else 1
    )

    results.append({
        "video": item["relative_path"],
        "actual": item["label"],
        "predicted": predicted,
        "correct": int(predicted == item["label"]),
        "fake_probability": round(
            result["fake_probability"], 6
        ),
        "real_probability": round(
            result["real_probability"], 6
        ),
        "frames": result["frames"]
    })

    if index % 50 == 0 or index == len(videos):
        print(f"Processed {index}/{len(videos)}")


# ==========================================
# SAVE PREDICTIONS
# ==========================================

if not actual:
    raise RuntimeError("No validation videos were evaluated.")

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# DEFAULT THRESHOLD METRICS
# ==========================================

actual_array = np.array(actual)
fake_array = np.array(fake_probabilities)

# Correct mapping: fake probability >= 0.5 -> class 0
default_predictions = (
    fake_array < 0.5
).astype(int)

print("\n========== VIDEO-LEVEL VALIDATION ==========")

print("Evaluated:", len(actual))
print("Failed:", len(failed))

print(
    "Accuracy:",
    round(
        accuracy_score(actual_array, default_predictions),
        4
    )
)

print(
    "Balanced accuracy:",
    round(
        balanced_accuracy_score(
            actual_array,
            default_predictions
        ),
        4
    )
)

print(
    "Macro F1:",
    round(
        f1_score(
            actual_array,
            default_predictions,
            average="macro"
        ),
        4
    )
)

print("\nConfusion matrix (rows=actual, columns=predicted):")

print(
    confusion_matrix(
        actual_array,
        default_predictions,
        labels=[0, 1]
    )
)

print("\nClassification report:")

print(
    classification_report(
        actual_array,
        default_predictions,
        labels=[0, 1],
        target_names=["FAKE", "REAL"],
        zero_division=0
    )
)


# ==========================================
# THRESHOLD ANALYSIS
# VALIDATION DATA ONLY
# ==========================================

threshold_results = []

for threshold in np.arange(0.05, 0.951, 0.01):

    # Correct mapping:
    # fake probability >= threshold -> FAKE (0)
    # otherwise -> REAL (1)

    predictions = (
        fake_array < threshold
    ).astype(int)

    threshold_results.append({
        "threshold": float(threshold),

        "accuracy": accuracy_score(
            actual_array,
            predictions
        ),

        "balanced_accuracy": balanced_accuracy_score(
            actual_array,
            predictions
        ),

        "macro_f1": f1_score(
            actual_array,
            predictions,
            average="macro",
            zero_division=0
        )
    })


# ==========================================
# BEST THRESHOLDS
# ==========================================

best_balanced = max(
    threshold_results,
    key=lambda x: x["balanced_accuracy"]
)

best_macro = max(
    threshold_results,
    key=lambda x: x["macro_f1"]
)

print("\n========== VALIDATION THRESHOLD ANALYSIS ==========")

print("\nBest balanced-accuracy threshold:")

print(
    "Threshold:",
    round(best_balanced["threshold"], 2)
)

print(
    "Balanced accuracy:",
    round(best_balanced["balanced_accuracy"], 4)
)

print(
    "Macro F1:",
    round(best_balanced["macro_f1"], 4)
)

print("\nBest macro-F1 threshold:")

print(
    "Threshold:",
    round(best_macro["threshold"], 2)
)

print(
    "Macro F1:",
    round(best_macro["macro_f1"], 4)
)

print(
    "Balanced accuracy:",
    round(best_macro["balanced_accuracy"], 4)
)


# ==========================================
# FAILED VIDEOS
# ==========================================

print("\nFailed videos:")

for video in failed:
    print(video)


# ==========================================
# COMPLETION
# ==========================================

print("\nResults saved:", OUTPUT)

print(
    "\nNOTE: Thresholds were selected using validation data. "
    "Their performance on this same validation set is not "
    "independent test performance. Keep the official test set "
    "separate."
)

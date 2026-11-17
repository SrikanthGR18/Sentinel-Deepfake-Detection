
import sys
import csv
from pathlib import Path

# Add project root to Python import path
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

ROOT = Path("datasets/CelebDF-v2")
TEST_LIST = ROOT / "List_of_testing_videos.txt"

CHECKPOINT = Path("weights/best_model_clean.pth")

OUTPUT = Path("results/clean_official_test.csv")

NUM_FRAMES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


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

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

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
# READ OFFICIAL TEST LIST
# ==========================================

videos = []

with open(TEST_LIST, "r", encoding="utf-8") as file:

    for line in file:

        parts = line.strip().split(maxsplit=1)

        if len(parts) != 2:
            continue

        label = int(parts[0])
        video_path = ROOT / parts[1]

        if label not in (0, 1):
            raise ValueError(
                f"Unexpected label: {label}"
            )

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        videos.append((video_path, label))


print("Official test videos:", len(videos))


# ==========================================
# EXTRACT FRAMES
# ==========================================

def extract_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
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

    predicted = (
        1 if real_probability >= fake_probability
        else 0
    )

    return {
        "predicted": predicted,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "frames": len(frames)
    }


# ==========================================
# EVALUATION
# ==========================================

actual = []
predicted = []
results = []
failed = []

for index, (video_path, true_label) in enumerate(
    videos, start=1
):

    result = predict_video(video_path)

    if result is None:
        failed.append(str(video_path))
        continue

    actual.append(true_label)
    predicted.append(result["predicted"])

    results.append({
        "video": video_path.relative_to(ROOT).as_posix(),
        "actual": true_label,
        "predicted": result["predicted"],
        "correct": int(
            true_label == result["predicted"]
        ),
        "fake_probability": round(
            result["fake_probability"], 6
        ),
        "real_probability": round(
            result["real_probability"], 6
        ),
        "frames": result["frames"]
    })

    print(
        f"[{index}/{len(videos)}] "
        f"{video_path.name} | "
        f"Actual={true_label} | "
        f"Predicted={result['predicted']}"
    )


# ==========================================
# SAVE RESULTS
# ==========================================

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
        if results else [
            "video", "actual", "predicted",
            "correct", "fake_probability",
            "real_probability", "frames"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# METRICS
# ==========================================

if not actual:
    raise RuntimeError("No videos were evaluated.")

print("\n========== OFFICIAL TEST RESULTS ==========")

print("Evaluated:", len(actual))
print("Failed:", len(failed))

print(
    "Accuracy:",
    round(accuracy_score(actual, predicted), 4)
)

print(
    "Balanced accuracy:",
    round(
        balanced_accuracy_score(actual, predicted),
        4
    )
)

print(
    "Macro F1:",
    round(
        f1_score(actual, predicted, average="macro"),
        4
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(actual, predicted))

print("\nClassification report:")
print(
    classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["FAKE", "REAL"],
        zero_division=0
    )
)

print("\nFailed videos:")
for video in failed:
    print(video)

print("\nResults saved:", OUTPUT)

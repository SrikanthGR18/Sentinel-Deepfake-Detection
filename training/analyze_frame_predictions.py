
import sys
import csv
from pathlib import Path

# Add project root
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

ROOT = PROJECT_ROOT / "datasets/CelebDF-v2"

CHECKPOINT = PROJECT_ROOT / "weights/best_model_clean.pth"

OUTPUT = PROJECT_ROOT / "results/frame_diagnostics_20261102.csv"

NUM_FRAMES = 10

VIDEOS = {
    "correct_id2_0008": ROOT / "Celeb-real/id2_0008.mp4",
    "correct_id3_0001": ROOT / "Celeb-real/id3_0001.mp4",
    "false_positive_id1_0007": ROOT / "Celeb-real/id1_0007.mp4",
    "false_positive_id6_0005": ROOT / "Celeb-real/id6_0005.mp4"
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# SAFETY CHECKS
# ==========================================

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}\n"
        "Change OUTPUT filename before running again."
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

for name, path in VIDEOS.items():
    if not path.exists():
        raise FileNotFoundError(path)


# ==========================================
# TRANSFORM
# Same as official evaluation
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
# Same checkpoint and architecture
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
# FRAME EXTRACTION
# Same sampling as official evaluation
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

        tensor = transform(
            Image.fromarray(frame)
        )

        frames.append((int(index), tensor))

    cap.release()

    return frames


# ==========================================
# INDIVIDUAL FRAME PREDICTIONS
# ==========================================

all_results = []

for video_name, video_path in VIDEOS.items():

    print(f"\nAnalyzing: {video_name}")

    frames = extract_frames(video_path)

    if not frames:
        print("ERROR: Could not extract frames.")
        continue

    tensors = torch.stack(
        [frame[1] for frame in frames]
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model(tensors)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    fake_probs = probabilities[:, 0].cpu().numpy()
    real_probs = probabilities[:, 1].cpu().numpy()

    for position, (frame_index, _) in enumerate(frames):

        fake_probability = float(fake_probs[position])
        real_probability = float(real_probs[position])

        predicted = (
            1 if real_probability >= fake_probability
            else 0
        )

        result = {
            "video": video_name,
            "frame_number": position + 1,
            "source_frame_index": frame_index,
            "fake_probability": round(fake_probability, 6),
            "real_probability": round(real_probability, 6),
            "predicted": predicted
        }

        all_results.append(result)

        print(
            f"Frame {position + 1:02d} | "
            f"Fake={fake_probability:.4f} | "
            f"Real={real_probability:.4f} | "
            f"Prediction={'REAL' if predicted == 1 else 'FAKE'}"
        )


# ==========================================
# SAVE FRAME-LEVEL RESULTS
# ==========================================

if not all_results:
    raise RuntimeError("No frame predictions were generated.")

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
        fieldnames=list(all_results[0].keys())
    )

    writer.writeheader()
    writer.writerows(all_results)


# ==========================================
# VIDEO-LEVEL SUMMARY
# ==========================================

print("\n========== VIDEO-LEVEL SUMMARY ==========")

for video_name in VIDEOS:

    rows = [
        row for row in all_results
        if row["video"] == video_name
    ]

    if not rows:
        continue

    avg_fake = np.mean([
        row["fake_probability"] for row in rows
    ])

    avg_real = np.mean([
        row["real_probability"] for row in rows
    ])

    fake_count = sum(
        row["predicted"] == 0 for row in rows
    )

    real_count = sum(
        row["predicted"] == 1 for row in rows
    )

    video_prediction = (
        "REAL" if avg_real >= avg_fake else "FAKE"
    )

    print(f"\n{video_name}")
    print("Average fake probability:", round(float(avg_fake), 4))
    print("Average real probability:", round(float(avg_real), 4))
    print("Video prediction:", video_prediction)
    print("Frames predicting FAKE:", fake_count)
    print("Frames predicting REAL:", real_count)

print("\nFrame predictions saved to:")
print(OUTPUT)

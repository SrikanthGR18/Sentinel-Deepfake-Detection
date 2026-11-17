import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
import sys

# Add project root to Python import path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchvision import transforms
from PIL import Image

from models.video_model import get_model


# =========================================================
# CONFIGURATION
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_DIR = ROOT / "test_videos"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "Original": ROOT / "weights" / "best_model.pth",
    "Clean": ROOT / "weights" / "best_model_clean.pth",
    "Generative": ROOT / "weights" / "best_model_generative.pth"
}

NUM_FRAMES = 10


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

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    sample_count = min(num_frames, total)

    indices = np.linspace(
        0,
        total - 1,
        sample_count,
        dtype=int
    )

    indices = np.unique(indices)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(checkpoint_path):

    model = get_model()

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# PREDICT VIDEO USING PRE-EXTRACTED FRAMES
# =========================================================

def predict_video(model, frames):

    if len(frames) == 0:
        return None

    real_probs = []
    fake_probs = []
    frame_predictions = []

    with torch.no_grad():
        for frame in frames:
            img = Image.fromarray(frame)
            img = transform(img)
            img = img.unsqueeze(0).to(DEVICE)

            output = model(img)
            prob = torch.softmax(output, dim=1)[0]

            fake_prob = float(prob[0].cpu())
            real_prob = float(prob[1].cpu())

            fake_probs.append(fake_prob)
            real_probs.append(real_prob)

            if real_prob >= fake_prob:
                frame_predictions.append("REAL")
            else:
                frame_predictions.append("FAKE")

    real_probability = float(np.mean(real_probs))
    fake_probability = float(np.mean(fake_probs))

    prediction = (
        "REAL"
        if real_probability >= fake_probability
        else "FAKE"
    )

    return {
        "prediction": prediction,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "real_votes": frame_predictions.count("REAL"),
        "fake_votes": frame_predictions.count("FAKE"),
        "frames": len(frames)
    }


# =========================================================
# DISCOVER TEST VIDEOS
# =========================================================

def get_test_videos():

    videos = []

    categories = {
        "real": "REAL",
        "fake": "FAKE",
        "external_ai": "FAKE"
    }

    extensions = {".mp4", ".avi", ".mov", ".mkv"}

    for folder, label in categories.items():
        folder_path = TEST_DIR / folder

        if not folder_path.exists():
            print(f"WARNING: Missing folder: {folder_path}")
            continue

        for video_path in sorted(folder_path.rglob("*")):
            if video_path.suffix.lower() in extensions:
                videos.append({
                    "path": video_path,
                    "name": video_path.name,
                    "category": folder,
                    "actual": label
                })

    return videos


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():

    print("\n======================================")
    print(" SENTINEL: MODEL COMPARISON EVALUATION")
    print("======================================")

    print(f"Device: {DEVICE}")

    videos = get_test_videos()

    print(f"\nVideos discovered: {len(videos)}")

    if len(videos) == 0:
        print("ERROR: No test videos found.")
        return

    # -----------------------------------------------------
    # Extract frames once for every video
    # -----------------------------------------------------

    video_data = []

    print("\nExtracting frames...")

    for video in videos:
        frames = extract_frames(
            video["path"],
            NUM_FRAMES
        )

        if len(frames) == 0:
            print(f"ERROR: Could not read {video['name']}")
            return

        video_data.append({
            **video,
            "frames_data": frames
        })

        print(
            f"{video['name']}: "
            f"{len(frames)} frames extracted"
        )

    # -----------------------------------------------------
    # Evaluate all checkpoints
    # -----------------------------------------------------

    all_results = []

    for model_name, checkpoint_path in MODELS.items():

        print("\n--------------------------------------")
        print(f"Evaluating: {model_name} Model")
        print("--------------------------------------")

        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return

        model = load_model(checkpoint_path)

        for video in video_data:
            result = predict_video(
                model,
                video["frames_data"]
            )

            actual = video["actual"]
            predicted = result["prediction"]
            correct = predicted == actual

            row = {
                "video": video["name"],
                "category": video["category"],
                "actual": actual,
                "model": model_name,
                "predicted": predicted,
                "correct": correct,
                "real_probability": result["real_probability"],
                "fake_probability": result["fake_probability"],
                "real_votes": result["real_votes"],
                "fake_votes": result["fake_votes"],
                "frames": result["frames"]
            }

            all_results.append(row)

            print(
                f"\n{video['name']}"
                f"\nActual: {actual}"
                f"\nPredicted: {predicted}"
                f"\nREAL: {result['real_probability'] * 100:.2f}%"
                f"\nFAKE: {result['fake_probability'] * 100:.2f}%"
                f"\nResult: {'CORRECT' if correct else 'WRONG'}"
            )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------

    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "11_video_model_comparison.csv"
    df.to_csv(csv_path, index=False)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n\n======================================")
    print(" FINAL COMPARISON")
    print("======================================")

    for model_name in MODELS:
        model_df = df[df["model"] == model_name]
        correct = int(model_df["correct"].sum())
        total = len(model_df)
        accuracy = (correct / total) * 100

        print(f"\n{model_name} Model")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2f}%")

        print("\nCategory-wise results:")
        for category in ["real", "fake", "external_ai"]:
            category_df = model_df[model_df["category"] == category]

            if len(category_df) == 0:
                continue

            category_correct = int(category_df["correct"].sum())
            category_total = len(category_df)

            print(
                f"{category}: "
                f"{category_correct}/{category_total}"
            )

    print("\n======================================")
    print(f"Results saved to: {csv_path}")
    print("======================================")


if __name__ == "__main__":
    main()
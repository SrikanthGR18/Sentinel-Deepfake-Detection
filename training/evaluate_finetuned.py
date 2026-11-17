
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.inference_experiment import predict_video


TEST_ROOT = ROOT / "test_videos"
OUTPUT = ROOT / "results" / "evaluation_finetuned.csv"

VIDEO_GROUPS = [
    (TEST_ROOT / "real", "REAL", "FaceForensics++ Original"),
    (TEST_ROOT / "fake", "FAKE", "FaceForensics++ Deepfakes"),
    (TEST_ROOT / "external_ai", "FAKE", "External AI Generated"),
]

EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for folder, actual, category in VIDEO_GROUPS:
        if not folder.exists():
            print("Missing folder:", folder)
            continue

        for video in sorted(folder.iterdir()):
            if not video.is_file() or video.suffix.lower() not in EXTENSIONS:
                continue

            print(f"\nTesting: {video.name} | Actual: {actual}")

            try:
                result = predict_video(str(video))
                status = result.get("status", "Error")

                if status == "Authentic":
                    predicted = "REAL"
                elif status in ("Deepfake", "Fake"):
                    predicted = "FAKE"
                else:
                    predicted = "ERROR"

                row = {
                    "video": video.name,
                    "category": category,
                    "actual": actual,
                    "predicted": predicted,
                    "status": status,
                    "real_probability": result.get("real_probability", 0),
                    "fake_probability": result.get("fake_probability", 0),
                    "correct": predicted == actual,
                }
            except Exception as exc:
                print("ERROR:", exc)
                row = {
                    "video": video.name,
                    "category": category,
                    "actual": actual,
                    "predicted": "ERROR",
                    "status": "Error",
                    "real_probability": 0,
                    "fake_probability": 0,
                    "correct": False,
                }

            print(
                f"Predicted: {row['predicted']} | "
                f"Real: {row['real_probability']}% | "
                f"Fake: {row['fake_probability']}% | "
                f"Correct: {row['correct']}"
            )
            rows.append(row)

    if not rows:
        print("No videos were evaluated.")
        return

    fieldnames = [
        "video", "category", "actual", "predicted", "status",
        "real_probability", "fake_probability", "correct",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    evaluated = [r for r in rows if r["predicted"] != "ERROR"]
    correct = sum(r["correct"] for r in evaluated)
    total = len(evaluated)

    print("\n========== FINE-TUNED MODEL RESULTS ==========")
    print(f"Videos evaluated successfully: {total}/{len(rows)}")
    if total:
        print(f"Accuracy: {100 * correct / total:.1f}% ({correct}/{total})")
    print("Results saved to:", OUTPUT)
    print("This is a small exploratory test, not an independent benchmark.")


if __name__ == "__main__":
    main()

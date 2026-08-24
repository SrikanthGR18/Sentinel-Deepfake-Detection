import sys
from pathlib import Path
import csv

# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.inference import predict_video


# =========================================================
# TEST DIRECTORIES
# =========================================================

TEST_ROOT = PROJECT_ROOT / "test_videos"

REAL_DIR = TEST_ROOT / "real"
FAKE_DIR = TEST_ROOT / "fake"
EXTERNAL_AI_DIR = TEST_ROOT / "external_ai"

RESULTS_FILE = TEST_ROOT / "evaluation_results.csv"


# =========================================================
# HELPER
# =========================================================

def get_videos(folder):
    """
    Return all MP4/MOV/AVI/MKV videos from a folder.
    """

    if not folder.exists():
        return []

    extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv"
    }

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in extensions
        ]
    )


# =========================================================
# PREDICTION
# =========================================================

def test_video(video_path, actual_label, category):

    print("\n----------------------------------------")
    print("Video:", video_path.name)
    print("Category:", category)
    print("Actual:", actual_label)
    print("----------------------------------------")

    try:

        result = predict_video(str(video_path))

        predicted_status = result.get("status", "Error")

        real_probability = result.get(
            "real_probability",
            0
        )

        fake_probability = result.get(
            "fake_probability",
            0
        )

        score = result.get(
            "score",
            0
        )

        # Convert numpy values to normal Python numbers
        try:
            real_probability = float(real_probability)
        except:
            real_probability = 0.0

        try:
            fake_probability = float(fake_probability)
        except:
            fake_probability = 0.0

        try:
            score = float(score)
        except:
            score = 0.0

        # Determine predicted class
        if predicted_status == "Authentic":
            predicted_label = "REAL"

        elif predicted_status in ["Deepfake", "Fake"]:
            predicted_label = "FAKE"

        else:
            predicted_label = "ERROR"

        # Determine correctness
        if predicted_label == actual_label:

            correct = True

        else:

            correct = False

        print("\nPrediction:")
        print("Status:", predicted_status)
        print(
            f"Real Probability: {real_probability:.2f}%"
        )
        print(
            f"Fake Probability: {fake_probability:.2f}%"
        )
        print(
            f"Score: {score:.2f}"
        )
        print(
            "Correct:",
            "YES" if correct else "NO"
        )

        return {
            "video": video_path.name,
            "category": category,
            "actual": actual_label,
            "predicted": predicted_label,
            "status": predicted_status,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "score": score,
            "correct": correct
        }

    except Exception as e:

        print("\nERROR while processing video:")
        print(e)

        return {
            "video": video_path.name,
            "category": category,
            "actual": actual_label,
            "predicted": "ERROR",
            "status": "Error",
            "real_probability": 0,
            "fake_probability": 0,
            "score": 0,
            "correct": False
        }


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():

    print("\n")
    print("=" * 60)
    print("          SENTINEL BASELINE EVALUATION")
    print("=" * 60)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nTest directory:")
    print(TEST_ROOT)

    # -----------------------------------------------------
    # FIND VIDEOS
    # -----------------------------------------------------

    real_videos = get_videos(REAL_DIR)
    fake_videos = get_videos(FAKE_DIR)
    external_ai_videos = get_videos(EXTERNAL_AI_DIR)

    print("\n========== DATASET SUMMARY ==========")

    print(
        "Real videos:",
        len(real_videos)
    )

    print(
        "Fake videos:",
        len(fake_videos)
    )

    print(
        "External AI videos:",
        len(external_ai_videos)
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    results = []

    # -----------------------------------------------------
    # REAL VIDEOS
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("                    REAL VIDEOS")
    print("=" * 60)

    for video in real_videos:

        result = test_video(
            video,
            "REAL",
            "FaceForensics++ Original"
        )

        results.append(result)

    # -----------------------------------------------------
    # FAKE VIDEOS
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("                    FAKE VIDEOS")
    print("=" * 60)

    for video in fake_videos:

        result = test_video(
            video,
            "FAKE",
            "FaceForensics++ Deepfakes"
        )

        results.append(result)

    # -----------------------------------------------------
    # EXTERNAL AI VIDEOS
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("                 EXTERNAL AI")
    print("=" * 60)

    for video in external_ai_videos:

        result = test_video(
            video,
            "FAKE",
            "External AI Generated"
        )

        results.append(result)

    # =====================================================
    # CALCULATE METRICS
    # =====================================================

    valid_results = [
        result
        for result in results
        if result["predicted"] != "ERROR"
    ]

    total = len(valid_results)

    correct = sum(
        result["correct"]
        for result in valid_results
    )

    # -----------------------------------------------------
    # Confusion matrix values
    # -----------------------------------------------------

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in valid_results:

        actual = result["actual"]
        predicted = result["predicted"]

        # FAKE correctly detected as FAKE
        if actual == "FAKE" and predicted == "FAKE":
            true_positive += 1

        # REAL correctly detected as REAL
        elif actual == "REAL" and predicted == "REAL":
            true_negative += 1

        # REAL incorrectly detected as FAKE
        elif actual == "REAL" and predicted == "FAKE":
            false_positive += 1

        # FAKE incorrectly detected as REAL
        elif actual == "FAKE" and predicted == "REAL":
            false_negative += 1

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0

    # -----------------------------------------------------
    # Precision
    # -----------------------------------------------------

    if true_positive + false_positive > 0:

        precision = (
            true_positive /
            (true_positive + false_positive)
        )

    else:

        precision = 0

    # -----------------------------------------------------
    # Recall
    # -----------------------------------------------------

    if true_positive + false_negative > 0:

        recall = (
            true_positive /
            (true_positive + false_negative)
        )

    else:

        recall = 0

    # -----------------------------------------------------
    # F1 Score
    # -----------------------------------------------------

    if precision + recall > 0:

        f1 = (
            2 *
            precision *
            recall /
            (precision + recall)
        )

    else:

        f1 = 0

    # =====================================================
    # PRINT FINAL RESULTS
    # =====================================================

    print("\n")
    print("=" * 60)
    print("              SENTINEL RESULTS")
    print("=" * 60)

    print("\nTotal videos tested:", total)
    print("Correct predictions:", correct)

    print(
        f"\nAccuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )

    print("\n========== CONFUSION MATRIX ==========")

    print(
        f"True Positive  (FAKE → FAKE): {true_positive}"
    )

    print(
        f"True Negative  (REAL → REAL): {true_negative}"
    )

    print(
        f"False Positive (REAL → FAKE): {false_positive}"
    )

    print(
        f"False Negative (FAKE → REAL): {false_negative}"
    )

    # =====================================================
    # SAVE CSV
    # =====================================================

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "video",
                "category",
                "actual",
                "predicted",
                "status",
                "real_probability",
                "fake_probability",
                "score",
                "correct"
            ]
        )

        writer.writeheader()

        writer.writerows(results)

    print("\nEvaluation CSV saved to:")

    print(RESULTS_FILE)

    print("\n")
    print("=" * 60)
    print("              EVALUATION COMPLETE")
    print("=" * 60)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
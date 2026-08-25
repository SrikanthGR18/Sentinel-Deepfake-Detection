import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from analysis.sentinel_pipeline import analyze_sentinel


# =========================================================
# TEST VIDEO
# =========================================================

VIDEO_PATH = (
    PROJECT_ROOT
    / "test_videos"
    / "external_ai"
    / "gemini_fake.mp4"
)


# =========================================================
# TEST
# =========================================================

print("\n================================================")
print("       SENTINEL FULL PIPELINE TEST")
print("================================================")

print("\nProject root:")
print(PROJECT_ROOT)

print("\nVideo:")
print(VIDEO_PATH)

print("\nFile exists:")
print(VIDEO_PATH.exists())


if not VIDEO_PATH.exists():

    print("\nERROR: Test video does not exist.")
    sys.exit(1)


try:

    result = analyze_sentinel(
        str(VIDEO_PATH)
    )

    print("\n================================================")
    print("             FINAL RESULT")
    print("================================================")

    print(
        f"Video score       : "
        f"{result['video_score']:.2f}"
    )

    print(
        f"Temporal score    : "
        f"{result['temporal_score']}"
    )

    print(
        f"MFCC score        : "
        f"{result['mfcc_score']}"
    )

    print(
        f"Acoustic score    : "
        f"{result['acoustic_pattern_score']}"
    )

    print(
        f"Audio score       : "
        f"{result['audio_score']}"
    )

    print(
        f"Final DTFE score  : "
        f"{result['final_score']:.2f}"
    )

    print(
        f"Classification     : "
        f"{result['classification']}"
    )

    print(
        f"Confidence         : "
        f"{result['fusion_confidence']}"
    )

    print("\nSummary:")
    print(result["summary"])

    print("\n================================================")
    print("          PIPELINE TEST COMPLETE")
    print("================================================")


except Exception as e:

    print("\n================================================")
    print("              PIPELINE ERROR")
    print("================================================")

    print(type(e).__name__)
    print(e)

    raise
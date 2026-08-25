from pathlib import Path
import os
import subprocess
import shutil

from ammp import preprocess_media
from svim import analyze_audio
from temporal import analyze_temporal
from fusion import (
    FusionClassification,
    FusionInput,
    fuse_trust_scores,
)
from models.inference import predict_video


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline"


# =========================================================
# FFMPEG
# =========================================================

def get_ffmpeg_path():
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        return ffmpeg

    win_path = r"C:\ffmpeg\bin\ffmpeg.exe"

    if os.path.exists(win_path):
        return win_path

    return None


# =========================================================
# AUDIO EXTRACTION
# =========================================================

def extract_audio(video_path, audio_path):
    """
    Extract audio from a video using FFmpeg.

    This follows the same logic used by app.py.
    """

    ffmpeg = get_ffmpeg_path()

    if not ffmpeg:
        print("FFmpeg not found.")
        return False

    try:

        command = [
            ffmpeg,
            "-i",
            video_path,
            "-q:a",
            "0",
            "-map",
            "a?",
            "-y",
            audio_path,
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        return os.path.exists(audio_path)

    except Exception as e:

        print("FFmpeg Error:", e)

        return False


# =========================================================
# SENTINEL PIPELINE
# =========================================================

def analyze_sentinel(video_path):
    """
    Run the complete Sentinel analysis pipeline.

    Pipeline:

        Input Video
             ↓
            AMMP
             ↓
        ┌────┼─────────────┐
        ↓    ↓             ↓
      Video Temporal     Audio
      Model Analysis     SVIM
        │      │           │
        └──────┴───────────┘
                 ↓
             DTFE Fusion
                 ↓
          Final Classification
    """

    video_path = Path(video_path).resolve()

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    uid = video_path.stem

    print("\n================================================")
    print("           SENTINEL FULL PIPELINE")
    print("================================================")

    print(f"Video: {video_path}")

    # =====================================================
    # AMMP
    # =====================================================

    print("\n========== AMMP ==========")

    ammp_result = preprocess_media(
        media_path=str(video_path),
        output_dir=str(OUTPUT_DIR),
        uid=uid,
        extract_audio_fn=extract_audio,
    )

    # =====================================================
    # VIDEO / SPATIAL ANALYSIS
    # =====================================================

    print("\n========== VIDEO MODEL ==========")

    prediction = predict_video(
        str(video_path)
    )

    video_score = float(
        prediction["score"]
    )

    print(
        f"Video status: {prediction['status']}"
    )

    print(
        f"Real probability: "
        f"{prediction['real_probability']:.2f}%"
    )

    print(
        f"Fake probability: "
        f"{prediction['fake_probability']:.2f}%"
    )

    # =====================================================
    # TEMPORAL ANALYSIS
    # =====================================================

    print("\n========== TEMPORAL ==========")

    temporal_result = analyze_temporal(
        video_path=str(video_path),
        frame_paths=(
            ammp_result.frame_paths
            if ammp_result.frame_paths
            else None
        ),
    )

    temporal_score = None

    if temporal_result is not None:

        temporal_score = float(
            temporal_result.temporal_score
        )

        print(
            f"Temporal score: "
            f"{temporal_score:.2f}"
        )

        print(
            f"Temporal status: "
            f"{temporal_result.status}"
        )

    # =====================================================
    # SVIM AUDIO ANALYSIS
    # =====================================================

    print("\n========== SVIM AUDIO ==========")

    svim_result = None

    audio_path = (
        ammp_result.preprocessed_audio_path
    )

    if audio_path and Path(audio_path).exists():

        svim_result = analyze_audio(
            audio_path
        )

        print(
            f"MFCC score: "
            f"{svim_result.mfcc_score:.2f}"
        )

        print(
            f"Acoustic pattern score: "
            f"{svim_result.acoustic_pattern_score:.2f}"
        )

        print(
            f"Audio score: "
            f"{svim_result.audio_score:.2f}"
        )

    else:

        print(
            "No usable audio available."
        )

    # =====================================================
    # DTFE TRUST SCORE FUSION
    # =====================================================

    print("\n========== DTFE FUSION ==========")

    fusion_result = fuse_trust_scores(
        FusionInput(

            spatial_score=round(
                video_score,
                1
            ),

            temporal_score=(
                temporal_result.temporal_score
                if temporal_result is not None
                else None
            ),

            mfcc_score=(
                svim_result.mfcc_score
                if svim_result is not None
                else None
            ),

            acoustic_pattern_score=(
                svim_result.acoustic_pattern_score
                if svim_result is not None
                else None
            ),

            audio_only=False,

            has_audio=(
                svim_result is not None
            ),

            has_video=True,
        )
    )

    classification = FusionClassification(
        fusion_result.fusion_classification
    )

    final_score = float(
        fusion_result.fusion_final_score
    )

    print(
        f"Final score: {final_score:.2f}"
    )

    print(
        f"Classification: "
        f"{classification.value}"
    )

    print(
        f"Confidence: "
        f"{fusion_result.fusion_confidence_level.value}"
    )

    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "video": str(video_path),

        # -----------------------------
        # Video
        # -----------------------------

        "video_score": video_score,

        "video_status": prediction[
            "status"
        ],

        "video_reason": prediction[
            "reason"
        ],

        "video_real_probability": float(
            prediction[
                "real_probability"
            ]
        ),

        "video_fake_probability": float(
            prediction[
                "fake_probability"
            ]
        ),

        "video_real_votes": prediction[
            "real_votes"
        ],

        "video_fake_votes": prediction[
            "fake_votes"
        ],

        "video_frames": prediction[
            "frames"
        ],

        # -----------------------------
        # Temporal
        # -----------------------------

        "temporal_score": temporal_score,

        # -----------------------------
        # SVIM
        # -----------------------------

        "mfcc_score": (
            float(
                svim_result.mfcc_score
            )
            if svim_result is not None
            else None
        ),

        "acoustic_pattern_score": (
            float(
                svim_result.acoustic_pattern_score
            )
            if svim_result is not None
            else None
        ),

        "audio_score": (
            float(
                svim_result.audio_score
            )
            if svim_result is not None
            else None
        ),

        # -----------------------------
        # DTFE
        # -----------------------------

        "final_score": final_score,

        "classification": classification.value,

        "fusion_confidence": (
            fusion_result
            .fusion_confidence_level
            .value
        ),

        "summary": fusion_result.summary,
    }
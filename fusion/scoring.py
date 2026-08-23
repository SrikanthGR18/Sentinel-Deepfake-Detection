"""Trust score fusion formulas from PROJECT_SPECIFICATION."""

from __future__ import annotations

from typing import Optional, Tuple

from fusion.types import FusionClassification, FusionConfidenceLevel

VIDEO_SPATIAL_WEIGHT = 0.60
VIDEO_TEMPORAL_WEIGHT = 0.40
AUDIO_MFCC_WEIGHT = 0.70
AUDIO_ACOUSTIC_WEIGHT = 0.30
FINAL_VIDEO_WEIGHT = 0.60
FINAL_AUDIO_WEIGHT = 0.40

CLASSIFICATION_FAKE_MAX = 40.0
CLASSIFICATION_SUSPICIOUS_MAX = 70.0


def round_score(value: float) -> float:
    """Clip to 0-100 and round to one decimal place."""
    clipped = float(max(0.0, min(100.0, value)))
    return round(clipped, 1)


def compute_video_score(
    spatial_score: float,
    temporal_score: float,
) -> float:
    combined = (
        VIDEO_SPATIAL_WEIGHT * spatial_score
        + VIDEO_TEMPORAL_WEIGHT * temporal_score
    )
    return round_score(combined)


def compute_audio_score(
    mfcc_score: float,
    acoustic_pattern_score: float,
) -> float:
    combined = (
        AUDIO_MFCC_WEIGHT * mfcc_score
        + AUDIO_ACOUSTIC_WEIGHT * acoustic_pattern_score
    )
    return round_score(combined)


def compute_final_score(
    video_score: float,
    audio_score: float,
) -> float:
    combined = (
        FINAL_VIDEO_WEIGHT * video_score
        + FINAL_AUDIO_WEIGHT * audio_score
    )
    return round_score(combined)


def classify_final_score(final_score: float) -> Tuple[FusionClassification, FusionConfidenceLevel]:
    if final_score <= CLASSIFICATION_FAKE_MAX:
        return FusionClassification.FAKE, FusionConfidenceLevel.LOW
    if final_score <= CLASSIFICATION_SUSPICIOUS_MAX:
        return FusionClassification.SUSPICIOUS, FusionConfidenceLevel.MEDIUM
    return FusionClassification.AUTHENTIC, FusionConfidenceLevel.HIGH


def dashboard_result_labels(
    classification: FusionClassification,
) -> Tuple[str, str]:
    """Map spec classification to existing dashboard strings and colors."""
    if classification == FusionClassification.FAKE:
        return "AI GENERATED / SUSPICIOUS", "red"
    if classification == FusionClassification.SUSPICIOUS:
        return "POSSIBLY FAKE / INCONCLUSIVE", "yellow"
    return "LIKELY REAL", "green"

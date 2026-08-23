"""Trust Score Fusion engine (DTFE)."""

from __future__ import annotations

from typing import Optional

from fusion.scoring import (
    classify_final_score,
    compute_audio_score,
    compute_final_score,
    compute_video_score,
    round_score,
)
from fusion.types import FusionInput, FusionMode, FusionResult


def _round_input(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round_score(float(value))


def fuse_trust_scores(inputs: FusionInput) -> FusionResult:
    """
    Fuse AFCP (spatial), temporal, and SVIM scores per project specification.

    All composite scores use one decimal place on a 0-100 scale.
    """
    spatial = _round_input(inputs.spatial_score)
    temporal = _round_input(inputs.temporal_score)
    mfcc = _round_input(inputs.mfcc_score)
    acoustic = _round_input(inputs.acoustic_pattern_score)

    video_score: Optional[float] = None
    audio_score: Optional[float] = None
    final_score: float
    mode: FusionMode
    summary: str

    if inputs.audio_only and inputs.has_audio:
        mode = FusionMode.AUDIO_ONLY
        audio_score = compute_audio_score(mfcc, acoustic) if mfcc is not None and acoustic is not None else None
        if audio_score is None:
            final_score = 50.0
            summary = "Audio-only upload; insufficient SVIM scores for fusion."
        else:
            final_score = audio_score
            summary = (
                f"Audio-only fusion: AudioScore={audio_score} "
                f"(0.7×MFCC + 0.3×Acoustic)."
            )

    elif inputs.has_audio and inputs.has_video:
        mode = FusionMode.FULL
        if spatial is None or temporal is None:
            final_score = 50.0
            summary = "Missing spatial or temporal score; fusion defaulted to neutral."
        else:
            video_score = compute_video_score(spatial, temporal)
            audio_score = (
                compute_audio_score(mfcc, acoustic)
                if mfcc is not None and acoustic is not None
                else None
            )
            if audio_score is None:
                final_score = video_score
                summary = f"Video-only fallback within full media: VideoScore={video_score}."
            else:
                final_score = compute_final_score(video_score, audio_score)
                summary = (
                    f"Full fusion: VideoScore={video_score} "
                    f"(0.6×Spatial+0.4×Temporal), AudioScore={audio_score}, "
                    f"FinalScore={final_score}."
                )

    elif inputs.has_video:
        mode = FusionMode.VIDEO_ONLY
        if spatial is None or temporal is None:
            final_score = 50.0
            summary = "Video-only upload without complete spatial/temporal scores."
        else:
            video_score = compute_video_score(spatial, temporal)
            final_score = video_score
            summary = (
                f"Video-only fusion: VideoScore={video_score} "
                f"(0.6×Spatial+0.4×Temporal)."
            )

    else:
        mode = FusionMode.VIDEO_ONLY
        final_score = 50.0
        summary = "No modality scores available; neutral fusion result."

    classification, confidence = classify_final_score(final_score)

    result = FusionResult(
        fusion_spatial_score=spatial,
        fusion_temporal_score=temporal,
        fusion_video_score=video_score,
        fusion_mfcc_score=mfcc,
        fusion_acoustic_pattern_score=acoustic,
        fusion_audio_score=audio_score,
        fusion_final_score=final_score,
        fusion_classification=classification.value,
        fusion_confidence_level=confidence,
        mode=mode,
        summary=summary,
    )
    _log_fusion(result)
    return result


def _log_fusion(result: FusionResult) -> None:
    print("\n========== TRUST SCORE FUSION (DTFE) ==========")
    print(f"Mode: {result.mode.value}")
    print(f"SpatialScore: {result.fusion_spatial_score}")
    print(f"TemporalScore: {result.fusion_temporal_score}")
    print(f"VideoScore: {result.fusion_video_score}")
    print(f"MFCCScore: {result.fusion_mfcc_score}")
    print(f"AcousticPatternScore: {result.fusion_acoustic_pattern_score}")
    print(f"AudioScore: {result.fusion_audio_score}")
    print(f"FinalScore: {result.fusion_final_score}")
    print(f"Classification: {result.fusion_classification}")
    print(f"ConfidenceLevel: {result.fusion_confidence_level.value}")
    print(f"Summary: {result.summary}")
    print("===============================================\n")

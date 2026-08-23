"""Temporal consistency analysis orchestration."""

from __future__ import annotations

from typing import List, Optional

from temporal.flow import compute_optical_flow_diagnostics, load_frame_sequence
from temporal.scoring import (
    compute_frame_transition_score,
    compute_motion_consistency_score,
    compute_temporal_score,
    compute_temporal_stability_score,
    confidence_level_from_score,
    status_and_reason,
)
from temporal.types import FlowDiagnostics, TemporalConfidenceLevel, TemporalResult

NEUTRAL_SCORE = 50.0


def _neutral_result(status: str, reason: str) -> TemporalResult:
    diagnostics = FlowDiagnostics(
        mean_flow_magnitude=0.0,
        flow_magnitude_std=0.0,
        mean_flow_angle=0.0,
        max_flow_magnitude=0.0,
        flow_spike_ratio=0.0,
        sudden_transition_count=0,
        frame_pair_count=0,
        processed_frame_count=0,
    )
    return TemporalResult(
        motion_consistency_score=NEUTRAL_SCORE,
        temporal_stability_score=NEUTRAL_SCORE,
        frame_transition_score=NEUTRAL_SCORE,
        temporal_score=NEUTRAL_SCORE,
        confidence_level=TemporalConfidenceLevel.MEDIUM,
        diagnostics=diagnostics,
        status=status,
        reason=reason,
    )


def analyze_temporal(
    video_path: Optional[str] = None,
    frame_paths: Optional[List[str]] = None,
) -> TemporalResult:
    """
    Analyze temporal consistency using OpenCV optical flow.

    Prefers AMMP preprocessed frame paths when available.
    """
    frames = load_frame_sequence(video_path=video_path, frame_paths=frame_paths)

    if len(frames) < 2:
        return _neutral_result(
            "NO VIDEO",
            "Insufficient frames for temporal optical-flow analysis.",
        )

    try:
        flow_data = compute_optical_flow_diagnostics(frames)
        if not flow_data:
            return _neutral_result("UNKNOWN", "Optical flow computation failed.")

        motion_score, motion_detail = compute_motion_consistency_score(
            flow_data["pair_mean_magnitudes"],
            flow_data["mean_flow_magnitude"],
        )
        stability_score, stability_detail = compute_temporal_stability_score(
            flow_data["flow_magnitude_std"],
            flow_data["mean_flow_magnitude"],
            flow_data.get("lip_region_flow_std"),
        )
        transition_score, transition_detail = compute_frame_transition_score(
            flow_data["flow_spike_ratio"],
            flow_data["max_flow_magnitude"],
        )
        temporal_score = compute_temporal_score(
            motion_score,
            stability_score,
            transition_score,
        )
        confidence = confidence_level_from_score(temporal_score)
        status, reason = status_and_reason(
            temporal_score,
            motion_detail,
            stability_detail,
            transition_detail,
        )

        diagnostics = FlowDiagnostics(
            mean_flow_magnitude=flow_data["mean_flow_magnitude"],
            flow_magnitude_std=flow_data["flow_magnitude_std"],
            mean_flow_angle=flow_data["mean_flow_angle"],
            max_flow_magnitude=flow_data["max_flow_magnitude"],
            flow_spike_ratio=flow_data["flow_spike_ratio"],
            sudden_transition_count=flow_data["sudden_transition_count"],
            frame_pair_count=flow_data["frame_pair_count"],
            processed_frame_count=flow_data["processed_frame_count"],
            lip_region_flow_std=flow_data.get("lip_region_flow_std"),
        )

        result = TemporalResult(
            motion_consistency_score=motion_score,
            temporal_stability_score=stability_score,
            frame_transition_score=transition_score,
            temporal_score=temporal_score,
            confidence_level=confidence,
            diagnostics=diagnostics,
            status=status,
            reason=reason,
        )
        _log_diagnostics(result)
        return result

    except Exception as exc:
        print("Temporal analysis error:", exc)
        return _neutral_result("ERROR", f"Temporal analysis failed: {exc}")


def _log_diagnostics(result: TemporalResult) -> None:
    diag = result.diagnostics
    print("\n========== TEMPORAL ANALYSIS ==========")
    print(f"Processed frames: {diag.processed_frame_count}")
    print(f"Frame pairs: {diag.frame_pair_count}")
    print(f"Mean flow magnitude: {diag.mean_flow_magnitude:.4f}")
    print(f"Flow magnitude std: {diag.flow_magnitude_std:.4f}")
    print(f"Mean flow angle (rad): {diag.mean_flow_angle:.4f}")
    print(f"Max flow magnitude: {diag.max_flow_magnitude:.4f}")
    print(f"Flow spike ratio: {diag.flow_spike_ratio:.4f}")
    print(f"Sudden transitions: {diag.sudden_transition_count}")
    print(f"MotionConsistencyScore: {result.motion_consistency_score}")
    print(f"TemporalStabilityScore: {result.temporal_stability_score}")
    print(f"FrameTransitionScore: {result.frame_transition_score}")
    print(f"TemporalScore: {result.temporal_score}")
    print(f"TemporalConfidenceLevel: {result.confidence_level.value}")
    print(f"TemporalScore (normalized): {result.temporal_score_normalized:.2f}")
    print(f"Status: {result.status}")
    print("=======================================\n")

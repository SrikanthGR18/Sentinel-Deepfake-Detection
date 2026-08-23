"""Continuous temporal sub-scores on a 0-100 scale."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from temporal.types import TemporalConfidenceLevel

# Reference values for naturalness heuristics.
FLOW_MEAN_REF = 3.0
FLOW_STD_REF = 2.0
FLOW_MAX_REF = 12.0
SPIKE_RATIO_GOOD = 0.10
SPIKE_RATIO_BAD = 0.45
FROZEN_FLOW_MEAN = 0.15


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _ratio_score(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return _clamp01(value / reference)


def compute_motion_consistency_score(
    pair_mean_magnitudes: List[float],
    mean_flow_magnitude: float,
) -> Tuple[float, str]:
    if len(pair_mean_magnitudes) < 2:
        return 50.0, "insufficient motion samples"

    mean_motion = float(sum(pair_mean_magnitudes) / len(pair_mean_magnitudes))
    if mean_motion < 1e-6:
        return 15.0, "near-static sequence"

    cv_motion = float(np.std(pair_mean_magnitudes) / mean_motion)
    naturalness = _clamp01(1.0 / (1.0 + cv_motion))

    if mean_flow_magnitude < FROZEN_FLOW_MEAN:
        naturalness *= 0.6

    score = round(100.0 * naturalness)
    detail = "stable inter-frame motion" if score >= 70 else "irregular motion energy"
    return score, detail


def compute_temporal_stability_score(
    flow_magnitude_std: float,
    mean_flow_magnitude: float,
    lip_region_flow_std: float | None,
) -> Tuple[float, str]:
    stability_from_std = _clamp01(1.0 - (flow_magnitude_std / FLOW_STD_REF))
    level_component = _ratio_score(mean_flow_magnitude, FLOW_MEAN_REF)
    naturalness = 0.55 * stability_from_std + 0.45 * level_component

    if lip_region_flow_std is not None:
        lip_component = _ratio_score(lip_region_flow_std, 1.5)
        naturalness = 0.75 * naturalness + 0.25 * lip_component

    if flow_magnitude_std > FLOW_STD_REF * 1.5:
        naturalness *= 0.75

    score = round(100.0 * _clamp01(naturalness))
    detail = "consistent optical flow" if score >= 70 else "unstable flow field"
    return score, detail


def compute_frame_transition_score(
    flow_spike_ratio: float,
    max_flow_magnitude: float,
) -> Tuple[float, str]:
    spike_penalty = _clamp01(flow_spike_ratio / SPIKE_RATIO_BAD)
    spike_component = 1.0 - spike_penalty

    max_component = _clamp01(1.0 - max(0.0, max_flow_magnitude - FLOW_MAX_REF) / FLOW_MAX_REF)
    naturalness = 0.65 * spike_component + 0.35 * max_component

    if flow_spike_ratio <= SPIKE_RATIO_GOOD:
        naturalness = min(1.0, naturalness + 0.1)

    score = round(100.0 * _clamp01(naturalness))
    detail = (
        "smooth frame transitions"
        if score >= 70
        else "abrupt frame or motion spikes detected"
    )
    return score, detail


def compute_temporal_score(
    motion_consistency_score: float,
    temporal_stability_score: float,
    frame_transition_score: float,
) -> float:
    combined = (
        0.40 * motion_consistency_score
        + 0.35 * temporal_stability_score
        + 0.25 * frame_transition_score
    )
    return float(max(0.0, min(100.0, round(combined))))


def confidence_level_from_score(temporal_score: float) -> TemporalConfidenceLevel:
    if temporal_score <= 40:
        return TemporalConfidenceLevel.LOW
    if temporal_score <= 70:
        return TemporalConfidenceLevel.MEDIUM
    return TemporalConfidenceLevel.HIGH


def status_and_reason(
    temporal_score: float,
    motion_detail: str,
    stability_detail: str,
    transition_detail: str,
) -> Tuple[str, str]:
    if temporal_score <= 40:
        status = "SUSPICIOUS / INCONSISTENT"
        reason = (
            "Temporal analysis detected inconsistent motion: "
            f"motion ({motion_detail}); stability ({stability_detail}); "
            f"transitions ({transition_detail})."
        )
    elif temporal_score <= 70:
        status = "MODERATELY CONSISTENT"
        reason = (
            "Temporal analysis found mixed frame-to-frame behavior: "
            f"motion ({motion_detail}); stability ({stability_detail}); "
            f"transitions ({transition_detail})."
        )
    else:
        status = "TEMPORALLY CONSISTENT"
        reason = (
            "Temporal analysis detected natural optical flow patterns: "
            f"motion ({motion_detail}); stability ({stability_detail}); "
            f"transitions ({transition_detail})."
        )
    return status, reason

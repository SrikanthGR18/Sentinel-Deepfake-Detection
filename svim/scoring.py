"""Continuous, explainable SVIM scoring on a 0-100 scale."""

from __future__ import annotations

from typing import Dict, Tuple

# Reference values for naturalness mapping (hand-tuned heuristics).
MFCC_VAR_REF = 4000.0
MFCC_VAR_LOW = 400.0
MFCC_DELTA_VAR_REF = 500.0

CONTRAST_STD_REF = 800.0
CONTRAST_STD_LOW = 200.0
RMS_STD_REF = 0.30
RMS_STD_LOW = 0.05
RMS_MEAN_REF = 0.12
ZCR_MEAN_REF = 0.08
ZCR_MEAN_LOW = 0.03
SIGNAL_VAR_REF = 0.002


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _ratio_score(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return _clamp01(value / reference)


def compute_mfcc_score(features: Dict[str, float]) -> Tuple[float, str]:
    mfcc_var = features.get("mfcc_variance", 0.0)
    mfcc_delta_var = features.get("mfcc_delta_variance", 0.0)

    var_component = _ratio_score(mfcc_var, MFCC_VAR_REF)
    delta_component = _ratio_score(mfcc_delta_var, MFCC_DELTA_VAR_REF)
    naturalness = 0.65 * var_component + 0.35 * delta_component

    notes = []
    if mfcc_var < MFCC_VAR_LOW:
        naturalness *= 0.55
        notes.append("low MFCC variance")
    if mfcc_delta_var < MFCC_DELTA_VAR_REF * 0.25:
        naturalness *= 0.85
        notes.append("flat MFCC dynamics")

    score = round(100.0 * _clamp01(naturalness))
    detail = ", ".join(notes) if notes else "stable cepstral variation"
    return score, detail


def compute_acoustic_pattern_score(features: Dict[str, float]) -> Tuple[float, str]:
    contrast_std = features.get("spectral_contrast_std", 0.0)
    rms_std = features.get("rms_std", 0.0)
    rms_mean = features.get("rms_mean", 0.0)
    zcr_mean = features.get("zcr_mean", 0.0)
    signal_var = features.get("signal_variance", 0.0)

    contrast_component = _ratio_score(contrast_std, CONTRAST_STD_REF)
    rms_mod_component = _ratio_score(rms_std, RMS_STD_REF)
    rms_level_component = _ratio_score(rms_mean, RMS_MEAN_REF)
    zcr_component = _ratio_score(zcr_mean, ZCR_MEAN_REF)
    var_component = _ratio_score(signal_var, SIGNAL_VAR_REF)

    naturalness = (
        0.30 * contrast_component
        + 0.25 * rms_mod_component
        + 0.15 * rms_level_component
        + 0.20 * zcr_component
        + 0.10 * var_component
    )

    notes = []
    if contrast_std < CONTRAST_STD_LOW:
        naturalness *= 0.75
        notes.append("low spectral contrast diversity")
    if rms_std < RMS_STD_LOW:
        naturalness *= 0.80
        notes.append("stable synthetic energy")
    if zcr_mean < ZCR_MEAN_LOW:
        naturalness *= 0.80
        notes.append("robotic zero-crossing pattern")
    if signal_var < SIGNAL_VAR_REF * 0.05:
        naturalness *= 0.85
        notes.append("very flat waveform")

    score = round(100.0 * _clamp01(naturalness))
    detail = ", ".join(notes) if notes else "natural acoustic modulation"
    return score, detail


def compute_audio_score(mfcc_score: float, acoustic_pattern_score: float) -> float:
    combined = 0.7 * mfcc_score + 0.3 * acoustic_pattern_score
    return float(max(0.0, min(100.0, round(combined))))


def status_and_reason(
    audio_score: float,
    mfcc_detail: str,
    acoustic_detail: str,
) -> Tuple[str, str]:
    if audio_score <= 40:
        status = "SUSPICIOUS / AI"
        reason = (
            "SVIM detected synthetic speech cues: "
            f"MFCC ({mfcc_detail}); acoustic pattern ({acoustic_detail})."
        )
    elif audio_score <= 70:
        status = "POSSIBLY FAKE"
        reason = (
            "SVIM detected moderate irregularities: "
            f"MFCC ({mfcc_detail}); acoustic pattern ({acoustic_detail})."
        )
    else:
        status = "AUTHENTIC / REAL"
        reason = (
            "SVIM detected natural vocal dynamics: "
            f"MFCC ({mfcc_detail}); acoustic pattern ({acoustic_detail})."
        )
    return status, reason

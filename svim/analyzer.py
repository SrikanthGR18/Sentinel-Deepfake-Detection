"""SVIM analysis orchestration."""

from __future__ import annotations

import os

from svim.features import extract_features, load_and_prepare
from svim.scoring import (
    compute_acoustic_pattern_score,
    compute_audio_score,
    compute_mfcc_score,
    status_and_reason,
)
from svim.types import SVIMResult

NEUTRAL_SCORE = 50.0


def _neutral_result(status: str, reason: str) -> SVIMResult:
    return SVIMResult(
        mfcc_score=NEUTRAL_SCORE,
        acoustic_pattern_score=NEUTRAL_SCORE,
        audio_score=NEUTRAL_SCORE,
        signal_variance=0.0,
        spectral_contrast=0.0,
        rms_energy=0.0,
        zcr=0.0,
        status=status,
        reason=reason,
    )


def analyze_audio(audio_path: str) -> SVIMResult:
    """
    Run Spectral Voice Irregularity Mapping on an audio file.

    Returns scores on a 0-100 scale with MFCC and acoustic sub-scores.
    """
    if not os.path.exists(audio_path):
        return _neutral_result("NO AUDIO", "Audio file missing.")

    try:
        y, sr = load_and_prepare(audio_path)
        if len(y) == 0:
            return _neutral_result("UNKNOWN", "Empty audio.")

        features = extract_features(y, sr)
        if not features:
            return _neutral_result("UNKNOWN", "Feature extraction failed.")

        mfcc_score, mfcc_detail = compute_mfcc_score(features)
        acoustic_score, acoustic_detail = compute_acoustic_pattern_score(features)
        audio_score = compute_audio_score(mfcc_score, acoustic_score)
        status, reason = status_and_reason(audio_score, mfcc_detail, acoustic_detail)

        result = SVIMResult(
            mfcc_score=mfcc_score,
            acoustic_pattern_score=acoustic_score,
            audio_score=audio_score,
            signal_variance=features["signal_variance"],
            spectral_contrast=features["spectral_contrast_std"],
            rms_energy=features["rms_mean"],
            zcr=features["zcr_mean"],
            status=status,
            reason=reason,
            mfcc_variance=features.get("mfcc_variance"),
            mfcc_delta_variance=features.get("mfcc_delta_variance"),
            spectral_contrast_std=features.get("spectral_contrast_std"),
            rms_std=features.get("rms_std"),
            zcr_std=features.get("zcr_std"),
        )

        _log_diagnostics(features, result)
        return result

    except Exception as exc:
        print("SVIM Audio Error:", exc)
        return _neutral_result("ERROR", f"SVIM processing failed: {exc}")


def _log_diagnostics(features: dict, result: SVIMResult) -> None:
    print("\n========== SVIM AUDIO DIAGNOSTICS ==========")
    print(f"MFCC variance: {features.get('mfcc_variance', 0):.2f}")
    print(f"MFCC delta variance: {features.get('mfcc_delta_variance', 0):.2f}")
    print(f"Spectral contrast std: {features.get('spectral_contrast_std', 0):.2f}")
    print(f"RMS mean: {features.get('rms_mean', 0):.6f}")
    print(f"RMS std: {features.get('rms_std', 0):.6f}")
    print(f"ZCR mean: {features.get('zcr_mean', 0):.6f}")
    print(f"Signal variance: {features.get('signal_variance', 0):.8f}")
    print(f"MFCCScore: {result.mfcc_score}")
    print(f"AcousticPatternScore: {result.acoustic_pattern_score}")
    print(f"AudioScore: {result.audio_score}")
    print(f"Status: {result.status}")
    print("============================================\n")

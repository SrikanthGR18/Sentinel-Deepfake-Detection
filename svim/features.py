"""SVIM feature extraction (Librosa + NumPy)."""

from __future__ import annotations

from typing import Dict, Tuple

import librosa
import numpy as np

SAMPLE_RATE = 16000
MAX_DURATION_SEC = 30.0
N_MFCC = 13


def load_and_prepare(audio_path: str) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION_SEC)
    if len(y) == 0:
        return y, sr
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract MFCC, spectral contrast, RMS, ZCR, and signal variance."""
    if len(y) == 0:
        return {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "mfcc_variance": float(np.var(mfcc)),
        "mfcc_delta_variance": float(np.var(mfcc_delta)),
        "spectral_contrast_mean": float(np.mean(contrast)),
        "spectral_contrast_std": float(np.std(contrast)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "signal_variance": float(np.var(y)),
    }

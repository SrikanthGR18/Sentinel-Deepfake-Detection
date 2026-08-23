"""Audio quality assessment and normalization."""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
MAX_DURATION_SEC = 30.0


def _estimate_noise_level(y: np.ndarray, sr: int) -> float:
    """High-frequency energy ratio as a simple noise proxy (0–1 scale)."""
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    total_energy = float(np.sum(stft) + 1e-9)
    high_band = stft[freqs >= 4000.0, :]
    high_energy = float(np.sum(high_band))
    return float(np.clip(high_energy / total_energy, 0.0, 1.0))


def assess_audio_signal(y: np.ndarray, sr: int) -> dict:
    """Compute noise level, signal variance, and RMS energy."""
    if len(y) == 0:
        return {}

    noise_level = _estimate_noise_level(y, sr)
    signal_variance = float(np.var(y))
    rms = librosa.feature.rms(y=y)[0]
    rms_energy = float(np.mean(rms))
    return {
        "noise_level": noise_level,
        "signal_variance": signal_variance,
        "rms_energy": rms_energy,
    }


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=MAX_DURATION_SEC)
    return y, sr


def normalize_audio(y: np.ndarray, sr: int) -> np.ndarray:
    if len(y) == 0:
        return y
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y


def save_wav(path: str, y: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, y, sr)


def process_audio_file(
    source_path: str,
    output_path: str,
) -> Tuple[dict, Optional[str]]:
    """
    Load audio, assess quality on raw signal, normalize, and write WAV.
    Returns metrics dict and output path (None on failure).
    """
    if not os.path.exists(source_path):
        return {}, None

    try:
        y, sr = load_audio(source_path)
        if len(y) == 0:
            return {}, None

        metrics = assess_audio_signal(y, sr)
        y_norm = normalize_audio(y, sr)
        if len(y_norm) == 0:
            return metrics, None

        save_wav(output_path, y_norm, sr)
        if not os.path.exists(output_path):
            return metrics, None

        return metrics, output_path
    except Exception as exc:
        print("AMMP audio processing error:", exc)
        return {}, None


def resolve_audio_source(
    media_path: str,
    output_dir: str,
    uid: str,
    is_audio_only: bool,
    extract_audio_fn: Optional[Callable[[str, str], bool]],
) -> Optional[str]:
    """Return a WAV path suitable for assessment (extracted or copied)."""
    raw_path = os.path.join(output_dir, f"{uid}_raw.wav")

    if is_audio_only:
        ext = media_path.rsplit(".", 1)[-1].lower()
        if ext == "wav":
            return media_path
        y, sr = load_audio(media_path)
        save_wav(raw_path, y, sr)
        return raw_path if os.path.exists(raw_path) else None

    if extract_audio_fn is None:
        return None

    if extract_audio_fn(media_path, raw_path):
        return raw_path

    return None

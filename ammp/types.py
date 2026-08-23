"""Structured types for AMMP preprocessing outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AMMPMetrics:
    """Quality metrics from Adaptive Multi-Modal Preprocessing."""

    blur: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    motion_consistency: Optional[float] = None
    noise_level: Optional[float] = None
    signal_variance: Optional[float] = None
    rms_energy: Optional[float] = None


@dataclass
class AMMPResult:
    """Outputs produced by the AMMP pipeline."""

    metrics: AMMPMetrics = field(default_factory=AMMPMetrics)
    frame_paths: List[str] = field(default_factory=list)
    frames_dir: Optional[str] = None
    preprocessed_audio_path: Optional[str] = None
    raw_audio_path: Optional[str] = None
    video_ok: bool = False
    audio_ok: bool = False
    is_audio_only: bool = False
    message: str = ""

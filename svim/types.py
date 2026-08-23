"""Structured types for SVIM audio analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SVIMResult:
    """Spectral Voice Irregularity Mapping analysis output (0-100 scores)."""

    mfcc_score: float
    acoustic_pattern_score: float
    audio_score: float
    signal_variance: float
    spectral_contrast: float
    rms_energy: float
    zcr: float
    status: str
    reason: str
    # Raw feature details for diagnostics / future charts
    mfcc_variance: Optional[float] = None
    mfcc_delta_variance: Optional[float] = None
    spectral_contrast_std: Optional[float] = None
    rms_std: Optional[float] = None
    zcr_std: Optional[float] = None

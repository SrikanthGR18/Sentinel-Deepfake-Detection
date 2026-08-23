"""Types for Trust Score Fusion (DTFE)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FusionMode(str, Enum):
    FULL = "FULL"
    VIDEO_ONLY = "VIDEO_ONLY"
    AUDIO_ONLY = "AUDIO_ONLY"


class FusionConfidenceLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class FusionClassification(str, Enum):
    FAKE = "Fake"
    SUSPICIOUS = "Suspicious"
    AUTHENTIC = "Authentic"


@dataclass
class FusionInput:
    """Inputs gathered from AFCP, Temporal, and SVIM modules."""

    spatial_score: Optional[float] = None
    temporal_score: Optional[float] = None
    mfcc_score: Optional[float] = None
    acoustic_pattern_score: Optional[float] = None
    audio_only: bool = False
    has_audio: bool = False
    has_video: bool = True


@dataclass
class FusionResult:
    """Spec-aligned trust fusion outputs on a 0-100 scale (one decimal for composites)."""

    fusion_spatial_score: Optional[float]
    fusion_temporal_score: Optional[float]
    fusion_video_score: Optional[float]
    fusion_mfcc_score: Optional[float]
    fusion_acoustic_pattern_score: Optional[float]
    fusion_audio_score: Optional[float]
    fusion_final_score: float
    fusion_classification: str
    fusion_confidence_level: FusionConfidenceLevel
    mode: FusionMode
    summary: str

    @property
    def final_score_normalized(self) -> float:
        return float(max(0.0, min(1.0, self.fusion_final_score / 100.0)))

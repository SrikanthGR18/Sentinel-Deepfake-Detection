"""Structured types for temporal consistency analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TemporalConfidenceLevel(str, Enum):
    """Interpretable confidence band for temporal authenticity (DTFE-ready)."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class FlowDiagnostics:
    """Raw optical-flow statistics used for scoring and future charts."""

    mean_flow_magnitude: float
    flow_magnitude_std: float
    mean_flow_angle: float
    max_flow_magnitude: float
    flow_spike_ratio: float
    sudden_transition_count: int
    frame_pair_count: int
    processed_frame_count: int
    lip_region_flow_std: Optional[float] = None


@dataclass
class TemporalResult:
    """
    Temporal analysis output on a 0-100 scale.

    Use temporal_score_normalized for future DTFE fusion without API changes.
    """

    motion_consistency_score: float
    temporal_stability_score: float
    frame_transition_score: float
    temporal_score: float
    confidence_level: TemporalConfidenceLevel
    diagnostics: FlowDiagnostics
    status: str
    reason: str

    @property
    def temporal_score_normalized(self) -> float:
        """Map TemporalScore to 0-1 for future DTFE (e.g. 0.6*video + 0.4*temporal)."""
        return float(max(0.0, min(1.0, self.temporal_score / 100.0)))

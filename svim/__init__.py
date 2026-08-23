"""Spectral Voice Irregularity Mapping (SVIM) for Sentinel."""

from svim.analyzer import analyze_audio
from svim.types import SVIMResult

__all__ = ["SVIMResult", "analyze_audio"]

"""Frame loading and OpenCV Farneback optical flow."""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

MAX_FRAMES = 60
TARGET_WIDTH = 480

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= TARGET_WIDTH:
        return frame
    scale = TARGET_WIDTH / width
    new_size = (TARGET_WIDTH, int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _iter_frames_from_paths(frame_paths: List[str]) -> Iterator[np.ndarray]:
    for path in sorted(frame_paths):
        frame = cv2.imread(path)
        if frame is not None:
            yield _resize_frame(frame)


def _iter_frames_from_video(video_path: str) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        return

    sample_count = min(MAX_FRAMES, total_frames)
    step = max(1, total_frames // sample_count)
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        yield _resize_frame(frame)
        frame_idx += step

    cap.release()


def load_frame_sequence(
    video_path: Optional[str] = None,
    frame_paths: Optional[List[str]] = None,
) -> List[np.ndarray]:
    if frame_paths:
        frames = list(_iter_frames_from_paths(frame_paths))
        if frames:
            return frames

    if video_path and os.path.exists(video_path):
        return list(_iter_frames_from_video(video_path))

    return []


def _lip_region_flow_std(flow: np.ndarray, frame_shape: Tuple[int, int]) -> float:
    """Lower-center ROI flow magnitude std as a lightweight lip-motion proxy."""
    height, width = frame_shape
    y0 = int(height * 0.55)
    x0 = int(width * 0.30)
    x1 = int(width * 0.70)
    roi = flow[y0:height, x0:x1]
    if roi.size == 0:
        return 0.0
    magnitude = np.sqrt(roi[..., 0] ** 2 + roi[..., 1] ** 2)
    return float(np.std(magnitude))


def compute_optical_flow_diagnostics(frames: List[np.ndarray]) -> Optional[dict]:
    """
    Compute Farneback optical flow between consecutive frames.

    Returns dict of diagnostics or None if insufficient frames.
    """
    if len(frames) < 2:
        return None

    pair_mean_magnitudes: List[float] = []
    pair_mean_angles: List[float] = []
    pair_max_magnitudes: List[float] = []
    lip_flow_stds: List[float] = []
    sudden_spikes = 0

    prev_gray: Optional[np.ndarray] = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK_PARAMS)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])

        mean_mag = float(np.mean(magnitude))
        max_mag = float(np.max(magnitude))
        mean_ang = float(np.mean(angle))

        pair_mean_magnitudes.append(mean_mag)
        pair_max_magnitudes.append(max_mag)
        pair_mean_angles.append(mean_ang)
        lip_flow_stds.append(_lip_region_flow_std(flow, gray.shape))

        prev_gray = gray

    if not pair_mean_magnitudes:
        return None

    magnitudes = np.array(pair_mean_magnitudes, dtype=np.float64)
    mean_mag = float(np.mean(magnitudes))
    std_mag = float(np.std(magnitudes))
    spike_threshold = mean_mag + 2.0 * std_mag if std_mag > 1e-9 else mean_mag * 2.5

    for mag in pair_mean_magnitudes:
        if mag > spike_threshold and mag > 1.0:
            sudden_spikes += 1

    frame_pair_count = len(pair_mean_magnitudes)
    flow_spike_ratio = float(sudden_spikes / frame_pair_count) if frame_pair_count else 0.0

    return {
        "mean_flow_magnitude": mean_mag,
        "flow_magnitude_std": std_mag,
        "mean_flow_angle": float(np.mean(pair_mean_angles)),
        "max_flow_magnitude": float(np.max(pair_max_magnitudes)),
        "flow_spike_ratio": flow_spike_ratio,
        "sudden_transition_count": sudden_spikes,
        "frame_pair_count": frame_pair_count,
        "processed_frame_count": len(frames),
        "lip_region_flow_std": float(np.mean(lip_flow_stds)) if lip_flow_stds else None,
        "pair_mean_magnitudes": pair_mean_magnitudes,
        "pair_max_magnitudes": pair_max_magnitudes,
    }

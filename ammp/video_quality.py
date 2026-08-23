"""Video quality assessment, frame selection, and enhancement."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

MAX_SAMPLE_FRAMES = 60
MAX_SELECTED_FRAMES = 60

BRIGHTNESS_LOW = 40.0
BRIGHTNESS_HIGH = 220.0
BLUR_DENOISE_THRESHOLD = 30.0
CONTRAST_CLAHE_THRESHOLD = 35.0


def _frame_metrics(gray: np.ndarray) -> Tuple[float, float, float]:
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return blur, brightness, contrast


def _motion_consistency_score(motion_vals: List[float]) -> float:
    if len(motion_vals) < 2:
        return 1.0
    mean_motion = float(np.mean(motion_vals))
    if mean_motion < 1e-6:
        return 0.0
    cv_motion = float(np.std(motion_vals)) / mean_motion
    return float(1.0 / (1.0 + cv_motion))


def _frame_quality_score(
    blur: float,
    brightness: float,
    contrast: float,
    static_ratio: float,
) -> float:
    blur_component = min(blur / 200.0, 1.0) * 0.35
    if BRIGHTNESS_LOW <= brightness <= BRIGHTNESS_HIGH:
        brightness_component = 0.25
    else:
        distance = min(
            abs(brightness - BRIGHTNESS_LOW),
            abs(brightness - BRIGHTNESS_HIGH),
        )
        brightness_component = max(0.0, 0.25 - distance / 400.0)
    contrast_component = min(contrast / 80.0, 1.0) * 0.25
    motion_component = max(0.0, 1.0 - static_ratio) * 0.15
    return blur_component + brightness_component + contrast_component + motion_component


def _enhance_frame(
    frame: np.ndarray,
    metrics_mean: Tuple[float, float, float],
) -> np.ndarray:
    blur_mean, brightness_mean, contrast_mean = metrics_mean
    enhanced = frame.copy()

    if contrast_mean < CONTRAST_CLAHE_THRESHOLD:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(
            cv2.merge([l_channel, a_channel, b_channel]),
            cv2.COLOR_LAB2BGR,
        )

    if brightness_mean < BRIGHTNESS_LOW:
        beta = int((BRIGHTNESS_LOW - brightness_mean) * 0.6)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.0, beta=beta)
    elif brightness_mean > BRIGHTNESS_HIGH:
        beta = -int((brightness_mean - BRIGHTNESS_HIGH) * 0.4)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=0.95, beta=beta)

    if blur_mean < BLUR_DENOISE_THRESHOLD:
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            h=4,
            hColor=4,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    return enhanced


def assess_video_quality(video_path: str) -> Tuple[dict, List[dict]]:
    """
    Sample frames from video and return aggregate metrics plus per-frame records.
    Each record: index, frame (BGR), blur, brightness, contrast, motion, static_ratio, score.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        return {}, []

    sample_count = min(MAX_SAMPLE_FRAMES, total_frames)
    step = max(1, total_frames // sample_count)

    blur_vals: List[float] = []
    brightness_vals: List[float] = []
    contrast_vals: List[float] = []
    motion_vals: List[float] = []
    frame_records: List[dict] = []

    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur, brightness, contrast = _frame_metrics(gray)
        motion = 0.0
        static_ratio = 1.0

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = float(np.mean(diff))
            motion_vals.append(motion)
            static_ratio = float(np.sum(diff <= 1) / diff.size)

        score = _frame_quality_score(blur, brightness, contrast, static_ratio)

        blur_vals.append(blur)
        brightness_vals.append(brightness)
        contrast_vals.append(contrast)

        frame_records.append(
            {
                "index": frame_idx,
                "frame": frame,
                "blur": blur,
                "brightness": brightness,
                "contrast": contrast,
                "motion": motion,
                "static_ratio": static_ratio,
                "score": score,
            }
        )

        prev_gray = gray
        frame_idx += step

    cap.release()

    if not blur_vals:
        return {}, []

    aggregates = {
        "blur": float(np.mean(blur_vals)),
        "brightness": float(np.mean(brightness_vals)),
        "contrast": float(np.mean(contrast_vals)),
        "motion_consistency": _motion_consistency_score(motion_vals),
    }
    return aggregates, frame_records


def select_and_save_frames(
    frame_records: List[dict],
    aggregates: dict,
    output_dir: str,
    uid: str,
) -> Tuple[List[str], str]:
    """
    Select top-quality frames, enhance them, and save as JPEG sequence.
    Returns sorted frame paths and the frames directory.
    """
    if not frame_records:
        return [], ""

    select_count = min(MAX_SELECTED_FRAMES, len(frame_records))
    ranked = sorted(frame_records, key=lambda item: item["score"], reverse=True)
    selected = sorted(ranked[:select_count], key=lambda item: item["index"])

    frames_dir = os.path.join(output_dir, f"{uid}_ammp_frames")
    os.makedirs(frames_dir, exist_ok=True)

    metrics_mean = (
        aggregates.get("blur", 0.0),
        aggregates.get("brightness", 128.0),
        aggregates.get("contrast", 0.0),
    )

    frame_paths: List[str] = []
    for seq, record in enumerate(selected):
        enhanced = _enhance_frame(record["frame"], metrics_mean)
        path = os.path.join(frames_dir, f"frame_{seq:04d}.jpg")
        cv2.imwrite(path, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        frame_paths.append(path)

    return frame_paths, frames_dir


def process_video(
    video_path: str,
    output_dir: str,
    uid: str,
) -> Tuple[dict, List[str], str]:
    """Run full video AMMP branch: assess, select, enhance, save frames."""
    aggregates, frame_records = assess_video_quality(video_path)
    if not aggregates:
        return {}, [], ""

    frame_paths, frames_dir = select_and_save_frames(
        frame_records,
        aggregates,
        output_dir,
        uid,
    )
    return aggregates, frame_paths, frames_dir

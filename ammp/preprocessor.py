"""AMMP orchestration: adaptive multi-modal preprocessing."""

from __future__ import annotations

import os
from typing import Callable, Optional

from ammp.audio_quality import process_audio_file, resolve_audio_source
from ammp.types import AMMPMetrics, AMMPResult
from ammp.video_quality import process_video

AUDIO_EXTENSIONS = {"wav", "mp3"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}


def _media_extension(media_path: str) -> str:
    return media_path.rsplit(".", 1)[-1].lower()


def is_audio_only_upload(media_path: str) -> bool:
    return _media_extension(media_path) in AUDIO_EXTENSIONS


def _log_metrics(metrics: AMMPMetrics, result: AMMPResult) -> None:
    print("\n========== AMMP PREPROCESSING ==========")
    print(f"Audio-only: {result.is_audio_only}")
    print(f"Video OK: {result.video_ok} | Audio OK: {result.audio_ok}")
    if metrics.blur is not None:
        print(f"Blur: {metrics.blur:.2f}")
        print(f"Brightness: {metrics.brightness:.2f}")
        print(f"Contrast: {metrics.contrast:.2f}")
        print(f"Motion consistency: {metrics.motion_consistency:.4f}")
    if metrics.noise_level is not None:
        print(f"Noise level: {metrics.noise_level:.4f}")
        print(f"Signal variance: {metrics.signal_variance:.6f}")
        print(f"RMS energy: {metrics.rms_energy:.6f}")
    if result.frames_dir:
        print(f"Frames dir: {result.frames_dir} ({len(result.frame_paths)} frames)")
    if result.preprocessed_audio_path:
        print(f"Preprocessed audio: {result.preprocessed_audio_path}")
    print(f"Message: {result.message}")
    print("==========================================\n")


def preprocess_media(
    media_path: str,
    output_dir: str,
    uid: str,
    extract_audio_fn: Optional[Callable[[str, str], bool]] = None,
) -> AMMPResult:
    """
    Run Adaptive Multi-Modal Preprocessing on uploaded media.

    Video: assess quality, select frames, enhance, save JPEG sequence (no MP4 rebuild).
    Audio: assess quality, normalize, write preprocessed WAV.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = AMMPMetrics()
    result = AMMPResult(metrics=metrics)
    result.is_audio_only = is_audio_only_upload(media_path)
    ext = _media_extension(media_path)

    preprocessed_audio = os.path.join(output_dir, f"{uid}_ammp.wav")

    # --- Video branch (skipped for audio-only uploads) ---
    if not result.is_audio_only and ext in VIDEO_EXTENSIONS:
        aggregates, frame_paths, frames_dir = process_video(
            media_path,
            output_dir,
            uid,
        )
        if aggregates:
            metrics.blur = aggregates["blur"]
            metrics.brightness = aggregates["brightness"]
            metrics.contrast = aggregates["contrast"]
            metrics.motion_consistency = aggregates["motion_consistency"]
            result.frame_paths = frame_paths
            result.frames_dir = frames_dir or None
            result.video_ok = len(frame_paths) > 0
            result.message = (
                f"Selected {len(frame_paths)} enhanced frames."
                if result.video_ok
                else "Video opened but no frames were saved."
            )
        else:
            result.message = "Video quality assessment failed; using original media for AFCP."

    elif result.is_audio_only:
        result.message = "Audio-only upload; video preprocessing skipped."

    # --- Audio branch ---
    raw_audio = resolve_audio_source(
        media_path,
        output_dir,
        uid,
        result.is_audio_only,
        extract_audio_fn,
    )
    result.raw_audio_path = raw_audio

    if raw_audio:
        audio_metrics, preprocessed_path = process_audio_file(
            raw_audio,
            preprocessed_audio,
        )
        if audio_metrics:
            metrics.noise_level = audio_metrics["noise_level"]
            metrics.signal_variance = audio_metrics["signal_variance"]
            metrics.rms_energy = audio_metrics["rms_energy"]
        if preprocessed_path:
            result.preprocessed_audio_path = preprocessed_path
            result.audio_ok = True
            if result.message:
                result.message += " "
            result.message += "Audio normalized and saved."
        elif not result.audio_ok:
            if result.message:
                result.message += " "
            result.message += "Audio assessment failed."
    else:
        if not result.is_audio_only:
            if result.message:
                result.message += " "
            result.message += "No audio stream available for AMMP."
        else:
            result.message = "Could not load audio-only file for AMMP."

    _log_metrics(metrics, result)
    return result

"""
training/prepare_generative_data.py
Extracts frames from raw source videos into datasets/generative_finetune/
for transfer learning without touching Celeb-DF or sentinel_clean.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Target directory
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "generative_finetune"

# Configure number of frames per clip
NUM_FRAMES = 10


def extract_video_frames(video_path: Path, output_folder: Path, prefix: str):
    output_folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    indices = np.unique(
        np.linspace(0, total_frames - 1, min(NUM_FRAMES, total_frames), dtype=int)
    )

    saved_count = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        # Save frame as JPEG (native resolution preserved)
        frame_name = f"{prefix}_frame_{i+1:02d}.jpg"
        save_path = output_folder / frame_name
        cv2.imwrite(str(save_path), frame)
        saved_count += 1

    cap.release()
    return saved_count


def process_directory(source_folder: Path, split: str, class_name: str):
    target_dir = OUTPUT_ROOT / split / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    videos = [v for v in source_folder.glob("*") if v.suffix.lower() in video_extensions]

    total_saved = 0
    for video in videos:
        count = extract_video_frames(video, target_dir, prefix=video.stem)
        total_saved += count
        print(f"[{split.upper()}/{class_name}] {video.name} -> {count} frames")

    return total_saved


if __name__ == "__main__":
    print("Run this helper to populate:")
    print(f"  {OUTPUT_ROOT / 'train'}")
    print(f"  {OUTPUT_ROOT / 'val'}")
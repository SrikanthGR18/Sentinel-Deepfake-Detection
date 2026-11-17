
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Four videos selected from official test results
VIDEOS = {
    "correct_id2_0008": ROOT / "datasets/CelebDF-v2/Celeb-real/id2_0008.mp4",
    "correct_id3_0001": ROOT / "datasets/CelebDF-v2/Celeb-real/id3_0001.mp4",
    "false_positive_id1_0007": ROOT / "datasets/CelebDF-v2/Celeb-real/id1_0007.mp4",
    "false_positive_id6_0005": ROOT / "datasets/CelebDF-v2/Celeb-real/id6_0005.mp4",
}

# Unique output directory; never overwrite previous results
OUTPUT = ROOT / "results/diagnostic_frames_20261102"

if OUTPUT.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUTPUT}\n"
        "Rename OUTPUT in this script before running again."
    )

OUTPUT.mkdir(parents=True)

SAMPLE_COUNT = 10
THUMB_WIDTH = 320
THUMB_HEIGHT = 180

contact_images = []

for video_name, video_path in VIDEOS.items():

    print(f"\nProcessing: {video_name}")

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    frame_indices = np.linspace(
        0, total_frames - 1,
        min(SAMPLE_COUNT, total_frames),
        dtype=int
    )

    video_output = OUTPUT / video_name
    video_output.mkdir()

    thumbnails = []

    for i, frame_index in enumerate(frame_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            print(f"Warning: Could not read frame {frame_index}")
            continue

        # Save original extracted frame
        frame_path = video_output / f"frame_{i + 1:02d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        # Create contact sheet thumbnail
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((THUMB_WIDTH, THUMB_HEIGHT))

        canvas = Image.new(
            "RGB",
            (THUMB_WIDTH, THUMB_HEIGHT),
            "black"
        )

        x = (THUMB_WIDTH - image.width) // 2
        y = (THUMB_HEIGHT - image.height) // 2
        canvas.paste(image, (x, y))

        thumbnails.append(canvas)

    cap.release()

    print(f"Saved {len(thumbnails)} frames to {video_output}")

    # Use five evenly spaced frames for the comparison sheet
    selected = [
        thumbnails[i]
        for i in [0, 2, 4, 6, 8]
        if i < len(thumbnails)
    ]

    contact_images.append((video_name, selected))


# Create comparison contact sheet
columns = 5
label_height = 35
row_height = THUMB_HEIGHT + label_height

sheet = Image.new(
    "RGB",
    (columns * THUMB_WIDTH, len(contact_images) * row_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for row, (video_name, thumbnails) in enumerate(contact_images):

    y = row * row_height

    draw.text(
        (10, y + 8),
        video_name,
        fill="black"
    )

    for col, thumbnail in enumerate(thumbnails):
        sheet.paste(
            thumbnail,
            (col * THUMB_WIDTH, y + label_height)
        )

contact_path = OUTPUT / "comparison_contact_sheet.jpg"
sheet.save(contact_path, quality=95)

print("\nDiagnostic extraction completed.")
print(f"Contact sheet: {contact_path}")
print(f"All extracted frames: {OUTPUT}")

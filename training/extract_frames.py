import os
import cv2
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===========================
# Configuration
# ===========================

DATASET_ROOT = Path("datasets/CelebDF-v2")
OUTPUT_ROOT = Path("datasets")
FRAME_SIZE = (224, 224)
FRAMES_PER_VIDEO = 10
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# -----------------------------
# Function 1
# -----------------------------
def get_video_files():

    real = []

    fake = []

    real_folders = [
        DATASET_ROOT / "Celeb-real",
        DATASET_ROOT / "YouTube-real"
    ]

    fake_folder = DATASET_ROOT / "Celeb-synthesis"

    for folder in real_folders:

        for file in folder.glob("*.mp4"):

            real.append(file)

    for file in fake_folder.glob("*.mp4"):

        fake.append(file)

    print(f"Real videos : {len(real)}")

    print(f"Fake videos : {len(fake)}")

    return real, fake


# -----------------------------
# Function 2
# -----------------------------
def split_dataset(real, fake):

    train_real, val_real = train_test_split(
        real,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    train_fake, val_fake = train_test_split(
        fake,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print("\nDataset Split")
    print("--------------------")
    print(f"Train Real : {len(train_real)}")
    print(f"Validation Real : {len(val_real)}")
    print(f"Train Fake : {len(train_fake)}")
    print(f"Validation Fake : {len(val_fake)}")

    return train_real, val_real, train_fake, val_fake
def create_output_dirs():

    folders = [
        OUTPUT_ROOT / "train" / "real",
        OUTPUT_ROOT / "train" / "fake",
        OUTPUT_ROOT / "val" / "real",
        OUTPUT_ROOT / "val" / "fake",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("\nOutput folders created.")
def extract_frames(video_list, save_folder, prefix):

    count = 0

    for video in tqdm(video_list):

        cap = cv2.VideoCapture(str(video))

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total < FRAMES_PER_VIDEO:
            cap.release()
            continue

        indices = [
            int(i * total / FRAMES_PER_VIDEO)
            for i in range(FRAMES_PER_VIDEO)
        ]

        for j, idx in enumerate(indices):

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.resize(frame, FRAME_SIZE)

            filename = f"{prefix}_{count:06d}_{j}.jpg"

            cv2.imwrite(
                str(save_folder / filename),
                frame
            )

        cap.release()

        count += 1

    print(f"{prefix} : {count} videos processed.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    real, fake = get_video_files()

    train_real, val_real, train_fake, val_fake = split_dataset(real, fake)

    create_output_dirs()

    print("\nExtracting Training Frames...")

    extract_frames(
        train_real,
        OUTPUT_ROOT / "train" / "real",
        "real"
    )

    extract_frames(
        train_fake,
        OUTPUT_ROOT / "train" / "fake",
        "fake"
    )

    print("\nExtracting Validation Frames...")

    extract_frames(
        val_real,
        OUTPUT_ROOT / "val" / "real",
        "real"
    )

    extract_frames(
        val_fake,
        OUTPUT_ROOT / "val" / "fake",
        "fake"
    )

    print("\nFinished.")
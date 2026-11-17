
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path("datasets/CelebDF-v2")
OUTPUT = Path("datasets/sentinel_clean")

TEST_LIST = ROOT / "List_of_testing_videos.txt"

FRAMES_PER_VIDEO = 10
FRAME_SIZE = (224, 224)
TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ==========================================
# READ OFFICIAL TEST LIST
# ==========================================

def read_test_list():

    excluded = set()

    with open(TEST_LIST, "r") as file:

        for line in file:

            parts = line.strip().split(maxsplit=1)

            if len(parts) != 2:
                continue

            label, relative_path = parts

            excluded.add(relative_path.replace("\\", "/"))

    return excluded


# ==========================================
# COLLECT REMAINING VIDEOS
# ==========================================

def collect_videos(excluded):

    real = []
    fake = []

    for folder in ["Celeb-real", "YouTube-real"]:

        for video in (ROOT / folder).glob("*.mp4"):

            relative = video.relative_to(ROOT).as_posix()

            if relative not in excluded:
                real.append(video)

    for video in (ROOT / "Celeb-synthesis").glob("*.mp4"):

        relative = video.relative_to(ROOT).as_posix()

        if relative not in excluded:
            fake.append(video)

    print("Remaining videos:")
    print("Real:", len(real))
    print("Fake:", len(fake))

    return real, fake


# ==========================================
# CREATE VIDEO-LEVEL SPLIT
# ==========================================

def split_videos(real, fake):

    train_real, val_real = train_test_split(
        real,
        train_size=TRAIN_RATIO,
        random_state=SEED
    )

    train_fake, val_fake = train_test_split(
        fake,
        train_size=TRAIN_RATIO,
        random_state=SEED
    )

    return {
        "train": {
            "real": train_real,
            "fake": train_fake
        },
        "val": {
            "real": val_real,
            "fake": val_fake
        }
    }


# ==========================================
# EXTRACT FRAMES
# ==========================================

def extract_video(video, destination):

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return 0

    indices = np.linspace(
        0,
        total - 1,
        min(FRAMES_PER_VIDEO, total),
        dtype=int
    )

    saved = 0

    video_id = f"{video.parent.name}_{video.stem}"

    for number, index in enumerate(indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = cv2.resize(
            frame,
            FRAME_SIZE
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        filename = f"{video_id}_{number:02d}.jpg"

        output_path = destination / filename

        if cv2.imwrite(str(output_path), frame):
            saved += 1

    cap.release()

    return saved


# ==========================================
# MAIN
# ==========================================

def main():

    if not ROOT.exists():
        raise FileNotFoundError(ROOT)

    if not TEST_LIST.exists():
        raise FileNotFoundError(TEST_LIST)

    # Never overwrite an existing dataset.
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. "
            "Inspect it before proceeding."
        )

    excluded = read_test_list()

    print("Official test videos excluded:", len(excluded))

    real, fake = collect_videos(excluded)

    splits = split_videos(real, fake)

    manifest_dir = OUTPUT / "manifests"
    manifest_dir.mkdir(parents=True)

    for split_name, classes in splits.items():

        rows = []

        for class_name, videos in classes.items():

            destination = (
                OUTPUT / split_name / class_name
            )

            destination.mkdir(parents=True)

            print(
                f"\nProcessing {split_name} "
                f"{class_name}: {len(videos)} videos"
            )

            for video in tqdm(videos):

                frames_saved = extract_video(
                    video,
                    destination
                )

                rows.append({
                    "video": video.relative_to(ROOT).as_posix(),
                    "label": 1 if class_name == "real" else 0,
                    "class": class_name,
                    "split": split_name,
                    "frames_saved": frames_saved
                })

        manifest_path = (
            manifest_dir / f"{split_name}.csv"
        )

        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "video",
                    "label",
                    "class",
                    "split",
                    "frames_saved"
                ]
            )

            writer.writeheader()
            writer.writerows(rows)

    print("\nClean dataset preparation complete.")
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import cv2

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.inference import predict_video


# =========================================================
# VIDEO PATH
# =========================================================

VIDEO_PATH = PROJECT_ROOT / "test_videos" / "fake" / "fake_video.mp4"

print("\n====================================")
print("       SENTINEL VIDEO TEST")
print("====================================")

print("Project root:")
print(PROJECT_ROOT)

print("\nVideo path:")
print(VIDEO_PATH)

print("\nFile exists:")
print(VIDEO_PATH.exists())

# =========================================================
# DIRECT OPENCV TEST
# =========================================================

print("\n========== OPENCV TEST ==========")

cap = cv2.VideoCapture(str(VIDEO_PATH))

print("Video opened:", cap.isOpened())

if cap.isOpened():

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Total frames:", total_frames)
    print("FPS:", fps)

    ret, frame = cap.read()

    print("First frame read:", ret)

    if ret:
        print("Frame shape:", frame.shape)

    cap.release()

else:

    print("ERROR: OpenCV could not open the video.")

    cap.release()

print("=================================")


# =========================================================
# SENTINEL MODEL TEST
# =========================================================

print("\n========== AI MODEL TEST ==========")

result = predict_video(str(VIDEO_PATH))

print("\nPrediction result:")
print(result)

print("===================================")
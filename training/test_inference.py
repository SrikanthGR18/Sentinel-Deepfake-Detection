import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.inference import predict_video

video = "datasets/CelebDF-v2/Celeb-real/id0_0001.mp4"

score, status, reason = predict_video(video)

print(score)
print(status)
print(reason)
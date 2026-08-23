import cv2
import torch
import numpy as np

from torchvision import transforms
from PIL import Image

from models.video_model import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()

model.load_state_dict(
    torch.load(
        "weights/best_model.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def extract_frames(video_path, num_frames=10):

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return []

    indices = np.linspace(
        0,
        total - 1,
        num_frames,
        dtype=int
    )

    frames = []

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if ret:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

    cap.release()

    return frames


def predict_video(video_path):

    frames = extract_frames(video_path)

    if len(frames) == 0:

        return {
            "score": 50.0,
            "status": "Error",
            "reason": "No frames extracted.",
            "real_probability": 0,
            "fake_probability": 0,
            "real_votes": 0,
            "fake_votes": 0,
            "frames": 0
        }

    real_probs = []
    fake_probs = []
    frame_predictions = []

    with torch.no_grad():

        for frame in frames:

            img = Image.fromarray(frame)

            img = transform(img)

            img = img.unsqueeze(0).to(DEVICE)

            output = model(img)

            prob = torch.softmax(output, dim=1)[0]

            fake_prob = float(prob[0].cpu())

            real_prob = float(prob[1].cpu())

            fake_probs.append(fake_prob)

            real_probs.append(real_prob)

            if real_prob >= fake_prob:
                frame_predictions.append("REAL")
            else:
                frame_predictions.append("FAKE")

    real_probability = np.mean(real_probs)

    fake_probability = np.mean(fake_probs)

    real_votes = frame_predictions.count("REAL")

    fake_votes = frame_predictions.count("FAKE")

    confidence = max(real_probability, fake_probability)

    print("\n========== AI MODEL ==========")
    print(f"Frames Analysed : {len(frame_predictions)}")
    print(f"REAL Votes      : {real_votes}")
    print(f"FAKE Votes      : {fake_votes}")
    print(f"REAL Probability: {real_probability*100:.2f}%")
    print(f"FAKE Probability: {fake_probability*100:.2f}%")
    print("==============================")

    if real_probability >= fake_probability:

        status = "Authentic"

        score = real_probability * 100

    else:

        status = "Deepfake"

        # DTFE expects a trust score.
        # Lower trust = more fake.
        score = (1 - fake_probability) * 100

    reason = (
        f"Analyzed {len(frame_predictions)} sampled frames. "
        f"REAL votes: {real_votes}, "
        f"FAKE votes: {fake_votes}. "
        f"Average confidence: {confidence*100:.2f}%."
    )

    return {
        "score": round(score, 2),
        "status": status,
        "reason": reason,
        "real_probability": round(real_probability * 100, 2),
        "fake_probability": round(fake_probability * 100, 2),
        "real_votes": real_votes,
        "fake_votes": fake_votes,
        "frames": len(frame_predictions)
    }
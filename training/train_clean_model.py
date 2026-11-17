
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

from models.video_model import get_model


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path("datasets/sentinel_clean")
CHECKPOINT = Path("weights/best_model_clean.pth")

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# DATASET
# ==========================================

class CleanDataset(Dataset):

    def __init__(self, split):

        self.samples = []

        for class_name, label in [
            ("fake", 0),
            ("real", 1)
        ]:

            folder = ROOT / split / class_name

            for path in folder.glob("*.jpg"):

                # Exclude the video that produced only one frame
                if path.name.startswith(
                    "Celeb-real_id27_0005_"
                ):
                    continue

                self.samples.append((path, label))

        print(
            f"{split}: {len(self.samples)} frames"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = transform(image)

        return image, label


# ==========================================
# DATA LOADERS
# ==========================================

train_dataset = CleanDataset("train")
val_dataset = CleanDataset("val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ==========================================
# CLASS-WEIGHTED LOSS
# ==========================================

labels = [
    label for _, label in train_dataset.samples
]

fake_count = labels.count(0)
real_count = labels.count(1)

total = fake_count + real_count

class_weights = torch.tensor([
    total / (2 * fake_count),
    total / (2 * real_count)
], dtype=torch.float32).to(DEVICE)

print("\nClass distribution:")
print("Fake:", fake_count)
print("Real:", real_count)
print("Class weights:", class_weights)


# ==========================================
# MODEL
# ==========================================

# Start from ImageNet weights.
# Do NOT load the old Sentinel checkpoint:
# it was trained using videos from the
# official Celeb-DF test-list collection.

model = get_model().to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING
# ==========================================

def train_one_epoch():

    model.train()

    total_loss = 0

    for images, labels in tqdm(
        train_loader,
        desc="Training"
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ==========================================
# VALIDATION
# ==========================================

def validate():

    model.eval()

    total_loss = 0

    actual = []
    predicted = []

    with torch.no_grad():

        for images, labels in tqdm(
            val_loader,
            desc="Validation"
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            actual.extend(labels.cpu().numpy())
            predicted.extend(predictions.cpu().numpy())

    metrics = {
        "loss": total_loss / len(val_loader),
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "macro_f1": f1_score(
            actual, predicted, average="macro"
        )
    }

    print("\nValidation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(actual, predicted))

    print("\nClassification report:")
    print(classification_report(
        actual,
        predicted,
        target_names=["FAKE", "REAL"],
        zero_division=0
    ))

    return metrics


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    best_f1 = -1

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(EPOCHS):

        print(
            f"\n========== EPOCH "
            f"{epoch + 1}/{EPOCHS} =========="
        )

        train_loss = train_one_epoch()

        metrics = validate()

        print(f"\nTrain loss: {train_loss:.4f}")

        if metrics["macro_f1"] > best_f1:

            best_f1 = metrics["macro_f1"]

            torch.save(
                model.state_dict(),
                CHECKPOINT
            )

            print("\nBest clean checkpoint saved!")

    print("\nTraining complete.")
    print("Best validation macro-F1:", best_f1)
    print("Checkpoint:", CHECKPOINT)

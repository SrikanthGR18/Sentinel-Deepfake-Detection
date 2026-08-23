import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DeepfakeDataset
from models.video_model import get_model

# ==========================
# Device
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", DEVICE)

# ==========================
# Hyperparameters
# ==========================

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10

# ==========================
# Dataset
# ==========================

train_dataset = DeepfakeDataset("datasets/train")
val_dataset = DeepfakeDataset("datasets/val")

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

# ==========================
# Model
# ==========================

model = get_model().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================
# Training
# ==========================

def train():

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    loss = running_loss / len(train_loader)

    accuracy = 100 * correct / total

    return loss, accuracy

# ==========================
# Validation
# ==========================

def validate():

    model.eval()

    running_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in tqdm(val_loader, desc="Validation"):

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    loss = running_loss / len(val_loader)

    accuracy = 100 * correct / total

    return loss, accuracy

# ==========================
# Main
# ==========================

if __name__ == "__main__":

    best_accuracy = 0

    Path("weights").mkdir(exist_ok=True)

    for epoch in range(EPOCHS):

        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        train_loss, train_acc = train()

        val_loss, val_acc = validate()

        print(f"\nTrain Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            torch.save(
                model.state_dict(),
                "weights/best_model.pth"
            )

            print("✅ Best model saved!")

    print("\nTraining Complete!")
"""
training/train_generative_finetune.py
Fine-tunes the classifier head with weight decay and lower learning rate
to preserve authentic video classification.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from models.video_model import get_model

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_CHECKPOINT = PROJECT_ROOT / "weights" / "best_model_clean.pth"
NEW_CHECKPOINT = PROJECT_ROOT / "weights" / "best_model_generative.pth"
DATA_ROOT = PROJECT_ROOT / "datasets" / "generative_finetune"

BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 3

print(f"Using device: {DEVICE}")

# ==========================================
# TRANSFORMS
# ==========================================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# DATASET
# ==========================================
class GenerativeDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.transform = transform
        self.samples = []

        for class_dir, label in [("fake", 0), ("real", 1)]:
            folder = split_dir / class_dir
            if folder.exists():
                for ext in ("*.jpg", "*.png", "*.jpeg"):
                    for file in folder.glob(ext):
                        self.samples.append((file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, label

# ==========================================
# MODEL SETUP
# ==========================================
def load_base_model():
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing base checkpoint: {BASE_CHECKPOINT}")

    model = get_model()
    state_dict = torch.load(BASE_CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    # Freeze entire feature extraction backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Fine-tune only the final classifier linear layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(DEVICE)

# ==========================================
# TRAINING
# ==========================================
def main():
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"

    train_data = GenerativeDataset(train_dir, transform=transform_train)
    val_data = GenerativeDataset(val_dir, transform=transform_val)

    print(f"Loaded {len(train_data)} train images, {len(val_data)} validation images.")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = load_base_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_macro_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"\n--- Epoch {epoch} Results ---")
        print(f"Train Loss: {total_train_loss / len(train_loader):.4f}")
        print(f"Val Loss:   {total_val_loss / len(val_loader):.4f}")
        print(f"Val Acc:    {val_acc * 100:.2f}% | Macro-F1: {val_f1 * 100:.2f}%")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(model.state_dict(), NEW_CHECKPOINT)
            print(f"--> Saved improved checkpoint: {NEW_CHECKPOINT}")

    print("\nTraining completed.")

if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import copy
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score

from dataset import DeepfakeDataset
from models.video_model import get_model


ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 2
PILOT_PER_CLASS = 1000

SOURCE = ROOT / "weights" / "best_model.pth"
OUTPUT = ROOT / "weights" / "best_model_finetuned.pth"
METRICS = ROOT / "results" / "finetune_baseline_metrics.json"

torch.manual_seed(42)

if not SOURCE.exists():
    raise FileNotFoundError(f"Baseline checkpoint not found: {SOURCE}")

train_full = DeepfakeDataset(str(ROOT / "datasets" / "train"))
val_full = DeepfakeDataset(str(ROOT / "datasets" / "val"))


def balanced_subset(dataset, per_class, seed=42):
    fake = [i for i, (_, y) in enumerate(dataset.samples) if y == 0]
    real = [i for i, (_, y) in enumerate(dataset.samples) if y == 1]

    generator = torch.Generator().manual_seed(seed)
    fake = [fake[i] for i in torch.randperm(len(fake), generator=generator).tolist()[:per_class]]
    real = [real[i] for i in torch.randperm(len(real), generator=generator).tolist()[:per_class]]

    return Subset(dataset, fake + real)


train_set = balanced_subset(train_full, PILOT_PER_CLASS, seed=42)
val_set = balanced_subset(val_full, PILOT_PER_CLASS, seed=123)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    pin_memory=torch.cuda.is_available()
)

model = get_model().to(DEVICE)
state = torch.load(SOURCE, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)

# Balanced subset means both classes contribute equally.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def run_epoch(loader, training):
    model.train(training)
    loss_sum = 0.0
    labels_all, predictions_all = [], []

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in tqdm(loader, desc="Train" if training else "Validate"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(outputs.argmax(1).cpu().tolist())

    return {
        "loss": loss_sum / len(loader.dataset),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels_all, predictions_all)
        ),
        "macro_f1": float(
            f1_score(labels_all, predictions_all, average="macro", zero_division=0)
        ),
    }


def main():
    ROOT.joinpath("results").mkdir(exist_ok=True)
    best_key = (-1.0, -1.0)
    history = []

    print("Device:", DEVICE)
    print("Starting from:", SOURCE)
    print("Training subset:", len(train_set))
    print("Validation subset:", len(val_set))

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_metrics = run_epoch(train_loader, True)
        val_metrics = run_epoch(val_loader, False)

        print("Train:", train_metrics)
        print("Validation:", val_metrics)

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        })

        key = (
            val_metrics["balanced_accuracy"],
            val_metrics["macro_f1"],
        )
        if key > best_key:
            best_key = key
            torch.save(model.state_dict(), OUTPUT)
            print("Saved:", OUTPUT)

        METRICS.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\nDone. Baseline checkpoint was not modified.")
    print("Experimental checkpoint:", OUTPUT)
    print("Metrics:", METRICS)


if __name__ == "__main__":
    main()

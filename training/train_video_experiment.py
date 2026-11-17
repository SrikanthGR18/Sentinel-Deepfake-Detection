
import sys
from pathlib import Path
import json
import copy

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from dataset import DeepfakeDataset
from models.video_model import get_model


# ==========================
# CONFIGURATION
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 3

# Pilot mode uses a subset to check the experiment quickly.
# Set PILOT = False only after the pilot works correctly.
PILOT = True
PILOT_PER_CLASS = 500

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"

CHECKPOINT = WEIGHTS_DIR / "best_model_weighted_experiment.pth"
METRICS_FILE = RESULTS_DIR / "weighted_experiment_metrics.json"

torch.manual_seed(42)

print("Device:", DEVICE)
print("Pilot mode:", PILOT)


# ==========================
# DATA
# ==========================

train_dataset = DeepfakeDataset(str(ROOT / "datasets" / "train"))
val_dataset = DeepfakeDataset(str(ROOT / "datasets" / "val"))


def select_balanced_subset(dataset, per_class):
    """Select a reproducible, balanced subset for pilot runs."""
    fake_indices = []
    real_indices = []

    for idx, (_, label) in enumerate(dataset.samples):
        if label == 0:
            fake_indices.append(idx)
        elif label == 1:
            real_indices.append(idx)

    generator = torch.Generator().manual_seed(42)

    fake_order = torch.randperm(
        len(fake_indices), generator=generator
    ).tolist()
    real_order = torch.randperm(
        len(real_indices), generator=generator
    ).tolist()

    chosen = (
        [fake_indices[i] for i in fake_order[:per_class]]
        + [real_indices[i] for i in real_order[:per_class]]
    )

    subset = copy.copy(dataset)
    subset.samples = [dataset.samples[i] for i in chosen]
    return subset


if PILOT:
    train_dataset = select_balanced_subset(
        train_dataset, PILOT_PER_CLASS
    )
    val_dataset = select_balanced_subset(
        val_dataset, min(PILOT_PER_CLASS, 1770)
    )

print("Training frames:", len(train_dataset))
print("Validation frames:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ==========================
# MODEL AND LOSS
# ==========================

model = get_model().to(DEVICE)

# Class 0 = FAKE, class 1 = REAL.
# Weight the minority real class more strongly.
fake_count = sum(label == 0 for _, label in train_dataset.samples)
real_count = sum(label == 1 for _, label in train_dataset.samples)

class_weights = torch.tensor(
    [
        1.0,
        fake_count / max(real_count, 1),
    ],
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==========================
# TRAINING / VALIDATION
# ==========================

def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for images, labels in tqdm(
            loader,
            desc="Training" if training else "Validation",
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    balanced_acc = balanced_accuracy_score(
        all_labels, all_predictions
    )
    macro_f1 = f1_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )
    matrix = confusion_matrix(
        all_labels, all_predictions, labels=[0, 1]
    )

    # Rows are actual classes [FAKE, REAL]; columns are predicted.
    fake_recall = matrix[0, 0] / max(matrix[0].sum(), 1)
    real_recall = matrix[1, 1] / max(matrix[1].sum(), 1)

    return {
        "loss": avg_loss,
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "fake_recall": float(fake_recall),
        "real_recall": float(real_recall),
        "confusion_matrix": matrix.tolist(),
    }


# ==========================
# MAIN
# ==========================

def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    history = []
    best_balanced_accuracy = -1.0
    best_macro_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

        train_metrics = run_epoch(train_loader, training=True)
        val_metrics = run_epoch(val_loader, training=False)

        print(
            f"Train loss: {train_metrics['loss']:.4f}"
        )
        print(
            f"Validation loss: {val_metrics['loss']:.4f}"
        )
        print(
            f"Balanced accuracy: "
            f"{val_metrics['balanced_accuracy'] * 100:.2f}%"
        )
        print(
            f"Macro-F1: {val_metrics['macro_f1'] * 100:.2f}%"
        )
        print(
            f"Fake recall: {val_metrics['fake_recall'] * 100:.2f}%"
        )
        print(
            f"Real recall: {val_metrics['real_recall'] * 100:.2f}%"
        )
        print("Confusion matrix [FAKE, REAL]:")
        print(val_metrics["confusion_matrix"])

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        })

        # Prefer balanced accuracy, breaking ties with macro-F1.
        key = (
            val_metrics["balanced_accuracy"],
            val_metrics["macro_f1"],
        )
        best_key = (
            best_balanced_accuracy,
            best_macro_f1,
        )

        if key > best_key:
            best_balanced_accuracy, best_macro_f1 = key
            torch.save(model.state_dict(), CHECKPOINT)
            print("Saved separate experimental checkpoint:", CHECKPOINT)

        METRICS_FILE.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    print("\nExperiment complete.")
    print("Checkpoint:", CHECKPOINT)
    print("Metrics:", METRICS_FILE)
    print("Original best_model.pth was not overwritten.")


if __name__ == "__main__":
    main()

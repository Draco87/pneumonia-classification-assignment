import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST

from models.cnn import PneumoniaCNNRegularized


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------
# Preprocessing and augmentation
# ---------------------------------------------------------

# Conservative augmentation for training only.
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=7),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# Validation images must NOT be randomly augmented.
val_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

train_dataset = PneumoniaMNIST(
    split="train",
    root="./data",
    download=False,
    transform=train_transform
)

val_dataset = PneumoniaMNIST(
    split="val",
    root="./data",
    download=False,
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = PneumoniaCNNRegularized().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------------------------------------------------
# Training history
# ---------------------------------------------------------

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

best_val_loss = float("inf")

os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

for epoch in range(EPOCHS):

    # =========================
    # Training
    # =========================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.float().view(-1, 1).to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_loss = running_loss / total
    train_accuracy = correct / total


    # =========================
    # Validation
    # =========================

    model.eval()

    running_val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = (
                labels.float()
                .view(-1, 1)
                .to(DEVICE)
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            running_val_loss += (
                loss.item() * images.size(0)
            )

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_loss = (
        running_val_loss / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )


    # =========================
    # Save history
    # =========================

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    # =========================
    # Save best checkpoint
    # =========================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "saved_models/best_regularized_model.pth"
        )

        saved = " <-- best model saved"

    else:
        saved = ""


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
        f"{saved}"
    )


# ---------------------------------------------------------
# Save history
# ---------------------------------------------------------

np.savez(
    "results/regularized_history.npz",

    train_loss=np.array(
        history["train_loss"]
    ),

    train_accuracy=np.array(
        history["train_accuracy"]
    ),

    val_loss=np.array(
        history["val_loss"]
    ),

    val_accuracy=np.array(
        history["val_accuracy"]
    )
)


print("\nRegularized training complete.")

print(
    "Best validation loss:",
    f"{best_val_loss:.4f}"
)

print(
    "Model saved to:",
    "saved_models/best_regularized_model.pth"
)
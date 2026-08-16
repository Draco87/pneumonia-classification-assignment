import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from medmnist import PneumoniaMNIST

from models.cnn import PneumoniaCNN


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cpu")
MODEL_PATH = "saved_models/best_baseline_model.pth"

os.makedirs("results", exist_ok=True)


# =========================================================
# PART 1 — TRAINING CURVES
# =========================================================

history = np.load(
    "results/baseline_history.npz"
)

train_loss = history["train_loss"]
val_loss = history["val_loss"]

train_accuracy = history["train_accuracy"]
val_accuracy = history["val_accuracy"]

epochs = np.arange(
    1,
    len(train_loss) + 1
)


# ---------------------------------------------------------
# Loss curve
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline CNN — Training and Validation Loss")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/baseline_loss_curve.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Accuracy curve
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline CNN — Training and Validation Accuracy")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/baseline_accuracy_curve.png",
    dpi=300
)

plt.close()


# =========================================================
# PART 2 — LOAD TEST DATA
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


test_dataset = PneumoniaMNIST(
    split="test",
    root="./data",
    download=False,
    transform=transform
)


# =========================================================
# PART 3 — LOAD BEST BASELINE MODEL
# =========================================================

model = PneumoniaCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


# =========================================================
# PART 4 — COLLECT PREDICTIONS
# =========================================================

records = []

with torch.no_grad():

    for index in range(len(test_dataset)):

        image, label = test_dataset[index]

        true_label = int(label.item())

        input_image = (
            image.unsqueeze(0)
            .to(DEVICE)
        )

        logit = model(input_image)

        probability = torch.sigmoid(
            logit
        ).item()

        prediction = (
            1 if probability >= 0.5 else 0
        )

        records.append({
            "index": index,
            "image": image,
            "true": true_label,
            "predicted": prediction,
            "probability": probability
        })


# =========================================================
# PART 5 — SAMPLE CORRECT PREDICTIONS
# =========================================================

correct_normal = [
    r for r in records
    if r["true"] == 0
    and r["predicted"] == 0
]

correct_pneumonia = [
    r for r in records
    if r["true"] == 1
    and r["predicted"] == 1
]


# Select three examples from each class
samples = (
    correct_normal[:3]
    +
    correct_pneumonia[:3]
)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(9, 6)
)


for ax, record in zip(
    axes.flat,
    samples
):

    image = (
        record["image"]
        .squeeze()
        .numpy()
    )

    # Undo normalization:
    # [-1, 1] -> [0, 1]
    image = (
        image * 0.5
        + 0.5
    )

    true_name = (
        "Pneumonia"
        if record["true"] == 1
        else "Normal"
    )

    probability = record["probability"]

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(
        f"True: {true_name}\n"
        f"P(Pneumonia): {probability:.3f}"
    )

    ax.axis("off")


plt.suptitle(
    "Baseline CNN — Correct Test Predictions"
)

plt.tight_layout()

plt.savefig(
    "results/sample_predictions.png",
    dpi=300
)

plt.close()


# =========================================================
# PART 6 — FAILURE CASES
# =========================================================

false_positives = [
    r for r in records
    if r["true"] == 0
    and r["predicted"] == 1
]

false_negatives = [
    r for r in records
    if r["true"] == 1
    and r["predicted"] == 0
]


print(
    f"False positives found: "
    f"{len(false_positives)}"
)

print(
    f"False negatives found: "
    f"{len(false_negatives)}"
)


# Select up to three of each
failure_samples = (
    false_positives[:3]
    +
    false_negatives[:3]
)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(9, 6)
)


for ax, record in zip(
    axes.flat,
    failure_samples
):

    image = (
        record["image"]
        .squeeze()
        .numpy()
    )

    image = (
        image * 0.5
        + 0.5
    )

    true_name = (
        "Pneumonia"
        if record["true"] == 1
        else "Normal"
    )

    predicted_name = (
        "Pneumonia"
        if record["predicted"] == 1
        else "Normal"
    )

    probability = record["probability"]

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(
        f"True: {true_name}\n"
        f"Pred: {predicted_name}\n"
        f"P(Pneumonia): {probability:.3f}"
    )

    ax.axis("off")


plt.suptitle(
    "Baseline CNN — Failure Cases"
)

plt.tight_layout()

plt.savefig(
    "results/failure_cases.png",
    dpi=300
)

plt.close()


print("\nSaved:")
print("results/baseline_loss_curve.png")
print("results/baseline_accuracy_curve.png")
print("results/sample_predictions.png")
print("results/failure_cases.png")
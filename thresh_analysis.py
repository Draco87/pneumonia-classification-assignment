import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

from models.cnn import (
    PneumoniaCNN,
    PneumoniaCNNRegularized
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cpu")
BATCH_SIZE = 64

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ---------------------------------------------------------
# Helper: get labels and probabilities
# ---------------------------------------------------------

def get_predictions(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_labels = []
    all_probabilities = []

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy().flatten()

            labels = (
                labels.numpy().flatten()
            )

            all_labels.extend(labels)
            all_probabilities.extend(probabilities)

    return (
        np.array(all_labels),
        np.array(all_probabilities)
    )


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

val_dataset = PneumoniaMNIST(
    split="val",
    root="./data",
    download=False,
    transform=transform
)

test_dataset = PneumoniaMNIST(
    split="test",
    root="./data",
    download=False,
    transform=transform
)


# ---------------------------------------------------------
# Load baseline model
# ---------------------------------------------------------

baseline_model = PneumoniaCNN().to(DEVICE)

baseline_model.load_state_dict(
    torch.load(
        "saved_models/best_baseline_model.pth",
        map_location=DEVICE
    )
)


# ---------------------------------------------------------
# Load regularized model
# ---------------------------------------------------------

regularized_model = (
    PneumoniaCNNRegularized().to(DEVICE)
)

regularized_model.load_state_dict(
    torch.load(
        "saved_models/best_regularized_model.pth",
        map_location=DEVICE
    )
)


# =========================================================
# PART 1 — ROC CURVES ON TEST SET
# =========================================================

test_labels, baseline_test_probs = (
    get_predictions(
        baseline_model,
        test_dataset
    )
)

_, regularized_test_probs = (
    get_predictions(
        regularized_model,
        test_dataset
    )
)


baseline_auc = roc_auc_score(
    test_labels,
    baseline_test_probs
)

regularized_auc = roc_auc_score(
    test_labels,
    regularized_test_probs
)


baseline_fpr, baseline_tpr, _ = roc_curve(
    test_labels,
    baseline_test_probs
)

regularized_fpr, regularized_tpr, _ = roc_curve(
    test_labels,
    regularized_test_probs
)


plt.figure(figsize=(7, 6))

plt.plot(
    baseline_fpr,
    baseline_tpr,
    label=f"Baseline CNN (AUC = {baseline_auc:.4f})"
)

plt.plot(
    regularized_fpr,
    regularized_tpr,
    label=f"Regularized CNN (AUC = {regularized_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")

plt.title(
    "ROC Curves — Pneumonia Classification"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/roc_curves.png",
    dpi=300
)

plt.close()


# =========================================================
# PART 2 — THRESHOLD ANALYSIS ON VALIDATION SET
# =========================================================

val_labels, baseline_val_probs = (
    get_predictions(
        baseline_model,
        val_dataset
    )
)


thresholds = np.arange(
    0.1,
    1.0,
    0.1
)


print("\nBaseline CNN — Validation Threshold Analysis")
print("------------------------------------------------------------")
print(
    f"{'Threshold':<12}"
    f"{'Sensitivity':<15}"
    f"{'Specificity':<15}"
    f"{'FP':<8}"
    f"{'FN':<8}"
)


sensitivities = []
specificities = []


for threshold in thresholds:

    predictions = (
        baseline_val_probs >= threshold
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        val_labels,
        predictions
    ).ravel()


    sensitivity = (
        tp / (tp + fn)
    )

    specificity = (
        tn / (tn + fp)
    )


    sensitivities.append(
        sensitivity
    )

    specificities.append(
        specificity
    )


    print(
        f"{threshold:<12.1f}"
        f"{sensitivity:<15.4f}"
        f"{specificity:<15.4f}"
        f"{fp:<8}"
        f"{fn:<8}"
    )


# ---------------------------------------------------------
# Threshold trade-off plot
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    thresholds,
    sensitivities,
    marker="o",
    label="Sensitivity"
)

plt.plot(
    thresholds,
    specificities,
    marker="o",
    label="Specificity"
)

plt.axvline(
    x=0.5,
    linestyle="--",
    label="Current Threshold (0.5)"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.title(
    "Validation Sensitivity–Specificity Trade-off"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/threshold_tradeoff.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\nTest ROC-AUC")
print("----------------------------")
print(
    f"Baseline:    {baseline_auc:.4f}"
)
print(
    f"Regularized: {regularized_auc:.4f}"
)


print("\nSaved:")
print("results/roc_curves.png")
print("results/threshold_tradeoff.png")
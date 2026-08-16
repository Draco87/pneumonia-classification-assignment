import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from models.cnn import PneumoniaCNN


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 64
DEVICE = torch.device("cpu")
MODEL_PATH = "saved_models/best_baseline_model.pth"

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ---------------------------------------------------------
# Test dataset
# ---------------------------------------------------------

test_dataset = PneumoniaMNIST(
    split="test",
    root="./data",
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = PneumoniaCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        labels = labels.float().view(-1, 1).to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).float()

        all_labels.extend(
            labels.cpu().numpy().flatten()
        )

        all_predictions.extend(
            predictions.cpu().numpy().flatten()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().flatten()
        )


# ---------------------------------------------------------
# Convert to arrays
# ---------------------------------------------------------

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)


print("\nTest Results")
print("-------------------------")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


print("\nClassification Report")
print("-------------------------")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Normal",
            "Pneumonia"
        ],
        digits=4
    )
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix")
print("-------------------------")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Pneumonia"
    ]
)

display.plot(
    values_format="d"
)

plt.title(
    "Baseline CNN - Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "results/baseline_confusion_matrix.png",
    dpi=300
)

plt.close()


print(
    "\nSaved:"
    "\nresults/baseline_confusion_matrix.png"
)
import numpy as np
import matplotlib.pyplot as plt

from medmnist import PneumoniaMNIST
from medmnist import INFO


# ---------------------------------------------------------
# 1. Dataset information
# ---------------------------------------------------------

data_flag = "pneumoniamnist"
info = INFO[data_flag]

print("Dataset:", info["description"])
print("Task:", info["task"])
print("Labels:", info["label"])


# ---------------------------------------------------------
# 2. Download and load the official dataset splits
# ---------------------------------------------------------

train_dataset = PneumoniaMNIST(
    split="train",
    download=True,
    root="./data"
)

val_dataset = PneumoniaMNIST(
    split="val",
    download=True,
    root="./data"
)

test_dataset = PneumoniaMNIST(
    split="test",
    download=True,
    root="./data"
)


# ---------------------------------------------------------
# 3. Inspect dataset sizes
# ---------------------------------------------------------

print("\nDataset Splits")
print("--------------------")
print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images:       {len(test_dataset)}")
print(
    f"Total images:      "
    f"{len(train_dataset) + len(val_dataset) + len(test_dataset)}"
)


# ---------------------------------------------------------
# 4. Inspect a sample image
# ---------------------------------------------------------

sample_image, sample_label = train_dataset[0]

print("\nSample")
print("--------------------")
print("Image type:", type(sample_image))
print("Image size:", sample_image.size)
print("Label:", sample_label)


# ---------------------------------------------------------
# 5. Calculate class distribution
# ---------------------------------------------------------

def print_class_distribution(dataset, split_name):
    labels = dataset.labels.flatten()

    classes, counts = np.unique(labels, return_counts=True)

    print(f"\n{split_name} class distribution")
    print("--------------------")

    for class_id, count in zip(classes, counts):
        class_name = info["label"][str(class_id)]
        percentage = 100 * count / len(labels)

        print(
            f"{class_name}: {count} "
            f"({percentage:.2f}%)"
        )


print_class_distribution(train_dataset, "Training")
print_class_distribution(val_dataset, "Validation")
print_class_distribution(test_dataset, "Test")


# ---------------------------------------------------------
# 6. Plot training class distribution
# ---------------------------------------------------------

train_labels = train_dataset.labels.flatten()

classes, counts = np.unique(
    train_labels,
    return_counts=True
)

class_names = [
    info["label"][str(class_id)]
    for class_id in classes
]

plt.figure(figsize=(6, 4))
plt.bar(class_names, counts)

plt.title("PneumoniaMNIST Training Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.savefig(
    "./results/class_distribution.png",
    dpi=300
)
plt.close()


# ---------------------------------------------------------
# 7. Visualize sample images
# ---------------------------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]

    label_id = int(label.item())
    class_name = info["label"][str(label_id)]

    ax.imshow(image, cmap="gray")
    ax.set_title(class_name)
    ax.axis("off")

plt.suptitle("Sample PneumoniaMNIST Images")
plt.tight_layout()

plt.savefig(
    "./results/sample_images.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("results/class_distribution.png")
print("results/sample_images.png")
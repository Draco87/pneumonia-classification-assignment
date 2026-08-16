import argparse

import torch

from PIL import Image
from torchvision import transforms

from models.cnn import PneumoniaCNN


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cpu")

MODEL_PATH = "saved_models/best_baseline_model.pth"


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((28, 28)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

def load_model():

    model = PneumoniaCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# ---------------------------------------------------------
# Predict one image
# ---------------------------------------------------------

def predict_image(image_path, model):

    image = Image.open(image_path)

    image_tensor = transform(image)

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logit = model(image_tensor)

        probability = torch.sigmoid(
            logit
        ).item()


    if probability >= 0.5:

        prediction = "Pneumonia"

    else:

        prediction = "Normal"


    return prediction, probability


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run pneumonia classification "
            "on a chest X-ray image."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the chest X-ray image"
    )

    args = parser.parse_args()


    model = load_model()


    prediction, probability = predict_image(
        args.image,
        model
    )


    print("\nPrediction")
    print("-------------------------")

    print(
        f"Predicted class: {prediction}"
    )

    print(
        "Pneumonia probability: "
        f"{probability:.4f}"
    )

    print(
        "Normal probability: "
        f"{1 - probability:.4f}"
    )
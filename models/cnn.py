import torch.nn as nn


class PneumoniaCNN(nn.Module):
    """
    Lightweight CNN for binary classification of 28x28
    grayscale chest X-ray images.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 1 x 28 x 28 -> 32 x 14 x 14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 32 x 14 x 14 -> 64 x 7 x 7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 64 x 7 x 7 -> 128 x 7 x 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # 128 x 7 x 7 -> 128 x 1 x 1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x

class PneumoniaCNNRegularized(nn.Module):
    """
    Regularized CNN used for Experiment 2.

    Uses the same convolutional feature extractor as the
    baseline, with dropout added before classification.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x
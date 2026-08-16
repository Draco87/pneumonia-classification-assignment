from medmnist import PneumoniaMNIST


dataset = PneumoniaMNIST(
    split="test",
    root="./data",
    download=False
)


image, label = dataset[0]

image.save(
    "results/example_test_xray.png"
)


print(
    "Saved image to "
    "results/example_test_xray.png"
)

print(
    "Ground-truth label:",
    int(label.item())
)

print(
    "0 = Normal, 1 = Pneumonia"
)
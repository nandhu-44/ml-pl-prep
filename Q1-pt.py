import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Define transformations (Normalize pixel values)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to tensor and scales to [0,1]
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1] for better training stability
])

# Download and load MNIST dataset
mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
mnist_testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Split training data into train (80%) and validation (10%)
train_size = int(0.8 * len(mnist_dataset))
val_size = len(mnist_dataset) - train_size
train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(mnist_testset)}")

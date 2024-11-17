from nexus.models.cv import CompactCNN
from nexus.training import Trainer
from nexus.data import Dataset, Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
import torchvision.datasets as datasets
import torch

# Configuration
config = {
    "num_classes": 10,
    "dropout": 0.2,
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 30
}

# Create transforms
transform = Compose([
    Resize(32),  # CIFAR-10 images are 32x32
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

eval_transform = Compose([
    Resize(32),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

eval_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    transform=eval_transform,
    download=True
)

# Create model
model = CompactCNN(config)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer="adam",
    learning_rate=config["learning_rate"]
)

# Train model
trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    batch_size=config["batch_size"],
    num_epochs=config["num_epochs"]
)

# Save the trained model
torch.save(model.state_dict(), "compact_cnn.pth") 
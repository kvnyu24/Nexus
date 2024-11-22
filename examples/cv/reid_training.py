from nexus.models.cv import PedestrianReID
from nexus.training import Trainer
from nexus.data import ImageDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms

# Configuration
config = {
    "hidden_dim": 512,
    "feature_dim": 2048,
    "num_classes": 1000,
    "learning_rate": 3e-4,
    "batch_size": 32,
    "image_size": 256
}

def get_reid_transforms(train: bool = True):
    transforms = [
        transforms.Resize(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    if train:
        transforms.insert(1, transforms.RandomHorizontalFlip())
        
    return transforms.Compose(transforms)

# Initialize model
model = PedestrianReID(config)

# Create datasets and dataloaders
train_dataset = ImageDataset(
    root="path/to/reid/dataset",
    transform=get_reid_transforms(train=True)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=4
)

# Training setup
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=config["learning_rate"]),
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train model
trainer.train(num_epochs=100)

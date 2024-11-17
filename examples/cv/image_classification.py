from nexus.models.cv import VisionTransformer
from nexus.training import Trainer, CosineWarmupScheduler
from nexus.core.config import ConfigManager
from nexus.data import Dataset, Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from nexus.training.losses import FocalLoss
import torchvision.datasets as datasets

# Load configuration
config = ConfigManager.load_config("configs/vit_base.yaml")
# Convert SimpleNamespace to dictionary
config_dict = vars(config)

# Create model
model = VisionTransformer(config_dict)

# Create transforms following ViT paper
train_transform = Compose([
    Resize(config.image_size),  # 224 from config
    RandomCrop(config.image_size, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ViT standard normalization
])

eval_transform = Compose([
    Resize(config.image_size),
    ToTensor(), 
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Download and create ImageNet datasets
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=train_transform,
    download=True
)

eval_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    transform=eval_transform,
    download=True
)

# Update model configuration for CIFAR-10
config_dict['num_classes'] = 10  # CIFAR-10 has 10 classes

# Create trainer with custom scheduler
trainer = Trainer(model=model)
scheduler = CosineWarmupScheduler(
    trainer.optimizer,
    warmup_steps=config.warmup_steps,  # 10000 from config
    max_steps=config.max_steps  # 100000 from config
)

# Train with custom loss
trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_fn=FocalLoss(gamma=2.0),
    scheduler=scheduler,
    batch_size=config.batch_size,  # 32 from config
    num_epochs=100
)

# Plot training metrics
trainer.plot_metrics(['loss', 'accuracy'], save_path='training_metrics.png')
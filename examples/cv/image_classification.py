from nexus.models.cv import VisionTransformer
from nexus.training import Trainer, CosineWarmupScheduler
from nexus.core.config import ConfigManager
from nexus.data import Dataset, Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from nexus.training.losses import FocalLoss
from nexus.utils.gpu import GPUManager, AutoDevice
import torchvision.datasets as datasets

# Initialize GPU manager
gpu_manager = GPUManager()
optimal_device = gpu_manager.get_optimal_device()

# Load configuration
config = ConfigManager.load_config("configs/vit_base.yaml")
config_dict = vars(config)

# Create model and move to optimal GPU
model = VisionTransformer(config_dict)
model = model.to(optimal_device)

# Create transforms following ViT paper
train_transform = Compose([
    Resize(config.image_size),
    RandomCrop(config.image_size, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
config_dict['num_classes'] = 10

# Create trainer with GPU support
trainer = Trainer(
    model=model,
    device=optimal_device  # Pass the optimal device to trainer
)

# Print GPU memory info before training
if gpu_manager.initialized:
    memory_info = gpu_manager.get_gpu_memory_info()
    for gpu in memory_info:
        print(f"GPU {gpu['device']}: {gpu['free']:.2f}MB free / {gpu['total']:.2f}MB total")

# Create scheduler
scheduler = CosineWarmupScheduler(
    trainer.optimizer,
    warmup_steps=config.warmup_steps,
    max_steps=config.max_steps
)

# Train with custom loss
with AutoDevice(model):  # Ensures model is on optimal device
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_fn=FocalLoss(gamma=2.0),
        scheduler=scheduler,
        batch_size=config.batch_size,
        num_epochs=100
    )

# Plot training metrics
trainer.plot_metrics(['loss', 'accuracy'], save_path='training_metrics.png')
from nexus.models.cv import VisionTransformer
from nexus.training import Trainer, CosineWarmupScheduler
from nexus.core.config import ConfigManager
from nexus.data import Dataset, Compose, Resize, ToTensor, Normalize
from nexus.training.losses import FocalLoss

# Load configuration
config = ConfigManager.load_config("configs/vit_base.yaml")

# Create model
model = VisionTransformer(config)

# Create datasets
transform = Compose([
    Resize(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = Dataset(
    data_dir="data/train",
    transform=transform
)

eval_dataset = Dataset(
    data_dir="data/val", 
    transform=transform
)

# Create trainer with custom scheduler
trainer = Trainer(model=model)
scheduler = CosineWarmupScheduler(
    trainer.optimizer,
    warmup_steps=1000,
    max_steps=50000
)

# Train with custom loss
trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_fn=FocalLoss(gamma=2.0),
    scheduler=scheduler,
    batch_size=32,
    num_epochs=100
)
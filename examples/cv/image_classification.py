from nexus.models.cv import VisionTransformer
from nexus.training import Trainer, CosineWarmupScheduler
from nexus.core.config import ConfigManager

# Load configuration
config = ConfigManager.load_config("configs/vit_base.yaml")

# Create model
model = VisionTransformer(config)

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
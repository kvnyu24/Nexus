import argparse
from nexus.models.cv import VisionTransformer
from nexus.training import Trainer, CosineWarmupScheduler
from nexus.core.config import ConfigManager
from nexus.data import Dataset, Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from nexus.training.losses import FocalLoss
from nexus.utils.gpu import GPUManager, AutoDevice
import torchvision.datasets as datasets
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR-10')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cpu', 'cuda', 'mps'],
                      help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Override batch size from config')
    parser.add_argument('--learning_rate', type=float, default=None,
                      help='Override learning rate from config')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--config', type=str, default="configs/vit_base.yaml",
                      help='Path to model configuration file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize device handling
    if args.device == 'auto':
        gpu_manager = GPUManager()
        device = gpu_manager.get_optimal_device()
    else:
        device = torch.device(args.device)
    
    # Load configuration
    config = ConfigManager.load_config(args.config)
    config_dict = vars(config)
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Create model and handle device-specific settings
    model = VisionTransformer(config_dict)
    
    # Device-specific adjustments
    if device.type == 'mps':
        model = model.to(torch.float32)
        config.batch_size = min(config.batch_size, 64)
        config.learning_rate *= 0.1
    
    # Ensure model parameters are float32 before training
    model = model.to(torch.float32)
    
    # Move model to device after setting dtype
    model = model.to(device)
    
    # Create transforms
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
    
    # Create datasets
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
    
    # Create trainer with the model after dtype and device setup
    trainer = Trainer(
        model=model,
        device=device,
        optimizer="adam",
        learning_rate=config.learning_rate * (1.0 if device.type != 'mps' else 0.1)
    )
    
    # Log training setup
    trainer.logger.info(f"Training on device: {device}")
    trainer.logger.info(f"Batch size: {config.batch_size}")
    trainer.logger.info(f"Learning rate: {config.learning_rate}")
    
    # Print GPU memory info if available
    if isinstance(device, torch.device) and device.type == 'cuda':
        memory_info = GPUManager().get_gpu_memory_info()
        for gpu in memory_info:
            print(f"GPU {gpu['device']}: {gpu['free']:.2f}MB free / {gpu['total']:.2f}MB total")
    
    # Create scheduler
    scheduler = CosineWarmupScheduler(
        trainer.optimizer,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps
    )
    
    # Create a custom collate function to ensure proper tensor types
    def collate_fn(batch):
        # Ensure images are float32 and on the correct device
        images = torch.stack([item[0] for item in batch]).to(torch.float32)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return {'image': images, 'labels': labels}
    
    # Train with custom loss
    with AutoDevice(model):
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss_fn=FocalLoss(gamma=2.0),
            scheduler=scheduler,
            batch_size=config.batch_size,
            num_epochs=args.num_epochs,
            collate_fn=collate_fn
        )
    
    # Save the trained model
    torch.save(model.state_dict(), "vit_cifar10.pth")

if __name__ == "__main__":
    main()
from nexus.models.cv.vae import EnhancedVAE
from nexus.training import Trainer
from nexus.data import Dataset, DataLoader, Compose, Resize, ToTensor, Normalize
import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from typing import Dict

# Configure VAE model
config = {
    "input_dim": 784,  # 28x28 for MNIST
    "hidden_dim": 400,
    "latent_dim": 20,
    "beta": 1.0,  # Beta-VAE parameter
    "architecture": "mlp"
}

# Initialize model
vae = EnhancedVAE(config)

# Create transforms using Nexus data transforms
transform = Compose([
    Resize(28),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

# Load MNIST dataset using torchvision but with Nexus transforms
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
val_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

# Use Nexus DataLoader instead of torch's DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Create custom trainer for VAE
class VAETrainer(Trainer):
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()
        
        # Extract batch data
        x, _ = batch
        x = x.to(self.device)
        x_flat = x.view(x.size(0), -1)
        
        # Forward pass
        outputs = self.model(x_flat)
        reconstructed = outputs["reconstruction"]
        mu = outputs["mu"]
        log_var = outputs["log_var"]
        
        # Compute losses
        reconstruction_loss = F.mse_loss(reconstructed, x_flat)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch
        
        # Total loss (Beta-VAE formulation)
        total_loss = reconstruction_loss + self.model.beta * kl_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kl_loss": kl_loss.item()
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            x, _ = batch
            x = x.to(self.device)
            x_flat = x.view(x.size(0), -1)
            
            outputs = self.model(x_flat)
            reconstructed = outputs["reconstruction"]
            
            val_loss = F.mse_loss(reconstructed, x_flat)
            
            return {"val_loss": val_loss.item()}

# Initialize trainer with checkpointing
trainer = VAETrainer(
    model=vae,
    optimizer="adam",
    learning_rate=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="checkpoints/vae"
)

# Train model
trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    train_loader=train_loader,
    eval_loader=val_loader,
    num_epochs=50,
    eval_frequency=5
)

# Generate samples after training
with torch.no_grad():
    # Sample from latent space
    z = torch.randn(16, config["latent_dim"]).to(trainer.device)
    samples = vae.decoder(z)
    samples = samples.view(-1, 1, 28, 28)  # Reshape for MNIST
    
    # Save generated samples
    torchvision.utils.save_image(
        samples,
        "vae_samples.png",
        nrow=4,
        normalize=True
    )
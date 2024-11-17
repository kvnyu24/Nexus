from nexus.models.nlp import ReasoningLLM, ChainOfThoughtModule
from nexus.training import Trainer
from nexus.data import TextProcessor
from nexus.utils.metrics import MetricsCalculator
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import torch.nn.functional as F
from nexus.data.text_processor import TextProcessor, ReasoningDataset


# Model configuration
config = {
    "vocab_size": 50000,
    "hidden_size": 768,
    "num_heads": 12,
    "num_reasoning_steps": 3,
    "max_seq_length": 512,
    "dropout": 0.1
}

# Add after config and before model initialization
step_weights = [1.0, 1.0, 1.0]  # Equal weighting for each reasoning step

def calculate_step_loss(step_output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate loss for a single reasoning step"""
    return F.cross_entropy(
        step_output.view(-1, config["vocab_size"]),
        labels.view(-1),
        ignore_index=-100  # Standard padding index
    )

# Initialize text processor and datasets
processor = TextProcessor(
    vocab_size=config["vocab_size"],
    max_length=config["max_seq_length"]
)

# Sample data for demonstration
sample_texts = [
    "Example text 1",
    "Example text 2",
    "Example text 3"
]

train_dataset = ReasoningDataset(sample_texts, processor)
eval_dataset = ReasoningDataset(sample_texts[:1], processor)  # Smaller eval set

# Initialize model with chain of thoughts
model = ReasoningLLM(config)
chain_of_thought = ChainOfThoughtModule(config)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer="adam",
    learning_rate=1e-4
)

# Training loop with reasoning visualization
def train_step(batch):
    # Get model outputs with reasoning steps
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"]
    )
    
    # Extract reasoning steps and attention maps
    reasoning_steps = outputs["reasoning_steps"]
    attention_maps = outputs["attention_maps"]
    
    # Calculate loss for each reasoning step
    step_losses = []
    for step_idx, step_output in enumerate(reasoning_steps):
        step_loss = calculate_step_loss(step_output, batch["labels"])
        step_losses.append(step_loss)
    
    # Combine losses with optional weighting
    total_loss = sum(w * l for w, l in zip(step_weights, step_losses))
    
    return {
        "loss": total_loss,
        "step_losses": step_losses,
        "attention_maps": attention_maps
    }

# Train model
trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    batch_size=16,
    num_epochs=10
) 
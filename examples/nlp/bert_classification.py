from nexus.models.nlp import BERTModel
from nexus.components.attention import MultiHeadAttention
from nexus.training import Trainer

# Configure and create model
config = {
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "vocab_size": 30000
}

model = BERTModel(config)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer="adam",
    learning_rate=1e-4,
    device="cuda"
)

# Train model
trainer.train(
    train_dataset=train_data,
    eval_dataset=eval_data,
    batch_size=32,
    num_epochs=10
) 
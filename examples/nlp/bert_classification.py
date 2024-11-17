from nexus.models.nlp import BERTModel
from nexus.components.attention import MultiHeadAttention
from nexus.training import Trainer
from nexus.utils import ExperimentManager
from nexus.data.datasets import BERTDataset
from nexus.data.tokenizer import BERTTokenizer
import torch

# Sample data for demonstration
train_texts = [
    "This is a positive example",
    "This is a negative example",
    "Another positive case",
    "Another negative instance"
]
train_labels = torch.tensor([1, 0, 1, 0])  # Binary classification

eval_texts = [
    "A positive test case",
    "A negative test case"
]
eval_labels = torch.tensor([1, 0])

# Initialize our BERT tokenizer and fit on training data
tokenizer = BERTTokenizer(vocab_size=30000)
tokenizer.fit(train_texts)  # Add this line to build vocabulary

# Configure and create model
config = {
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "vocab_size": tokenizer.vocab_size,  # Use tokenizer's vocab size
    "num_classes": 2,  # Binary classification
    "max_seq_length": 512
}

model = BERTModel(config)

# Create datasets
train_dataset = BERTDataset(
    texts=train_texts,
    labels=train_labels,
    tokenizer=tokenizer,
    max_length=512
)

eval_dataset = BERTDataset(
    texts=eval_texts,
    labels=eval_labels,
    tokenizer=tokenizer,
    max_length=512
)

# Create trainer with loss function
trainer = Trainer(
    model=model,
    optimizer="adam",
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train model
metrics = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    batch_size=2,  # Small batch size for demo
    num_epochs=3
)

print(f"Training completed! Final metrics: {metrics}") 
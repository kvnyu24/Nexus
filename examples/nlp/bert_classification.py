from nexus.models.nlp import BERTModel
from nexus.components.attention import MultiHeadAttention
from nexus.training import Trainer
from nexus.utils import ExperimentManager
from nexus.data.datasets import BERTDataset
from transformers import AutoTokenizer
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Configure and create model
config = {
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "vocab_size": 30000
}

model = BERTModel(config)

# Create datasets
train_dataset = BERTDataset(
    texts=train_texts,  # Your training texts
    labels=train_labels,  # Your training labels
    tokenizer=tokenizer,
    max_length=512
)

eval_dataset = BERTDataset(
    texts=eval_texts,  # Your evaluation texts
    labels=eval_labels,  # Your evaluation labels
    tokenizer=tokenizer,
    max_length=512
)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer="adam",
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train model
trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    batch_size=32,
    num_epochs=10
) 
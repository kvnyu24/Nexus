from nexus.models.nlp import TransformerClassifier
from nexus.data import TextProcessor, TextDataset
from nexus.data.tokenizer import SimpleTokenizer
from nexus.training import Trainer
from nexus.utils.metrics import MetricsCalculator
import torch

# Initialize our custom tokenizer
tokenizer = SimpleTokenizer(vocab_size=5000, min_freq=1)

# Sample data for demonstration
train_texts = [
    "This movie was great!",
    "Terrible waste of time",
    "Pretty average film",
    "Absolutely loved it",
    "Not worth watching"
]

train_labels = torch.tensor([1, 0, 1, 1, 0])  # 1 for positive, 0 for negative

eval_texts = [
    "Really enjoyed this one",
    "Disappointing experience"
]

eval_labels = torch.tensor([1, 0])

# Fit tokenizer on training data
tokenizer.fit(train_texts)

# Initialize components
processor = TextProcessor(tokenizer=tokenizer)
model = TransformerClassifier(
    num_classes=2,
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_layers=4,
    num_heads=8
)

trainer = Trainer(
    model=model,
    optimizer="adam",
    learning_rate=2e-5
)

# Create datasets
train_dataset = TextDataset(train_texts, train_labels, processor)
eval_dataset = TextDataset(eval_texts, eval_labels, processor)

# Train and evaluate
metrics = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    batch_size=2,
    num_epochs=3
)

# Print final metrics
print("Training completed!")
print(f"Final metrics: {metrics}")

# Test the tokenizer
sample_text = "This is a great movie!"
encoded = tokenizer.encode(sample_text, max_length=16)
decoded = tokenizer.decode(encoded)
print(f"\nSample tokenization:")
print(f"Original: {sample_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}") 
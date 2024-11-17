from nexus.models.nlp import TransformerClassifier
from nexus.data import TextProcessor, TextDataset
from nexus.training import Trainer
from nexus.utils.metrics import MetricsCalculator

# Initialize components
processor = TextProcessor(tokenizer=your_tokenizer)
model = TransformerClassifier(num_classes=3)
trainer = Trainer(model=model)

# Create datasets
train_dataset = TextDataset(train_texts, train_labels, processor)
eval_dataset = TextDataset(eval_texts, eval_labels, processor)

# Train and evaluate
metrics = trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    batch_size=32,
    num_epochs=5
) 
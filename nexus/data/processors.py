from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

class TextProcessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], processor: TextProcessor):
        self.texts = texts
        self.labels = labels
        self.processor = processor
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        processed = self.processor.process_text(text)
        processed["label"] = torch.tensor(label)
        
        return processed 
from typing import List, Dict, Any, Optional, Union
import torch
from torch.utils.data import Dataset
from nexus.utils.logging import Logger

class TextProcessor:
    """Processor for text data that handles tokenization and encoding."""
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True
    ):
        """Initialize text processor.
        
        Args:
            tokenizer: Tokenizer instance for text encoding
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length' or 'longest')
            truncation: Whether to truncate sequences longer than max_length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.logger = Logger(self.__class__.__name__)
        
    def process_text(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Process text input into model-ready format.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Dict containing input_ids and attention_mask tensors
        """
        if not text:
            raise ValueError("Empty text input provided")
            
        try:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            raise

class TextDataset(Dataset):
    """Dataset for text classification tasks."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        processor: TextProcessor,
        label_map: Optional[Dict[int, str]] = None
    ):
        """Initialize dataset.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            processor: TextProcessor instance
            label_map: Optional mapping from label ids to names
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
            
        self.texts = texts
        self.labels = labels
        self.processor = processor
        self.label_map = label_map
        self.logger = Logger(self.__class__.__name__)
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get processed item by index.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Dict containing processed text tensors and label
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range")
            
        try:
            text = self.texts[idx]
            label = self.labels[idx]
            
            processed = self.processor.process_text(text)
            processed["label"] = torch.tensor(label, dtype=torch.long)
            
            if self.label_map:
                processed["label_name"] = self.label_map[label]
                
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing item {idx}: {str(e)}")
            raise
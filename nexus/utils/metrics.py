from typing import Dict, List
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MetricsCalculator:
    @staticmethod
    def calculate_classification_metrics(
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        # Convert to numpy arrays
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
            
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average='weighted'
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    @staticmethod
    def calculate_regression_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        mse = torch.mean((predictions - targets) ** 2).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse)
        } 
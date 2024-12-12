from typing import Dict, List, Union, Optional
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix,
    r2_score, explained_variance_score
)

class MetricsCalculator:
    @staticmethod
    def calculate_classification_metrics(
        predictions: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        # Input validation
        if not isinstance(predictions, (torch.Tensor, np.ndarray)):
            raise TypeError("Predictions must be torch.Tensor or np.ndarray")
        if not isinstance(labels, (torch.Tensor, np.ndarray)):
            raise TypeError("Labels must be torch.Tensor or np.ndarray")
            
        # Convert to numpy arrays
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        if probabilities is not None and torch.is_tensor(probabilities):
            probabilities = probabilities.detach().cpu().numpy()
            
        # Ensure 1D arrays
        predictions = predictions.ravel()
        labels = labels.ravel()
        
        # Calculate basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            average=average,
            zero_division=0
        )
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics["true_positives"] = float(np.diag(cm).sum())
        metrics["false_positives"] = float(cm.sum(axis=0).sum() - np.diag(cm).sum())
        metrics["false_negatives"] = float(cm.sum(axis=1).sum() - np.diag(cm).sum())
        
        # Calculate AUC if probabilities provided
        if probabilities is not None:
            try:
                auc = roc_auc_score(labels, probabilities, multi_class='ovr')
                metrics["auc"] = float(auc)
            except ValueError:
                pass  # Skip AUC for invalid cases
                
        return metrics
        
    @staticmethod
    def calculate_regression_metrics(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        sample_weights: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, float]:
        # Input validation and conversion
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        if sample_weights is not None and torch.is_tensor(sample_weights):
            sample_weights = sample_weights.detach().cpu().numpy()
            
        # Ensure arrays are flattened
        predictions = predictions.ravel()
        targets = targets.ravel()
        
        # Calculate basic metrics
        mse = np.average((predictions - targets) ** 2, weights=sample_weights)
        mae = np.average(np.abs(predictions - targets), weights=sample_weights)
        rmse = np.sqrt(mse)
        
        # Calculate additional metrics
        r2 = r2_score(targets, predictions, sample_weight=sample_weights)
        explained_var = explained_variance_score(targets, predictions, 
                                              sample_weight=sample_weights)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "explained_variance": float(explained_var),
            "mape": float(mape)
        }
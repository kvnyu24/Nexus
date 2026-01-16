import torch
from typing import Any, Dict, Union, Tuple, List


class BatchProcessor:
    """Utility class for processing batches in training pipelines."""

    @staticmethod
    def to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
        """Move batch tensors to the specified device.

        Args:
            batch: Dictionary containing batch data, where values may be tensors
                   or other types.
            device: Target device (e.g., 'cuda', 'cpu').

        Returns:
            Dictionary with tensors moved to the specified device.
        """
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @staticmethod
    def normalize_batch(
        batch: Union[Dict[str, Any], Tuple, List],
        keys: Tuple[str, str] = ('image', 'label')
    ) -> Dict[str, Any]:
        """Normalize batch to dictionary format.

        Handles both tuple/list style batches (e.g., from standard PyTorch datasets)
        and dictionary style batches.

        Args:
            batch: Input batch, either as a tuple/list (images, labels) or a dictionary.
            keys: Tuple of (image_key, label_key) to use when converting tuple batches.
                  Defaults to ('image', 'label').

        Returns:
            Dictionary-style batch with the specified keys.
        """
        if isinstance(batch, (list, tuple)):
            images, labels = batch
            return {keys[0]: images, keys[1]: labels}
        return batch

    @staticmethod
    def prepare_batch(
        batch: Union[Dict[str, Any], Tuple, List],
        device: str,
        keys: Tuple[str, str] = ('image', 'label')
    ) -> Dict[str, Any]:
        """Normalize batch and move to device in one step.

        Convenience method that combines normalize_batch and to_device.

        Args:
            batch: Input batch, either as a tuple/list or dictionary.
            device: Target device (e.g., 'cuda', 'cpu').
            keys: Tuple of (image_key, label_key) to use when converting tuple batches.

        Returns:
            Normalized dictionary batch with tensors on the specified device.
        """
        normalized = BatchProcessor.normalize_batch(batch, keys)
        return BatchProcessor.to_device(normalized, device)

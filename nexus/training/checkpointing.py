from typing import Dict, Any, Optional
import os
import torch
import json
from pathlib import Path
import time

class CheckpointMixin:
    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        keep_last_n: int = 5
    ) -> str:
        """Save a training checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_{timestamp}.pt")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics or {}
        }
        
        # Save model config if available
        if hasattr(self.model, 'config'):
            checkpoint['model_config'] = self.model.config
            
        # Save scheduler state if available
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata separately for easy reading
        metadata = {
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics or {}
        }
        metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Cleanup old checkpoints if needed
        if keep_last_n > 0:
            self._cleanup_old_checkpoints(save_dir, keep_last_n)
            
        return checkpoint_path
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a training checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler if available
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint
        
    def _cleanup_old_checkpoints(self, save_dir: str, keep_last_n: int):
        """Remove old checkpoints, keeping only the last n."""
        checkpoints = []
        for f in os.listdir(save_dir):
            if f.startswith('checkpoint_') and f.endswith('.pt'):
                path = os.path.join(save_dir, f)
                timestamp = os.path.getmtime(path)
                checkpoints.append((path, timestamp))
                
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old checkpoints
        for path, _ in checkpoints[keep_last_n:]:
            os.remove(path)
            # Remove corresponding metadata file
            metadata_path = path.replace('.pt', '_metadata.json')
            if os.path.exists(metadata_path):
                os.remove(metadata_path) 
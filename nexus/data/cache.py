import torch
from typing import Dict, Any, Optional
import os
import hashlib
import pickle
from pathlib import Path

class DataCache:
    def __init__(
        self,
        cache_dir: str = ".cache/nexus",
        max_cache_size_gb: float = 10.0
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        
    def _get_cache_key(self, data: Any) -> str:
        """Generate a unique cache key for the data."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        return hashlib.md5(pickle.dumps(data)).hexdigest()
        
    def _get_cache_size(self) -> int:
        """Get current cache size in bytes."""
        total_size = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size
        
    def _cleanup_cache(self):
        """Remove oldest files if cache exceeds max size."""
        while self._get_cache_size() > self.max_cache_size:
            # Get oldest file
            files = [(f, f.stat().st_mtime) for f in self.cache_dir.glob("*")]
            if not files:
                break
            oldest_file = min(files, key=lambda x: x[1])[0]
            oldest_file.unlink()
            
    def save(self, key: str, data: Any):
        cache_path = self.cache_dir / f"{key}.pt"
        torch.save(data, cache_path)
        self._cleanup_cache()
        
    def load(self, key: str) -> Optional[Any]:
        cache_path = self.cache_dir / f"{key}.pt"
        if cache_path.exists():
            return torch.load(cache_path)
        return None 
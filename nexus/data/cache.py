import torch
from typing import Dict, Any, Optional, Union
import os
import hashlib
import pickle
from pathlib import Path
from nexus.utils.logging import Logger
import shutil
import numpy as np
from datetime import datetime

class DataCache:
    def __init__(
        self,
        cache_dir: str = ".cache/nexus",
        max_cache_size_gb: float = 10.0,
        cleanup_threshold: float = 0.9  # Cleanup when 90% full
    ):
        """Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cached files
            max_cache_size_gb: Maximum cache size in gigabytes
            cleanup_threshold: Fraction of max size that triggers cleanup
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = int(max_cache_size_gb * 1024 * 1024 * 1024)  # Convert to bytes
        self.cleanup_threshold = max(0.0, min(1.0, cleanup_threshold))
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for cache operations."""
        self.logger = Logger(self.__class__.__name__)

    def _get_cache_key(self, data: Any) -> str:
        """Generate a unique cache key for the data."""
        try:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                data = data.copy()
            return hashlib.sha256(pickle.dumps(data, protocol=4)).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to generate cache key: {str(e)}")
            raise

    def _get_cache_size(self) -> int:
        """Get current cache size in bytes."""
        try:
            return sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())
        except Exception as e:
            self.logger.error(f"Failed to calculate cache size: {str(e)}")
            return 0

    def _cleanup_cache(self):
        """Remove oldest files if cache exceeds threshold."""
        current_size = self._get_cache_size()
        threshold_size = int(self.max_cache_size * self.cleanup_threshold)
        
        if current_size <= threshold_size:
            return

        try:
            files = []
            for f in self.cache_dir.glob("*.pt"):
                try:
                    mtime = f.stat().st_mtime
                    size = f.stat().st_size
                    files.append((f, mtime, size))
                except OSError:
                    continue

            files.sort(key=lambda x: x[1])  # Sort by modification time
            
            freed_space = 0
            target_reduction = current_size - threshold_size
            
            for file_path, _, size in files:
                try:
                    file_path.unlink()
                    freed_space += size
                    self.logger.debug(f"Removed cached file: {file_path}")
                    if freed_space >= target_reduction:
                        break
                except OSError as e:
                    self.logger.warning(f"Failed to remove {file_path}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {str(e)}")

    def save(self, key: str, data: Any) -> bool:
        """Save data to cache with given key."""
        try:
            if not isinstance(key, str):
                raise ValueError("Cache key must be a string")
                
            cache_path = self.cache_dir / f"{key}.pt"
            tmp_path = cache_path.with_suffix('.tmp')
            
            # Save to temporary file first
            torch.save(data, tmp_path)
            tmp_path.rename(cache_path)
            
            self._cleanup_cache()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save to cache: {str(e)}")
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    def load(self, key: str) -> Optional[Any]:
        """Load data from cache by key."""
        try:
            if not isinstance(key, str):
                raise ValueError("Cache key must be a string")
                
            cache_path = self.cache_dir / f"{key}.pt"
            if not cache_path.exists():
                return None
                
            return torch.load(cache_path)
            
        except Exception as e:
            self.logger.error(f"Failed to load from cache: {str(e)}")
            return None
import json
import yaml
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import torch
import shutil
from nexus.utils.logging import Logger

class ExperimentManager:
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments",
        config: Optional[Dict[str, Any]] = None,
        save_frequency: int = 100,
        backup_enabled: bool = True,
        max_backups: int = 3
    ):
        self.logger = Logger("ExperimentManager")
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(base_dir) / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.backup_enabled = backup_enabled
        self.max_backups = max_backups
        
        # Save config if provided
        if config:
            self.save_config(config)
            
        # Initialize metrics tracking
        self.metrics_history = []
        self.best_metrics = {}
        
        # Create subdirectories
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.artifacts_dir = self.experiment_dir / "artifacts"
        self.backups_dir = self.experiment_dir / "backups"
        
        for directory in [self.checkpoints_dir, self.artifacts_dir, self.backups_dir]:
            directory.mkdir(exist_ok=True)
            
        self.logger.info(f"Initialized experiment at {self.experiment_dir}")
        
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration with error handling and backup"""
        config_path = self.experiment_dir / "config.yaml"
        try:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            if self.backup_enabled:
                shutil.copy(config_path, self.backups_dir / f"config_{self.timestamp}.yaml")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int, save: bool = True):
        """Log metrics with validation and optional best metric tracking"""
        try:
            metrics = {k: float(v) for k, v in metrics.items()}  # Validate numeric values
            metrics["step"] = step
            metrics["timestamp"] = datetime.now().isoformat()
            self.metrics_history.append(metrics)
            
            # Track best metrics
            for key, value in metrics.items():
                if key not in ["step", "timestamp"]:
                    if key not in self.best_metrics or value > self.best_metrics[key]["value"]:
                        self.best_metrics[key] = {"value": value, "step": step}
            
            # Save metrics periodically
            if save and len(self.metrics_history) % self.save_frequency == 0:
                self.save_metrics()
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
            
    def save_metrics(self):
        """Save metrics with error handling and backup"""
        metrics_path = self.experiment_dir / "metrics.json"
        try:
            with open(metrics_path, "w") as f:
                json.dump({
                    "history": self.metrics_history,
                    "best": self.best_metrics
                }, f, indent=2)
                
            if self.backup_enabled:
                self._manage_backups("metrics.json")
                
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
            
    def save_artifact(self, artifact: Any, name: str, use_torch: bool = True):
        """Save artifacts with format detection and error handling"""
        try:
            artifact_path = self.artifacts_dir / name
            
            if use_torch and torch.is_tensor(artifact) or isinstance(artifact, torch.nn.Module):
                torch.save(artifact, artifact_path.with_suffix('.pt'))
            elif isinstance(artifact, (dict, list)):
                if name.endswith(('.yml', '.yaml')):
                    with open(artifact_path, 'w') as f:
                        yaml.dump(artifact, f)
                else:
                    with open(artifact_path.with_suffix('.json'), 'w') as f:
                        json.dump(artifact, f, indent=2)
            else:
                with open(artifact_path, 'wb') as f:
                    torch.save(artifact, f)
                    
            self.logger.info(f"Saved artifact: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save artifact {name}: {e}")
            
    def _manage_backups(self, filename: str):
        """Manage rotating backups of files"""
        source = self.experiment_dir / filename
        if not source.exists():
            return
            
        backup_path = self.backups_dir / f"{filename}_{self.timestamp}"
        shutil.copy(source, backup_path)
        
        # Remove old backups if exceeding max_backups
        existing_backups = sorted(self.backups_dir.glob(f"{filename}_*"))
        while len(existing_backups) > self.max_backups:
            oldest_backup = existing_backups.pop(0)
            oldest_backup.unlink()
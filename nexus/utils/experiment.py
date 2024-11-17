import json
import yaml
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import torch

class ExperimentManager:
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments",
        config: Optional[Dict[str, Any]] = None
    ):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(base_dir) / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config if provided
        if config:
            self.save_config(config)
            
        # Initialize metrics tracking
        self.metrics_history = []
        
    def save_config(self, config: Dict[str, Any]):
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
            
    def log_metrics(self, metrics: Dict[str, float], step: int):
        metrics["step"] = step
        self.metrics_history.append(metrics)
        
        # Save metrics periodically
        if len(self.metrics_history) % 100 == 0:
            self.save_metrics()
            
    def save_metrics(self):
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f)
            
    def save_artifact(self, artifact: Any, name: str):
        artifact_path = self.experiment_dir / name
        torch.save(artifact, artifact_path) 
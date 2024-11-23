from typing import Dict, Any
from pathlib import Path
import yaml

class ExperimentGenerator:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        
    def generate_experiment(
        self,
        paper_data: Dict[str, Any],
        implementation: Dict[str, Any]
    ) -> None:
        """Generate experiment files and configuration"""
        # Create experiment directory
        exp_dir = self._create_experiment_dir(paper_data)
        
        # Generate config
        config = self._generate_config(paper_data, implementation)
        
        # Save config
        self._save_config(exp_dir, config)
        
        # Generate training script
        train_script = self._generate_training_script(config)
        self._save_training_script(exp_dir, train_script)
        
    def _create_experiment_dir(self, paper_data: Dict[str, Any]) -> Path:
        """Create experiment directory structure"""
        # Reference ExperimentManager pattern 
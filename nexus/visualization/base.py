from typing import Dict, Any
from pathlib import Path
from ..core import NexusModule

class BaseVisualizer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core settings
        self.output_dir = Path(config.get("output_dir", "visualizations"))
        self.dpi = config.get("dpi", 300)
        self.backend = config.get("backend", "matplotlib")
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate visualizer configuration"""
        required = ["output_dir"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def save_figure(self, fig, filename: str) -> None:
        """Save figure with proper directory handling"""
        save_path = self.output_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight') 
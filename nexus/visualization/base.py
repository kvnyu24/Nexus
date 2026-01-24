from typing import Dict, Any
from pathlib import Path
from ..core import NexusModule
from ..core.mixins import ConfigValidatorMixin


class BaseVisualizer(ConfigValidatorMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Validate config using ConfigValidatorMixin
        self.validate_config(config, required_keys=["output_dir"])

        # Core settings
        self.output_dir = Path(config.get("output_dir", "visualizations"))
        self.dpi = config.get("dpi", 300)
        self.backend = config.get("backend", "matplotlib")
                
    def save_figure(self, fig, filename: str) -> None:
        """Save figure with proper directory handling"""
        save_path = self.output_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight') 
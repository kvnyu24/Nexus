from typing import Dict, Any
from ....core.base import NexusModule

class SFTConfigValidator:
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate SFT configuration following FasterRCNN pattern"""
        required_keys = [
            "hidden_size",
            "vocab_size",
            "num_heads",
            "max_sequence_length"
        ]
        
        # Reference FasterRCNN validation pattern
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required SFT configuration key: {key}")
                
        # Validate dimensions
        if config["hidden_size"] % config["num_heads"] != 0:
            raise ValueError("hidden_size must be divisible by num_heads") 
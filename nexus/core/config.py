from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import yaml
import json


@dataclass
class ModelConfig:
    model_type: str
    hidden_size: int
    num_layers: int
    dropout: float
    activation: str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "activation": self.activation
        }

class ConfigManager:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            with open(save_path, 'w') as f:
                yaml.dump(config, f)
        elif save_path.endswith('.json'):
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {save_path}") 
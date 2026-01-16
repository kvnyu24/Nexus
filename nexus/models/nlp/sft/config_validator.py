from typing import Dict, Any, Optional, Union, List
from ....core.base import NexusModule
from nexus.utils.logging import Logger

class SFTConfigValidator:
    _logger = Logger("SFTConfigValidator")

    # Default ranges for common parameters
    PARAM_RANGES = {
        "hidden_size": (64, 8192),
        "vocab_size": (1000, 1000000),
        "num_heads": (1, 128),
        "max_sequence_length": (8, 32768),
        "dropout": (0.0, 1.0),
        "num_layers": (1, 48),
        "intermediate_size": (64, 32768),
        "initializer_range": (0.0001, 0.1),
        "layer_norm_eps": (1e-12, 1e-3),
        "attention_probs_dropout": (0.0, 0.5),
        "hidden_dropout": (0.0, 0.5)
    }

    @classmethod
    def validate_config(cls, config: Dict[str, Any], strict: bool = True) -> None:
        """
        Validate SFT configuration with comprehensive checks for model parameters.
        
        Args:
            config: Dictionary containing model configuration parameters
            strict: If True, enforces all validations. If False, allows some flexibility
            
        Raises:
            ValueError: If any validation checks fail
            TypeError: If parameter types are incorrect
        """
        # Required configuration keys with type validation
        required_keys = {
            "hidden_size": int,
            "vocab_size": int, 
            "num_heads": int,
            "max_sequence_length": int,
            "dropout": float,
            "num_layers": int
        }
        
        # Optional keys with type validation
        optional_keys = {
            "intermediate_size": int,
            "initializer_range": float,
            "layer_norm_eps": float,
            "attention_probs_dropout": float,
            "hidden_dropout": float,
            "gradient_checkpointing": bool,
            "use_cache": bool,
            "pad_token_id": int,
            "bos_token_id": int,
            "eos_token_id": int,
            "tie_word_embeddings": bool,
            "architectures": List[str]
        }
        
        # Check required keys exist with correct types
        for key, expected_type in required_keys.items():
            if key not in config:
                raise ValueError(f"Missing required SFT configuration key: {key}")
            if not isinstance(config[key], expected_type):
                raise TypeError(f"{key} must be of type {expected_type.__name__}, got {type(config[key]).__name__}")
        
        # Validate optional keys if present
        for key, expected_type in optional_keys.items():
            if key in config and not isinstance(config[key], expected_type):
                raise TypeError(f"{key} must be of type {expected_type.__name__}, got {type(config[key]).__name__}")
                
        # Validate value ranges and relationships
        cls._validate_ranges(config, strict)
        cls._validate_relationships(config)
        
    @classmethod
    def _validate_ranges(cls, config: Dict[str, Any], strict: bool) -> None:
        """Validate parameter ranges"""
        for param, (min_val, max_val) in cls.PARAM_RANGES.items():
            if param in config:
                value = config[param]
                if strict:
                    if not min_val <= value <= max_val:
                        raise ValueError(f"{param} must be between {min_val} and {max_val}, got {value}")
                else:
                    if value < min_val * 0.5 or value > max_val * 2:
                        raise ValueError(f"{param} is far outside recommended range [{min_val}, {max_val}], got {value}")
                        
    @classmethod                    
    def _validate_relationships(cls, config: Dict[str, Any]) -> None:
        """Validate relationships between parameters"""
        # Architecture constraints
        if config["hidden_size"] % config["num_heads"] != 0:
            raise ValueError(f"hidden_size ({config['hidden_size']}) must be divisible by num_heads ({config['num_heads']})")
            
        if "intermediate_size" in config:
            if config["intermediate_size"] < config["hidden_size"]:
                raise ValueError(f"intermediate_size ({config['intermediate_size']}) should not be smaller than hidden_size ({config['hidden_size']})")
        
        # Memory constraints
        mem_per_token = config["hidden_size"] * 4  # Rough estimate of memory per token
        total_mem = mem_per_token * config["max_sequence_length"] * config["num_layers"]
        if total_mem > 1e10:  # 10GB warning threshold
            cls._logger.warning(f"Model may require significant memory (rough estimate: {total_mem/1e9:.1f}GB)")
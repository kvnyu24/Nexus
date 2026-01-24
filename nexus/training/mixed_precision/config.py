"""Mixed precision training configuration."""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum

import torch


class FP8Format(Enum):
    """FP8 format types.

    E4M3: 4-bit exponent, 3-bit mantissa. Better dynamic range.
           Used for forward pass activations.
    E5M2: 5-bit exponent, 2-bit mantissa. Better precision.
           Used for gradients.
    """
    E4M3 = "e4m3"
    E5M2 = "e5m2"

    @property
    def max_value(self) -> float:
        """Maximum representable value for this format."""
        if self == FP8Format.E4M3:
            return 448.0  # 2^8 * (1 + 7/8)
        else:  # E5M2
            return 57344.0  # 2^15 * (1 + 3/4)

    @property
    def min_positive(self) -> float:
        """Minimum positive representable value."""
        if self == FP8Format.E4M3:
            return 2**-9  # Smallest subnormal
        else:  # E5M2
            return 2**-16


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training.

    This configuration controls the precision settings for different
    aspects of training, including computation, parameter storage,
    gradient accumulation, and optional FP8 support.

    Args:
        compute_dtype: Dtype for computation (float16, bfloat16, float32).
                      Determines the precision used in forward/backward passes.
        param_dtype: Dtype for parameters. If different from compute_dtype,
                    parameters are cast during computation.
        grad_dtype: Dtype for gradients. Affects gradient accumulation precision.
        use_fp8: Whether to use FP8 for applicable layers. Requires compatible
                hardware (e.g., NVIDIA H100, AMD MI300).
        fp8_format: FP8 format to use ('e4m3' or 'e5m2').
                   E4M3 is recommended for forward pass, E5M2 for gradients.
        loss_scale: Initial loss scale for fp16 training. Higher values
                   reduce underflow risk but may cause overflow.
        dynamic_loss_scale: Whether to use dynamic loss scaling.
                           Automatically adjusts scale based on gradient statistics.
        scale_growth_factor: Factor to increase scale when no overflow detected.
        scale_backoff_factor: Factor to decrease scale on overflow.
        scale_growth_interval: Number of steps between scale increases.
        fp8_amax_history_len: Length of amax history for FP8 scaling.
        fp8_amax_compute_algo: Algorithm for computing amax ('max', 'most_recent').
        enabled: Master switch to enable/disable mixed precision.

    Example:
        >>> config = MixedPrecisionConfig(
        ...     compute_dtype=torch.bfloat16,
        ...     use_fp8=True,
        ...     fp8_format=FP8Format.E4M3
        ... )
        >>> # Use in training
        >>> trainer = MixedPrecisionTrainer(model, config=config)
    """

    # Precision settings
    compute_dtype: torch.dtype = torch.float16
    param_dtype: torch.dtype = torch.float32
    grad_dtype: torch.dtype = torch.float32

    # FP8 settings
    use_fp8: bool = False
    fp8_format: FP8Format = FP8Format.E4M3

    # Loss scaling settings
    loss_scale: float = 65536.0
    dynamic_loss_scale: bool = True
    scale_growth_factor: float = 2.0
    scale_backoff_factor: float = 0.5
    scale_growth_interval: int = 2000

    # FP8 scaling settings
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: Literal["max", "most_recent"] = "max"

    # Master switch
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        valid_compute_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if self.compute_dtype not in valid_compute_dtypes:
            raise ValueError(
                f"compute_dtype must be one of {valid_compute_dtypes}, "
                f"got {self.compute_dtype}"
            )

        valid_param_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if self.param_dtype not in valid_param_dtypes:
            raise ValueError(
                f"param_dtype must be one of {valid_param_dtypes}, "
                f"got {self.param_dtype}"
            )

        if self.loss_scale <= 0:
            raise ValueError(f"loss_scale must be positive, got {self.loss_scale}")

        if self.scale_growth_factor <= 1.0:
            raise ValueError(
                f"scale_growth_factor must be > 1.0, got {self.scale_growth_factor}"
            )

        if not 0 < self.scale_backoff_factor < 1.0:
            raise ValueError(
                f"scale_backoff_factor must be in (0, 1), got {self.scale_backoff_factor}"
            )

        if self.scale_growth_interval <= 0:
            raise ValueError(
                f"scale_growth_interval must be positive, got {self.scale_growth_interval}"
            )

        if self.fp8_amax_history_len <= 0:
            raise ValueError(
                f"fp8_amax_history_len must be positive, got {self.fp8_amax_history_len}"
            )

        if self.fp8_amax_compute_algo not in ("max", "most_recent"):
            raise ValueError(
                f"fp8_amax_compute_algo must be 'max' or 'most_recent', "
                f"got {self.fp8_amax_compute_algo}"
            )

    @classmethod
    def fp16(cls) -> "MixedPrecisionConfig":
        """Create a configuration for FP16 mixed precision training.

        Returns:
            MixedPrecisionConfig configured for FP16.
        """
        return cls(
            compute_dtype=torch.float16,
            param_dtype=torch.float32,
            grad_dtype=torch.float32,
            dynamic_loss_scale=True
        )

    @classmethod
    def bf16(cls) -> "MixedPrecisionConfig":
        """Create a configuration for BF16 mixed precision training.

        BF16 has a larger dynamic range than FP16, so loss scaling
        is typically not needed.

        Returns:
            MixedPrecisionConfig configured for BF16.
        """
        return cls(
            compute_dtype=torch.bfloat16,
            param_dtype=torch.float32,
            grad_dtype=torch.float32,
            dynamic_loss_scale=False,
            loss_scale=1.0
        )

    @classmethod
    def fp8(cls, format: FP8Format = FP8Format.E4M3) -> "MixedPrecisionConfig":
        """Create a configuration for FP8 training.

        FP8 provides significant memory savings but requires
        compatible hardware and careful scaling.

        Args:
            format: FP8 format to use.

        Returns:
            MixedPrecisionConfig configured for FP8.
        """
        return cls(
            compute_dtype=torch.bfloat16,  # Compute in BF16, store in FP8
            param_dtype=torch.float32,
            grad_dtype=torch.float32,
            use_fp8=True,
            fp8_format=format,
            dynamic_loss_scale=True
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "compute_dtype": str(self.compute_dtype),
            "param_dtype": str(self.param_dtype),
            "grad_dtype": str(self.grad_dtype),
            "use_fp8": self.use_fp8,
            "fp8_format": self.fp8_format.value,
            "loss_scale": self.loss_scale,
            "dynamic_loss_scale": self.dynamic_loss_scale,
            "scale_growth_factor": self.scale_growth_factor,
            "scale_backoff_factor": self.scale_backoff_factor,
            "scale_growth_interval": self.scale_growth_interval,
            "fp8_amax_history_len": self.fp8_amax_history_len,
            "fp8_amax_compute_algo": self.fp8_amax_compute_algo,
            "enabled": self.enabled
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MixedPrecisionConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values.

        Returns:
            MixedPrecisionConfig instance.
        """
        # Parse dtype strings
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
        }

        compute_dtype = dtype_map.get(
            config_dict.get("compute_dtype", "torch.float16"),
            torch.float16
        )
        param_dtype = dtype_map.get(
            config_dict.get("param_dtype", "torch.float32"),
            torch.float32
        )
        grad_dtype = dtype_map.get(
            config_dict.get("grad_dtype", "torch.float32"),
            torch.float32
        )

        # Parse FP8 format
        fp8_format_str = config_dict.get("fp8_format", "e4m3")
        fp8_format = FP8Format.E4M3 if fp8_format_str == "e4m3" else FP8Format.E5M2

        return cls(
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            grad_dtype=grad_dtype,
            use_fp8=config_dict.get("use_fp8", False),
            fp8_format=fp8_format,
            loss_scale=config_dict.get("loss_scale", 65536.0),
            dynamic_loss_scale=config_dict.get("dynamic_loss_scale", True),
            scale_growth_factor=config_dict.get("scale_growth_factor", 2.0),
            scale_backoff_factor=config_dict.get("scale_backoff_factor", 0.5),
            scale_growth_interval=config_dict.get("scale_growth_interval", 2000),
            fp8_amax_history_len=config_dict.get("fp8_amax_history_len", 1024),
            fp8_amax_compute_algo=config_dict.get("fp8_amax_compute_algo", "max"),
            enabled=config_dict.get("enabled", True)
        )

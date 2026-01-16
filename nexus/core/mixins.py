import torch
from typing import Any, Dict, List, Optional, Union, Tuple


class InputValidationMixin:
    """Mixin class providing common input validation methods for tensor operations."""

    def validate_finite(self, x: torch.Tensor, name: str = "input") -> None:
        """Validate that a tensor contains only finite values (no inf or nan).

        Args:
            x: The tensor to validate.
            name: Name of the input for error messages.

        Raises:
            ValueError: If the tensor contains inf or nan values.
        """
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains inf or nan values")

    def validate_shape(
        self,
        x: torch.Tensor,
        expected_dims: int,
        name: str = "input"
    ) -> None:
        """Validate that a tensor has the expected number of dimensions.

        Args:
            x: The tensor to validate.
            expected_dims: Expected number of dimensions.
            name: Name of the input for error messages.

        Raises:
            ValueError: If the tensor does not have the expected dimensionality.
        """
        if x.dim() != expected_dims:
            raise ValueError(
                f"{name} must have {expected_dims} dimensions, got {x.dim()}"
            )

    def validate_dtype(
        self,
        x: torch.Tensor,
        expected_dtype: Union[torch.dtype, Tuple[torch.dtype, ...]],
        name: str = "input"
    ) -> None:
        """Validate that a tensor has the expected dtype.

        Args:
            x: The tensor to validate.
            expected_dtype: Expected dtype or tuple of acceptable dtypes.
            name: Name of the input for error messages.

        Raises:
            ValueError: If the tensor does not have the expected dtype.
        """
        if isinstance(expected_dtype, tuple):
            if x.dtype not in expected_dtype:
                dtype_names = ", ".join(str(d) for d in expected_dtype)
                raise ValueError(
                    f"{name} must have dtype in ({dtype_names}), got {x.dtype}"
                )
        else:
            if x.dtype != expected_dtype:
                raise ValueError(
                    f"{name} must have dtype {expected_dtype}, got {x.dtype}"
                )


class ConfigValidatorMixin:
    """Mixin class providing common configuration validation methods.

    This mixin reduces code duplication across model classes that need to
    validate configuration dictionaries with required/optional keys and
    numeric constraints.
    """

    def validate_config(
        self,
        config: Dict[str, Any],
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None
    ) -> None:
        """Validate that a configuration dictionary contains all required keys.

        Args:
            config: The configuration dictionary to validate.
            required_keys: List of keys that must be present in the config.
            optional_keys: Optional list of valid optional keys. If provided,
                any key not in required_keys or optional_keys will raise an error.

        Raises:
            ValueError: If any required key is missing from the config.
            ValueError: If optional_keys is provided and config contains
                keys not in required_keys or optional_keys.
        """
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        if optional_keys is not None:
            valid_keys = set(required_keys) | set(optional_keys)
            for key in config:
                if key not in valid_keys:
                    raise ValueError(
                        f"Unknown config key: {key}. "
                        f"Valid keys are: {sorted(valid_keys)}"
                    )

    def validate_positive(
        self,
        value: Union[int, float],
        name: str
    ) -> None:
        """Validate that a numeric value is positive (greater than zero).

        Args:
            value: The numeric value to validate.
            name: Name of the parameter for error messages.

        Raises:
            ValueError: If the value is not positive.
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def validate_range(
        self,
        value: Union[int, float],
        min_val: Union[int, float],
        max_val: Union[int, float],
        name: str
    ) -> None:
        """Validate that a numeric value falls within a specified range.

        Args:
            value: The numeric value to validate.
            min_val: Minimum allowed value (inclusive).
            max_val: Maximum allowed value (inclusive).
            name: Name of the parameter for error messages.

        Raises:
            ValueError: If the value is outside the specified range.
        """
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} must be in range [{min_val}, {max_val}], got {value}"
            )


class FeatureBankMixin:
    """Mixin class providing feature bank buffer management for circular buffer patterns.

    This mixin consolidates the duplicate buffer management pattern found across
    model files that need to maintain feature banks with circular buffer updates.
    It provides methods to register, update, and retrieve feature banks with
    proper handling of buffer pointers and filled flags.
    """

    def register_feature_bank(
        self,
        name: str,
        bank_size: int,
        feature_dim: int,
        dtype: torch.dtype = torch.float32
    ) -> None:
        """Register a feature bank buffer with pointer and filled flag.

        Creates three buffers:
        - {name}_bank: The main feature storage tensor of shape (bank_size, feature_dim)
        - {name}_ptr: A pointer tracking the next write position
        - {name}_filled: A flag indicating whether the bank has been fully filled

        Args:
            name: The name prefix for the buffer.
            bank_size: Maximum number of features to store in the bank.
            feature_dim: Dimensionality of each feature vector.
            dtype: Data type for the feature bank tensor. Defaults to torch.float32.
        """
        self.register_buffer(f"{name}_bank", torch.zeros(bank_size, feature_dim, dtype=dtype))
        self.register_buffer(f"{name}_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer(f"{name}_filled", torch.zeros(1, dtype=torch.bool))

    def update_feature_bank(self, name: str, features: torch.Tensor) -> None:
        """Update feature bank with new features, checking for finite values.

        Performs a circular buffer update, wrapping around when the end of the
        bank is reached. Non-finite values (inf or nan) are silently skipped
        to maintain data integrity.

        Args:
            name: The name prefix of the feature bank to update.
            features: New features to add, shape (batch_size, feature_dim).
        """
        # Skip non-finite values
        if not torch.isfinite(features).all():
            return

        bank = getattr(self, f"{name}_bank")
        ptr = getattr(self, f"{name}_ptr")
        filled = getattr(self, f"{name}_filled")

        batch_size = features.shape[0]
        bank_size = bank.shape[0]

        # Circular buffer update
        end_ptr = (ptr.item() + batch_size) % bank_size
        if ptr.item() + batch_size <= bank_size:
            bank[ptr.item():ptr.item() + batch_size] = features.detach()
        else:
            # Wrap around
            first_part = bank_size - ptr.item()
            bank[ptr.item():] = features[:first_part].detach()
            bank[:end_ptr] = features[first_part:].detach()

        ptr[0] = end_ptr
        if ptr.item() == 0 or filled.item():
            filled[0] = True

    def get_feature_bank(self, name: str) -> torch.Tensor:
        """Get the valid portion of the feature bank.

        Returns the entire bank if it has been fully filled at least once,
        otherwise returns only the portion that has been written to.

        Args:
            name: The name prefix of the feature bank to retrieve.

        Returns:
            Tensor containing the valid features in the bank.
        """
        bank = getattr(self, f"{name}_bank")
        ptr = getattr(self, f"{name}_ptr")
        filled = getattr(self, f"{name}_filled")

        if filled.item():
            return bank
        else:
            return bank[:ptr.item()]

    def is_bank_full(self, name: str) -> bool:
        """Check if feature bank is full.

        A bank is considered full once it has been completely filled at least
        once, even if new features have since wrapped around.

        Args:
            name: The name prefix of the feature bank to check.

        Returns:
            True if the bank has been fully filled at least once, False otherwise.
        """
        return getattr(self, f"{name}_filled").item()

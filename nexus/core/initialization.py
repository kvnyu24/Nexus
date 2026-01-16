"""
Weight Initialization Module for Nexus Framework.

This module provides comprehensive weight initialization utilities that consolidate
the various initialization patterns found across 18+ model files in the codebase.

Supported initialization strategies:
- Xavier (Glorot) uniform and normal
- Kaiming (He) uniform and normal
- Normal distribution with configurable mean/std
- Truncated normal distribution
- Orthogonal initialization
- Uniform distribution
- Constant initialization

Supported layer types:
- nn.Linear
- nn.Conv1d, nn.Conv2d, nn.Conv3d
- nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
- nn.Embedding
- nn.LayerNorm
- nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
- nn.GroupNorm
- nn.LSTM, nn.GRU, nn.RNN
"""

import math
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class InitMethod(str, Enum):
    """Enumeration of available weight initialization methods."""
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    ORTHOGONAL = "orthogonal"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    ZEROS = "zeros"
    ONES = "ones"


class WeightInitializer:
    """
    Comprehensive weight initialization utility class.

    Provides static methods for different initialization strategies and a unified
    interface for initializing all weights in a module hierarchy.

    Example usage:
        # Initialize a single layer
        WeightInitializer.kaiming_normal(linear_layer.weight, nonlinearity='relu')

        # Initialize an entire module
        WeightInitializer.initialize_module(
            model,
            method=InitMethod.KAIMING_NORMAL,
            nonlinearity='relu'
        )
    """

    # Default initialization parameters used across Nexus models
    DEFAULT_STD = 0.02
    DEFAULT_MEAN = 0.0
    DEFAULT_GAIN = 1.0

    # ==================== Static Initialization Methods ====================

    @staticmethod
    def xavier_uniform(
        tensor: torch.Tensor,
        gain: float = 1.0
    ) -> torch.Tensor:
        """
        Initialize tensor using Xavier (Glorot) uniform distribution.

        Fills the input tensor with values according to the method described in
        "Understanding the difficulty of training deep feedforward neural networks"
        by Glorot & Bengio (2010).

        Args:
            tensor: The tensor to initialize.
            gain: Scaling factor for the weights.

        Returns:
            The initialized tensor.
        """
        return nn.init.xavier_uniform_(tensor, gain=gain)

    @staticmethod
    def xavier_normal(
        tensor: torch.Tensor,
        gain: float = 1.0
    ) -> torch.Tensor:
        """
        Initialize tensor using Xavier (Glorot) normal distribution.

        Args:
            tensor: The tensor to initialize.
            gain: Scaling factor for the weights.

        Returns:
            The initialized tensor.
        """
        return nn.init.xavier_normal_(tensor, gain=gain)

    @staticmethod
    def kaiming_uniform(
        tensor: torch.Tensor,
        a: float = 0,
        mode: str = 'fan_in',
        nonlinearity: str = 'leaky_relu'
    ) -> torch.Tensor:
        """
        Initialize tensor using Kaiming (He) uniform distribution.

        Fills the input tensor with values according to the method described in
        "Delving deep into rectifiers: Surpassing human-level performance on
        ImageNet classification" by He et al. (2015).

        Args:
            tensor: The tensor to initialize.
            a: Negative slope of the rectifier (for leaky_relu).
            mode: Either 'fan_in' or 'fan_out'.
            nonlinearity: The non-linear function name.

        Returns:
            The initialized tensor.
        """
        return nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

    @staticmethod
    def kaiming_normal(
        tensor: torch.Tensor,
        a: float = 0,
        mode: str = 'fan_out',
        nonlinearity: str = 'relu'
    ) -> torch.Tensor:
        """
        Initialize tensor using Kaiming (He) normal distribution.

        Args:
            tensor: The tensor to initialize.
            a: Negative slope of the rectifier (for leaky_relu).
            mode: Either 'fan_in' or 'fan_out'.
            nonlinearity: The non-linear function name.

        Returns:
            The initialized tensor.
        """
        return nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

    @staticmethod
    def normal(
        tensor: torch.Tensor,
        mean: float = 0.0,
        std: float = 0.02
    ) -> torch.Tensor:
        """
        Initialize tensor using normal (Gaussian) distribution.

        This is the most common initialization pattern found in Nexus LLM models
        (GPT, Qwen, EdgeLLM, etc.) with std=0.02.

        Args:
            tensor: The tensor to initialize.
            mean: Mean of the normal distribution.
            std: Standard deviation of the normal distribution.

        Returns:
            The initialized tensor.
        """
        return nn.init.normal_(tensor, mean=mean, std=std)

    @staticmethod
    def truncated_normal(
        tensor: torch.Tensor,
        mean: float = 0.0,
        std: float = 0.02,
        a: Optional[float] = None,
        b: Optional[float] = None
    ) -> torch.Tensor:
        """
        Initialize tensor using truncated normal distribution.

        Values are drawn from a normal distribution and clamped to [a, b].
        Used in ViT and EnhancedSFT models for improved stability.

        Args:
            tensor: The tensor to initialize.
            mean: Mean of the normal distribution.
            std: Standard deviation of the normal distribution.
            a: Minimum cutoff value. Defaults to mean - 2*std.
            b: Maximum cutoff value. Defaults to mean + 2*std.

        Returns:
            The initialized tensor.
        """
        if a is None:
            a = mean - 2 * std
        if b is None:
            b = mean + 2 * std
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    @staticmethod
    def orthogonal(
        tensor: torch.Tensor,
        gain: float = 1.0
    ) -> torch.Tensor:
        """
        Initialize tensor with orthogonal initialization.

        Fills the input tensor with a (semi) orthogonal matrix. Used for
        state embeddings in Neural State Machine and RNN models.

        Args:
            tensor: The tensor to initialize (must be at least 2D).
            gain: Scaling factor for the orthogonal matrix.

        Returns:
            The initialized tensor.
        """
        return nn.init.orthogonal_(tensor, gain=gain)

    @staticmethod
    def uniform(
        tensor: torch.Tensor,
        a: float = 0.0,
        b: float = 1.0
    ) -> torch.Tensor:
        """
        Initialize tensor using uniform distribution.

        Args:
            tensor: The tensor to initialize.
            a: Lower bound of the uniform distribution.
            b: Upper bound of the uniform distribution.

        Returns:
            The initialized tensor.
        """
        return nn.init.uniform_(tensor, a=a, b=b)

    @staticmethod
    def constant(
        tensor: torch.Tensor,
        val: float
    ) -> torch.Tensor:
        """
        Initialize tensor with a constant value.

        Args:
            tensor: The tensor to initialize.
            val: The value to fill the tensor with.

        Returns:
            The initialized tensor.
        """
        return nn.init.constant_(tensor, val)

    @staticmethod
    def zeros(tensor: torch.Tensor) -> torch.Tensor:
        """Initialize tensor with zeros."""
        return nn.init.zeros_(tensor)

    @staticmethod
    def ones(tensor: torch.Tensor) -> torch.Tensor:
        """Initialize tensor with ones."""
        return nn.init.ones_(tensor)

    # ==================== Layer-Specific Initialization ====================

    @classmethod
    def init_linear(
        cls,
        module: nn.Linear,
        method: Union[str, InitMethod] = InitMethod.KAIMING_NORMAL,
        **kwargs
    ) -> None:
        """
        Initialize a Linear layer with the specified method.

        Args:
            module: The Linear layer to initialize.
            method: Initialization method to use.
            **kwargs: Additional arguments passed to the initialization function.
        """
        method = InitMethod(method) if isinstance(method, str) else method
        cls._init_weight(module.weight, method, **kwargs)
        if module.bias is not None:
            cls.zeros(module.bias)

    @classmethod
    def init_conv(
        cls,
        module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d,
                      nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d],
        method: Union[str, InitMethod] = InitMethod.KAIMING_NORMAL,
        **kwargs
    ) -> None:
        """
        Initialize a Convolutional layer with the specified method.

        Default uses Kaiming normal with fan_out mode, following ResNet/VGG patterns.

        Args:
            module: The Conv layer to initialize.
            method: Initialization method to use.
            **kwargs: Additional arguments passed to the initialization function.
        """
        method = InitMethod(method) if isinstance(method, str) else method

        # Set defaults for conv layers
        if method == InitMethod.KAIMING_NORMAL and 'mode' not in kwargs:
            kwargs['mode'] = 'fan_out'
        if method == InitMethod.KAIMING_NORMAL and 'nonlinearity' not in kwargs:
            kwargs['nonlinearity'] = 'relu'

        cls._init_weight(module.weight, method, **kwargs)
        if module.bias is not None:
            cls.zeros(module.bias)

    @classmethod
    def init_embedding(
        cls,
        module: nn.Embedding,
        method: Union[str, InitMethod] = InitMethod.NORMAL,
        **kwargs
    ) -> None:
        """
        Initialize an Embedding layer with the specified method.

        Default uses normal distribution with std=0.02 following LLM patterns.
        Zeros out the padding index if present.

        Args:
            module: The Embedding layer to initialize.
            method: Initialization method to use.
            **kwargs: Additional arguments passed to the initialization function.
        """
        method = InitMethod(method) if isinstance(method, str) else method

        # Set default std for normal initialization
        if method == InitMethod.NORMAL and 'std' not in kwargs:
            kwargs['std'] = cls.DEFAULT_STD

        cls._init_weight(module.weight, method, **kwargs)

        # Zero out padding embedding if present
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    @classmethod
    def init_layer_norm(
        cls,
        module: nn.LayerNorm
    ) -> None:
        """
        Initialize a LayerNorm layer.

        Sets weight to 1 and bias to 0, following standard practice.

        Args:
            module: The LayerNorm layer to initialize.
        """
        if module.elementwise_affine:
            cls.ones(module.weight)
            cls.zeros(module.bias)

    @classmethod
    def init_batch_norm(
        cls,
        module: Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d],
        weight_val: float = 1.0,
        bias_val: float = 0.0
    ) -> None:
        """
        Initialize a BatchNorm layer.

        Default sets weight to 1 and bias to 0. GAN models may use different
        values (e.g., normal distribution for weight).

        Args:
            module: The BatchNorm layer to initialize.
            weight_val: Value to initialize weight to.
            bias_val: Value to initialize bias to.
        """
        if module.affine:
            cls.constant(module.weight, weight_val)
            cls.constant(module.bias, bias_val)

    @classmethod
    def init_group_norm(
        cls,
        module: nn.GroupNorm
    ) -> None:
        """
        Initialize a GroupNorm layer.

        Args:
            module: The GroupNorm layer to initialize.
        """
        if module.affine:
            cls.ones(module.weight)
            cls.zeros(module.bias)

    @classmethod
    def init_rnn(
        cls,
        module: Union[nn.RNN, nn.LSTM, nn.GRU],
        method: Union[str, InitMethod] = InitMethod.ORTHOGONAL,
        **kwargs
    ) -> None:
        """
        Initialize an RNN/LSTM/GRU layer.

        Default uses orthogonal initialization for recurrent weights.

        Args:
            module: The RNN layer to initialize.
            method: Initialization method to use.
            **kwargs: Additional arguments passed to the initialization function.
        """
        method = InitMethod(method) if isinstance(method, str) else method

        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                cls._init_weight(param, method, **kwargs)
            elif 'weight_hh' in name:
                cls.orthogonal(param, gain=kwargs.get('gain', 1.0))
            elif 'bias' in name:
                cls.zeros(param)

    # ==================== Module-Level Initialization ====================

    @classmethod
    def initialize_module(
        cls,
        module: nn.Module,
        method: Union[str, InitMethod] = InitMethod.KAIMING_NORMAL,
        linear_method: Optional[Union[str, InitMethod]] = None,
        conv_method: Optional[Union[str, InitMethod]] = None,
        embedding_method: Optional[Union[str, InitMethod]] = None,
        rnn_method: Optional[Union[str, InitMethod]] = None,
        **kwargs
    ) -> nn.Module:
        """
        Initialize all weights in a module hierarchy.

        This method walks through all submodules and applies appropriate
        initialization based on layer type.

        Args:
            module: The module to initialize.
            method: Default initialization method for weight layers.
            linear_method: Override method for Linear layers.
            conv_method: Override method for Conv layers.
            embedding_method: Override method for Embedding layers.
            rnn_method: Override method for RNN layers.
            **kwargs: Additional arguments passed to initialization functions.

        Returns:
            The initialized module.
        """
        method = InitMethod(method) if isinstance(method, str) else method
        linear_method = InitMethod(linear_method) if linear_method else method
        conv_method = InitMethod(conv_method) if conv_method else method
        embedding_method = InitMethod(embedding_method) if embedding_method else InitMethod.NORMAL
        rnn_method = InitMethod(rnn_method) if rnn_method else InitMethod.ORTHOGONAL

        for m in module.modules():
            if isinstance(m, nn.Linear):
                cls.init_linear(m, linear_method, **kwargs)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                               nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                cls.init_conv(m, conv_method, **kwargs)
            elif isinstance(m, nn.Embedding):
                cls.init_embedding(m, embedding_method, **kwargs)
            elif isinstance(m, nn.LayerNorm):
                cls.init_layer_norm(m)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                cls.init_batch_norm(m)
            elif isinstance(m, nn.GroupNorm):
                cls.init_group_norm(m)
            elif isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
                cls.init_rnn(m, rnn_method, **kwargs)

        return module

    @classmethod
    def _init_weight(
        cls,
        tensor: torch.Tensor,
        method: InitMethod,
        **kwargs
    ) -> torch.Tensor:
        """
        Initialize a weight tensor using the specified method.

        Args:
            tensor: The tensor to initialize.
            method: The initialization method.
            **kwargs: Additional arguments for the initialization function.

        Returns:
            The initialized tensor.
        """
        if method == InitMethod.XAVIER_UNIFORM:
            return cls.xavier_uniform(tensor, gain=kwargs.get('gain', cls.DEFAULT_GAIN))
        elif method == InitMethod.XAVIER_NORMAL:
            return cls.xavier_normal(tensor, gain=kwargs.get('gain', cls.DEFAULT_GAIN))
        elif method == InitMethod.KAIMING_UNIFORM:
            return cls.kaiming_uniform(
                tensor,
                a=kwargs.get('a', 0),
                mode=kwargs.get('mode', 'fan_in'),
                nonlinearity=kwargs.get('nonlinearity', 'leaky_relu')
            )
        elif method == InitMethod.KAIMING_NORMAL:
            return cls.kaiming_normal(
                tensor,
                a=kwargs.get('a', 0),
                mode=kwargs.get('mode', 'fan_out'),
                nonlinearity=kwargs.get('nonlinearity', 'relu')
            )
        elif method == InitMethod.NORMAL:
            return cls.normal(
                tensor,
                mean=kwargs.get('mean', cls.DEFAULT_MEAN),
                std=kwargs.get('std', cls.DEFAULT_STD)
            )
        elif method == InitMethod.TRUNCATED_NORMAL:
            return cls.truncated_normal(
                tensor,
                mean=kwargs.get('mean', cls.DEFAULT_MEAN),
                std=kwargs.get('std', cls.DEFAULT_STD),
                a=kwargs.get('a'),
                b=kwargs.get('b')
            )
        elif method == InitMethod.ORTHOGONAL:
            return cls.orthogonal(tensor, gain=kwargs.get('gain', cls.DEFAULT_GAIN))
        elif method == InitMethod.UNIFORM:
            return cls.uniform(tensor, a=kwargs.get('a', 0.0), b=kwargs.get('b', 1.0))
        elif method == InitMethod.CONSTANT:
            return cls.constant(tensor, val=kwargs.get('val', 0.0))
        elif method == InitMethod.ZEROS:
            return cls.zeros(tensor)
        elif method == InitMethod.ONES:
            return cls.ones(tensor)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    # ==================== Utility Methods ====================

    @staticmethod
    def apply_weight_norm(module: nn.Module, name: str = 'weight') -> nn.Module:
        """
        Apply weight normalization to a module.

        Args:
            module: The module to apply weight normalization to.
            name: Name of the weight parameter.

        Returns:
            The module with weight normalization applied.
        """
        return nn.utils.weight_norm(module, name=name)

    @staticmethod
    def remove_weight_norm(module: nn.Module) -> None:
        """
        Remove weight normalization from all submodules.

        Args:
            module: The module to remove weight normalization from.
        """
        for m in module.modules():
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                pass

    @staticmethod
    def apply_spectral_norm(module: nn.Module, name: str = 'weight') -> nn.Module:
        """
        Apply spectral normalization to a module.

        Useful for GAN discriminators to enforce Lipschitz constraint.

        Args:
            module: The module to apply spectral normalization to.
            name: Name of the weight parameter.

        Returns:
            The module with spectral normalization applied.
        """
        return nn.utils.spectral_norm(module, name=name)

    @staticmethod
    def remove_spectral_norm(module: nn.Module) -> None:
        """
        Remove spectral normalization from all submodules.

        Args:
            module: The module to remove spectral normalization from.
        """
        for m in module.modules():
            try:
                nn.utils.remove_spectral_norm(m)
            except ValueError:
                pass


class WeightInitMixin:
    """
    Mixin class providing weight initialization functionality for nn.Module subclasses.

    This mixin consolidates the duplicate _init_weights patterns found across 18+
    model files in the Nexus codebase. Models can inherit from this mixin to get
    standardized weight initialization with minimal code.

    Example usage:
        class MyModel(WeightInitMixin, NexusModule):
            def __init__(self, config):
                super().__init__(config)
                self.linear = nn.Linear(256, 128)
                self.conv = nn.Conv2d(3, 64, 3)
                self.norm = nn.LayerNorm(128)

                # Initialize all weights using default settings
                self.apply_weight_init()

                # Or with custom settings
                self.apply_weight_init(
                    method=InitMethod.XAVIER_UNIFORM,
                    gain=0.5
                )

    Supported initialization presets:
        - 'llm': Normal distribution (std=0.02), used in GPT, Qwen, LLaMA
        - 'vision': Kaiming normal with fan_out, used in ResNet, VGG
        - 'vit': Truncated normal (std=0.02), used in Vision Transformer
        - 'gan': Normal distribution (std=0.02) for conv, different for BatchNorm
        - 'default': Kaiming normal for most layers
    """

    # Preset configurations for common model types
    INIT_PRESETS: Dict[str, Dict[str, Any]] = {
        'llm': {
            'method': InitMethod.NORMAL,
            'std': 0.02,
        },
        'vision': {
            'method': InitMethod.KAIMING_NORMAL,
            'mode': 'fan_out',
            'nonlinearity': 'relu',
        },
        'vit': {
            'method': InitMethod.TRUNCATED_NORMAL,
            'std': 0.02,
        },
        'gan': {
            'method': InitMethod.NORMAL,
            'std': 0.02,
        },
        'default': {
            'method': InitMethod.KAIMING_NORMAL,
            'mode': 'fan_out',
            'nonlinearity': 'relu',
        },
    }

    def _init_weights(self, module: nn.Module) -> None:
        """
        Default weight initialization callback for use with module.apply().

        This method provides the standard initialization pattern found across
        Nexus models. Override this method in subclasses for custom behavior.

        Args:
            module: The module to initialize.
        """
        if isinstance(module, nn.Linear):
            WeightInitializer.init_linear(module, self._get_init_method())
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            WeightInitializer.init_conv(module, self._get_init_method())
        elif isinstance(module, nn.Embedding):
            WeightInitializer.init_embedding(module)
        elif isinstance(module, nn.LayerNorm):
            WeightInitializer.init_layer_norm(module)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            WeightInitializer.init_batch_norm(module)
        elif isinstance(module, nn.GroupNorm):
            WeightInitializer.init_group_norm(module)
        elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            WeightInitializer.init_rnn(module)

    def _get_init_method(self) -> InitMethod:
        """
        Get the initialization method for this module.

        Override in subclasses to specify a different default method.
        Can also check for a config value like self.config.get('init_method').

        Returns:
            The initialization method to use.
        """
        if hasattr(self, 'config') and isinstance(self.config, dict):
            method = self.config.get('init_method', 'kaiming_normal')
            return InitMethod(method)
        return InitMethod.KAIMING_NORMAL

    def _get_init_kwargs(self) -> Dict[str, Any]:
        """
        Get additional initialization keyword arguments.

        Override in subclasses to provide custom initialization parameters.

        Returns:
            Dictionary of keyword arguments for initialization.
        """
        kwargs = {}
        if hasattr(self, 'config') and isinstance(self.config, dict):
            if 'initializer_range' in self.config:
                kwargs['std'] = self.config['initializer_range']
            if 'init_std' in self.config:
                kwargs['std'] = self.config['init_std']
            if 'init_gain' in self.config:
                kwargs['gain'] = self.config['init_gain']
        return kwargs

    def apply_weight_init(
        self,
        method: Optional[Union[str, InitMethod]] = None,
        preset: Optional[str] = None,
        **kwargs
    ) -> 'WeightInitMixin':
        """
        Apply weight initialization to all submodules.

        Args:
            method: Initialization method to use. If None, uses the default
                    from _get_init_method().
            preset: Use a preset configuration ('llm', 'vision', 'vit', 'gan', 'default').
                   Overrides method if provided.
            **kwargs: Additional arguments passed to initialization functions.

        Returns:
            Self for method chaining.
        """
        if preset is not None:
            if preset not in self.INIT_PRESETS:
                raise ValueError(
                    f"Unknown preset: {preset}. "
                    f"Available presets: {list(self.INIT_PRESETS.keys())}"
                )
            preset_config = self.INIT_PRESETS[preset].copy()
            method = preset_config.pop('method')
            kwargs = {**preset_config, **kwargs}

        if method is None:
            method = self._get_init_method()
            kwargs = {**self._get_init_kwargs(), **kwargs}

        WeightInitializer.initialize_module(self, method=method, **kwargs)
        return self

    def init_weights_llm(self, std: float = 0.02) -> 'WeightInitMixin':
        """
        Apply LLM-style weight initialization (normal distribution).

        This is the initialization pattern used in GPT, Qwen, LLaMA, EdgeLLM,
        and other language models in the Nexus codebase.

        Args:
            std: Standard deviation for normal distribution.

        Returns:
            Self for method chaining.
        """
        return self.apply_weight_init(preset='llm', std=std)

    def init_weights_vision(self, nonlinearity: str = 'relu') -> 'WeightInitMixin':
        """
        Apply vision model weight initialization (Kaiming normal).

        This is the initialization pattern used in ResNet, VGG, RCNN,
        and other vision models in the Nexus codebase.

        Args:
            nonlinearity: The non-linear function used after conv layers.

        Returns:
            Self for method chaining.
        """
        return self.apply_weight_init(preset='vision', nonlinearity=nonlinearity)

    def init_weights_vit(self, std: float = 0.02) -> 'WeightInitMixin':
        """
        Apply Vision Transformer weight initialization (truncated normal).

        Args:
            std: Standard deviation for truncated normal distribution.

        Returns:
            Self for method chaining.
        """
        return self.apply_weight_init(preset='vit', std=std)

    def init_weights_gan(
        self,
        std: float = 0.02,
        batchnorm_weight: float = 1.0
    ) -> 'WeightInitMixin':
        """
        Apply GAN-style weight initialization.

        This follows the initialization pattern used in BaseGAN, WGAN, and
        other generative models in the Nexus codebase.

        Args:
            std: Standard deviation for normal distribution.
            batchnorm_weight: Value to initialize BatchNorm weights to.

        Returns:
            Self for method chaining.
        """
        def init_gan_module(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                WeightInitializer.normal(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    WeightInitializer.zeros(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.affine:
                    WeightInitializer.normal(module.weight, mean=batchnorm_weight, std=std)
                    WeightInitializer.zeros(module.bias)

        self.apply(init_gan_module)
        return self

    def init_custom(
        self,
        init_fn: Callable[[nn.Module], None]
    ) -> 'WeightInitMixin':
        """
        Apply a custom initialization function to all submodules.

        Args:
            init_fn: A function that takes a module and initializes it.

        Returns:
            Self for method chaining.
        """
        self.apply(init_fn)
        return self

    def reset_parameters(self) -> 'WeightInitMixin':
        """
        Reset all parameters to their initial values.

        This applies the default initialization defined by _init_weights.

        Returns:
            Self for method chaining.
        """
        self.apply(self._init_weights)
        return self


# Convenience aliases for backward compatibility
def initialize_weights(
    module: nn.Module,
    method: str = 'kaiming_normal',
    nonlinearity: str = 'relu',
    scale: float = 1.0
) -> None:
    """
    Legacy function for weight initialization.

    This function is provided for backward compatibility. New code should
    use WeightInitializer.initialize_module() or WeightInitMixin instead.

    Args:
        module: The module to initialize.
        method: Initialization method ('kaiming_normal', 'xavier_normal', 'orthogonal').
        nonlinearity: Non-linearity for Kaiming initialization.
        scale: Scale factor applied to initialized weights.
    """
    method_map = {
        'kaiming_normal': InitMethod.KAIMING_NORMAL,
        'xavier_normal': InitMethod.XAVIER_NORMAL,
        'orthogonal': InitMethod.ORTHOGONAL,
    }

    init_method = method_map.get(method, InitMethod.KAIMING_NORMAL)

    for name, param in module.named_parameters():
        if 'weight' in name:
            WeightInitializer._init_weight(param, init_method, nonlinearity=nonlinearity)
            param.data *= scale
        elif 'bias' in name:
            WeightInitializer.zeros(param)

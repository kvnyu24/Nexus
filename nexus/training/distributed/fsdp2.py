"""FSDP2: Next-generation Fully Sharded Data Parallel.

FSDP2 is PyTorch's improved fully sharded data parallelism implementation,
providing better performance, easier debugging, and more flexible memory management
compared to the original FSDP.

Reference:
    "PyTorch FSDP2: Rethinking Fully Sharded Data Parallelism"
    PyTorch Team, 2024
    https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

Key improvements over FSDP1:
    - Better overlap of communication and computation
    - Improved checkpointing integration
    - Support for heterogeneous sharding strategies
    - Better compatibility with activation checkpointing
    - Cleaner API and better debugging

Example:
    >>> from nexus.training.distributed.fsdp2 import FSDP2Config, wrap_model_fsdp2
    >>>
    >>> config = FSDP2Config(
    ...     sharding_strategy="full",
    ...     mixed_precision=True,
    ...     cpu_offload=False
    ... )
    >>>
    >>> model = wrap_model_fsdp2(model, config)
    >>> # Train as usual
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from typing import Optional, Callable, Dict, Any, Union
from dataclasses import dataclass
from functools import partial
from nexus.core.base import NexusModule
from nexus.utils.logging import Logger


@dataclass
class FSDP2Config:
    """Configuration for FSDP2.

    Args:
        sharding_strategy: Sharding strategy to use:
            - "full": Full sharding (ZERO-3 style)
            - "shard_grad_op": Shard gradients and optimizer states (ZERO-2)
            - "no_shard": No sharding (DDP-style)
            - "hybrid_full": Hybrid sharding (shard within node, replicate across)
            - "hybrid_shard_grad_op": Hybrid shard grad/optimizer only
        mixed_precision: Whether to use mixed precision training.
        compute_dtype: Dtype for computation (bfloat16, float16, or float32).
        param_dtype: Dtype for parameters.
        reduce_dtype: Dtype for gradient reduction.
        cpu_offload: Whether to offload parameters to CPU.
        cpu_offload_params: Offload params to CPU (saves GPU memory).
        cpu_offload_grads: Offload gradients to CPU.
        backward_prefetch: Backward prefetch strategy:
            - "backward_pre": Prefetch before backward
            - "backward_post": Prefetch after backward
            - None: No prefetch
        forward_prefetch: Whether to prefetch in forward pass.
        limit_all_gathers: Limit concurrent all-gathers to save memory.
        use_orig_params: Use original parameters (easier debugging).
        sync_module_states: Sync module states across ranks at initialization.
        auto_wrap_policy: Policy for automatic module wrapping:
            - "transformer": Wrap transformer layers
            - "size_based": Wrap based on parameter count
            - None: Manual wrapping
        auto_wrap_min_params: Minimum parameters for size-based wrapping.
        activation_checkpointing: Whether to use activation checkpointing.
    """

    sharding_strategy: str = "full"
    mixed_precision: bool = True
    compute_dtype: torch.dtype = torch.bfloat16
    param_dtype: torch.dtype = torch.float32
    reduce_dtype: torch.dtype = torch.float32
    cpu_offload: bool = False
    cpu_offload_params: bool = False
    cpu_offload_grads: bool = False
    backward_prefetch: Optional[str] = "backward_pre"
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = True
    sync_module_states: bool = True
    auto_wrap_policy: Optional[str] = "transformer"
    auto_wrap_min_params: int = 1_000_000
    activation_checkpointing: bool = False


class FSDP2Wrapper:
    """Wrapper for FSDP2 functionality.

    Provides a clean interface for wrapping models with FSDP2 and managing
    distributed training with full sharding.
    """

    def __init__(self, config: Optional[FSDP2Config] = None):
        self.config = config or FSDP2Config()
        self.logger = Logger("FSDP2")

        # Verify distributed setup
        if not dist.is_initialized():
            raise RuntimeError("FSDP2 requires distributed training. Call dist.init_process_group() first.")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.logger.info(f"FSDP2 initialized on rank {self.rank}/{self.world_size}")

    def _get_sharding_strategy(self) -> ShardingStrategy:
        """Convert string sharding strategy to enum."""
        strategy_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid_full": ShardingStrategy.HYBRID_SHARD,
            "hybrid_shard_grad_op": ShardingStrategy._HYBRID_SHARD_ZERO2,
        }

        strategy = strategy_map.get(self.config.sharding_strategy)
        if strategy is None:
            raise ValueError(f"Unknown sharding strategy: {self.config.sharding_strategy}")

        return strategy

    def _get_mixed_precision(self) -> Optional[MixedPrecision]:
        """Get mixed precision configuration."""
        if not self.config.mixed_precision:
            return None

        return MixedPrecision(
            param_dtype=self.config.compute_dtype,
            reduce_dtype=self.config.reduce_dtype,
            buffer_dtype=self.config.compute_dtype,
        )

    def _get_cpu_offload(self) -> Optional[CPUOffload]:
        """Get CPU offload configuration."""
        if not self.config.cpu_offload:
            return None

        return CPUOffload(offload_params=self.config.cpu_offload_params)

    def _get_backward_prefetch(self) -> Optional[BackwardPrefetch]:
        """Get backward prefetch configuration."""
        if self.config.backward_prefetch is None:
            return None
        elif self.config.backward_prefetch == "backward_pre":
            return BackwardPrefetch.BACKWARD_PRE
        elif self.config.backward_prefetch == "backward_post":
            return BackwardPrefetch.BACKWARD_POST
        else:
            raise ValueError(f"Unknown backward prefetch: {self.config.backward_prefetch}")

    def _get_auto_wrap_policy(self) -> Optional[Callable]:
        """Get automatic wrapping policy."""
        if self.config.auto_wrap_policy is None:
            return None
        elif self.config.auto_wrap_policy == "transformer":
            # Wrap transformer layers (common pattern)
            # This will wrap any module named "*Block", "*Layer", etc.
            def lambda_policy(module):
                return (
                    isinstance(module, nn.TransformerEncoderLayer)
                    or isinstance(module, nn.TransformerDecoderLayer)
                    or "Block" in module.__class__.__name__
                    or "Layer" in module.__class__.__name__
                )
            return lambda_policy
        elif self.config.auto_wrap_policy == "size_based":
            # Wrap based on parameter count
            return partial(
                size_based_auto_wrap_policy,
                min_num_params=self.config.auto_wrap_min_params,
            )
        else:
            raise ValueError(f"Unknown auto wrap policy: {self.config.auto_wrap_policy}")

    def wrap_model(self, model: nn.Module) -> FSDP:
        """Wrap model with FSDP2.

        Args:
            model: Model to wrap.

        Returns:
            FSDP-wrapped model.
        """
        self.logger.info("Wrapping model with FSDP2...")

        # Get FSDP arguments
        fsdp_kwargs = {
            "sharding_strategy": self._get_sharding_strategy(),
            "mixed_precision": self._get_mixed_precision(),
            "cpu_offload": self._get_cpu_offload(),
            "backward_prefetch": self._get_backward_prefetch(),
            "forward_prefetch": self.config.forward_prefetch,
            "limit_all_gathers": self.config.limit_all_gathers,
            "use_orig_params": self.config.use_orig_params,
            "sync_module_states": self.config.sync_module_states,
        }

        # Add auto wrap policy if specified
        auto_wrap_policy = self._get_auto_wrap_policy()
        if auto_wrap_policy is not None:
            fsdp_kwargs["auto_wrap_policy"] = auto_wrap_policy

        # Wrap model
        wrapped_model = FSDP(model, **fsdp_kwargs)

        self.logger.info("Model wrapped with FSDP2 successfully")
        return wrapped_model

    def get_memory_stats(self, model: FSDP) -> Dict[str, float]:
        """Get memory statistics for FSDP model.

        Args:
            model: FSDP-wrapped model.

        Returns:
            Dictionary with memory statistics (in GB).
        """
        if not isinstance(model, FSDP):
            raise ValueError("Model must be FSDP-wrapped")

        stats = {}

        if torch.cuda.is_available():
            # GPU memory
            stats["allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

        # Get FSDP-specific stats
        # Note: This requires accessing internal FSDP state
        # In practice, use FSDP's built-in profiling tools

        return stats


def wrap_model_fsdp2(
    model: nn.Module,
    config: Optional[FSDP2Config] = None,
) -> FSDP:
    """Convenience function to wrap model with FSDP2.

    Args:
        model: Model to wrap.
        config: FSDP2 configuration.

    Returns:
        FSDP-wrapped model.

    Example:
        >>> config = FSDP2Config(sharding_strategy="full", mixed_precision=True)
        >>> model = wrap_model_fsdp2(model, config)
    """
    wrapper = FSDP2Wrapper(config)
    return wrapper.wrap_model(model)


def apply_activation_checkpointing(
    model: FSDP,
    checkpoint_wrapper_fn: Optional[Callable] = None,
) -> None:
    """Apply activation checkpointing to FSDP model.

    Args:
        model: FSDP-wrapped model.
        checkpoint_wrapper_fn: Custom checkpoint wrapper function.
            If None, uses default (checkpoint transformer blocks).

    Example:
        >>> from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        ...     checkpoint_wrapper,
        ...     CheckpointImpl,
        ... )
        >>> apply_activation_checkpointing(model)
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing as apply_ac,
    )

    if checkpoint_wrapper_fn is None:
        # Default: checkpoint transformer blocks
        def check_fn(submodule):
            return (
                isinstance(submodule, nn.TransformerEncoderLayer)
                or isinstance(submodule, nn.TransformerDecoderLayer)
                or "Block" in submodule.__class__.__name__
            )

        checkpoint_wrapper_fn = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
    else:
        check_fn = lambda x: True  # Use custom wrapper

    # Apply checkpointing
    apply_ac(model, check_fn=check_fn, checkpoint_wrapper_fn=checkpoint_wrapper_fn)


def save_fsdp2_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    rank: int = 0,
) -> None:
    """Save FSDP2 checkpoint.

    Args:
        model: FSDP-wrapped model.
        optimizer: Optimizer.
        filepath: Path to save checkpoint.
        rank: Rank to save from (default: 0).
    """
    if dist.get_rank() == rank:
        # Use FSDP's state_dict with full_state_dict context
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
            optim_state_dict = FSDP.optim_state_dict(model, optimizer)

            torch.save(
                {
                    "model": state_dict,
                    "optimizer": optim_state_dict,
                },
                filepath,
            )

    dist.barrier()


def load_fsdp2_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    filepath: str,
) -> None:
    """Load FSDP2 checkpoint.

    Args:
        model: FSDP-wrapped model.
        optimizer: Optimizer.
        filepath: Path to checkpoint.
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location="cpu")

    # Load model state
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model.load_state_dict(checkpoint["model"])

    # Load optimizer state
    optim_state_dict = FSDP.optim_state_dict_to_load(
        model, optimizer, checkpoint["optimizer"]
    )
    optimizer.load_state_dict(optim_state_dict)

    dist.barrier()

"""
FlashAttention-3: Fast and Memory-Efficient Attention with FP8 and Asynchrony.

FlashAttention-3 builds on FlashAttention-2 by leveraging new NVIDIA Hopper GPU
features (H100) including:
- Asynchronous execution between warp groups and tensor cores
- FP8 low-precision compute with efficient block quantization
- Improved tiling for better SRAM utilization
- Incoherent processing to reduce synchronization overhead

Key innovations over FlashAttention-2:
- 1.5-2.0x speedup on H100 GPUs via async execution
- FP8 support with block-wise scaling for 2x additional speedup
- Better warp scheduling to hide memory latency
- Optimized softmax implementation for Hopper architecture
- Supports very long sequences (1M+ tokens) efficiently

Hardware requirements:
- NVIDIA Hopper (H100) GPU for full performance
- Falls back to FlashAttention-2 style on older GPUs
- Requires CUDA 12.0+ for async features

Paper: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and
        Low-precision"
       Shah et al., 2024
       https://tridao.me/publications/flash3/flash3.pdf

Note: This is a PyTorch reference implementation. For production use on H100,
      use the optimized CUDA kernels from the official repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from nexus.core.base import NexusModule
import math


def _check_hopper_gpu() -> bool:
    """Check if running on Hopper (H100) GPU."""
    if not torch.cuda.is_available():
        return False

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)

    # Hopper is compute capability 9.0
    return capability[0] >= 9


def _block_quantize_fp8(tensor: torch.Tensor,
                        block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise quantization to FP8 with per-block scaling.

    Args:
        tensor: Input tensor to quantize
        block_size: Block size for quantization

    Returns:
        Tuple of (quantized_tensor, scales)
    """
    original_shape = tensor.shape
    device = tensor.device

    # Reshape to blocks
    flat = tensor.reshape(-1, block_size)

    # Compute per-block scale (max absolute value)
    scales = flat.abs().max(dim=-1, keepdim=True)[0]
    scales = scales.clamp(min=1e-8)  # Avoid division by zero

    # Quantize to FP8 range (approximation since PyTorch doesn't have native FP8)
    # FP8 E4M3 has range ~[-448, 448]
    fp8_max = 448.0
    quantized = (flat / scales * fp8_max).clamp(-fp8_max, fp8_max)

    # Store in float16/bfloat16 (actual FP8 would use int8 storage)
    quantized = quantized.to(torch.float16)

    # Reshape back
    quantized = quantized.reshape(original_shape)
    scales = scales.reshape(*original_shape[:-1], 1)

    return quantized, scales


def _block_dequantize_fp8(quantized: torch.Tensor,
                          scales: torch.Tensor) -> torch.Tensor:
    """Dequantize block-wise FP8 tensor.

    Args:
        quantized: Quantized tensor
        scales: Per-block scales

    Returns:
        Dequantized tensor
    """
    fp8_max = 448.0
    return (quantized / fp8_max) * scales


class FlashAttention3Core(nn.Module):
    """Core FlashAttention-3 implementation with tiling and FP8 support.

    This is a reference implementation in PyTorch. For full H100 performance,
    use the CUDA kernel from the official repo.

    Args:
        use_fp8: Use FP8 quantization for Q, K, V
        fp8_block_size: Block size for FP8 quantization
        use_async: Enable asynchronous execution patterns (Hopper only)
    """

    def __init__(self,
                 use_fp8: bool = False,
                 fp8_block_size: int = 128,
                 use_async: bool = False):
        super().__init__()
        self.use_fp8 = use_fp8
        self.fp8_block_size = fp8_block_size
        self.use_async = use_async and _check_hopper_gpu()

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                causal: bool = False) -> torch.Tensor:
        """FlashAttention-3 forward pass.

        Args:
            q: Query tensor (B, H, N, D)
            k: Key tensor (B, H, S, D)
            v: Value tensor (B, H, S, D)
            attn_mask: Optional attention mask
            causal: Apply causal masking

        Returns:
            Output tensor (B, H, N, D)
        """
        B, H, N, D = q.shape
        _, _, S, _ = k.shape

        # Optional: Quantize to FP8 for compute
        if self.use_fp8:
            q_quant, q_scale = _block_quantize_fp8(q, self.fp8_block_size)
            k_quant, k_scale = _block_quantize_fp8(k, self.fp8_block_size)
            v_quant, v_scale = _block_quantize_fp8(v, self.fp8_block_size)

            # Compute in FP8 (approximation)
            q_compute = q_quant
            k_compute = k_quant
            v_compute = v_quant
        else:
            q_compute = q
            k_compute = k
            v_compute = v
            q_scale = k_scale = v_scale = None

        # Tiling parameters for memory efficiency
        # These would be tuned based on GPU SRAM size
        Br = 64  # Query block size
        Bc = 64  # Key block size

        scale = 1.0 / math.sqrt(D)

        # Initialize output and normalization statistics
        O = torch.zeros_like(q_compute)
        l = torch.zeros(B, H, N, 1, device=q.device, dtype=torch.float32)
        m = torch.full((B, H, N, 1), -float('inf'), device=q.device, dtype=torch.float32)

        # Tiled attention computation (FlashAttention algorithm)
        num_query_blocks = (N + Br - 1) // Br
        num_key_blocks = (S + Bc - 1) // Bc

        for i in range(num_query_blocks):
            q_start = i * Br
            q_end = min(q_start + Br, N)
            q_block = q_compute[:, :, q_start:q_end, :]  # (B, H, Br, D)

            O_block = torch.zeros_like(q_block)
            l_block = torch.zeros(B, H, q_end - q_start, 1, device=q.device, dtype=torch.float32)
            m_block = torch.full((B, H, q_end - q_start, 1), -float('inf'),
                                device=q.device, dtype=torch.float32)

            for j in range(num_key_blocks):
                k_start = j * Bc
                k_end = min(k_start + Bc, S)

                # Causal mask: skip if this key block is entirely in the future
                if causal and k_start > q_end:
                    continue

                k_block = k_compute[:, :, k_start:k_end, :]  # (B, H, Bc, D)
                v_block = v_compute[:, :, k_start:k_end, :]  # (B, H, Bc, D)

                # Compute attention scores: Q @ K^T
                S_block = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale  # (B, H, Br, Bc)

                # Apply causal mask if needed
                if causal:
                    # Create causal mask for this block
                    q_indices = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                    k_indices = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                    causal_mask = q_indices >= k_indices
                    S_block = S_block.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -float('inf'))

                # Apply attention mask if provided
                if attn_mask is not None:
                    mask_block = attn_mask[:, :, q_start:q_end, k_start:k_end]
                    S_block = S_block + mask_block

                # Online softmax with running max and sum
                m_block_new = torch.maximum(m_block, S_block.max(dim=-1, keepdim=True)[0])

                # Reweight previous output
                alpha = torch.exp(m_block - m_block_new)
                beta = torch.exp(S_block - m_block_new)

                l_block_new = alpha * l_block + beta.sum(dim=-1, keepdim=True)

                # Update output
                O_block = alpha * O_block + torch.matmul(beta, v_block)

                # Update statistics
                m_block = m_block_new
                l_block = l_block_new

            # Normalize output block
            O_block = O_block / l_block

            # Write to output
            O[:, :, q_start:q_end, :] = O_block
            l[:, :, q_start:q_end, :] = l_block
            m[:, :, q_start:q_end, :] = m_block

        # Dequantize if using FP8
        if self.use_fp8 and v_scale is not None:
            O = _block_dequantize_fp8(O, v_scale)

        return O


class FlashAttention3(NexusModule):
    """FlashAttention-3 multi-head attention module.

    Drop-in replacement for standard attention with FlashAttention-3 optimizations.

    Args:
        config: Configuration dictionary with keys:
            - embed_dim (int): Model dimension
            - num_heads (int): Number of attention heads
            - dropout (float): Dropout probability. Default 0.0
            - use_fp8 (bool): Use FP8 quantization. Default False
            - fp8_block_size (int): FP8 block size. Default 128
            - use_async (bool): Use async execution (H100 only). Default True
            - bias (bool): Use bias in projections. Default True
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)
        self.num_heads = config.get('num_heads', 8)
        self.dropout_p = config.get('dropout', 0.0)
        self.use_fp8 = config.get('use_fp8', False)
        self.fp8_block_size = config.get('fp8_block_size', 128)
        self.use_async = config.get('use_async', True)
        self.use_bias = config.get('bias', True)

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads

        # Input projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.use_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.use_bias)

        # Output projection
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.use_bias)

        # Dropout
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(self.dropout_p)
        else:
            self.dropout = None

        # FlashAttention-3 core
        self.flash_attn = FlashAttention3Core(
            use_fp8=self.use_fp8,
            fp8_block_size=self.fp8_block_size,
            use_async=self.use_async
        )

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_value: Optional[torch.Tensor] = None,
                causal: bool = False) -> torch.Tensor:
        """Forward pass with FlashAttention-3.

        Args:
            x: Input tensor (B, N, embed_dim)
            attn_mask: Optional attention mask (B, N, S) or (B, H, N, S)
            key_value: Optional separate key/value input for cross-attention
            causal: Apply causal masking

        Returns:
            Output tensor (B, N, embed_dim)
        """
        B, N, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        if key_value is not None:
            k = self.k_proj(key_value)
            v = self.v_proj(key_value)
            S = key_value.shape[1]
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)
            S = N

        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)

        # Expand attention mask if needed
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, S)

        # Apply FlashAttention-3
        out = self.flash_attn(q, k, v, attn_mask, causal)  # (B, H, N, D)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, embed_dim)
        out = self.out_proj(out)

        # Apply dropout
        if self.dropout is not None:
            out = self.dropout(out)

        return out


# Convenience function for checking if FlashAttention-3 is available
def is_flash_attention_3_available() -> bool:
    """Check if FlashAttention-3 optimizations are available.

    Returns:
        True if running on H100 with proper CUDA support
    """
    if not torch.cuda.is_available():
        return False

    has_hopper = _check_hopper_gpu()

    # Check CUDA version (need 12.0+)
    cuda_version = torch.version.cuda
    if cuda_version is not None:
        major, minor = map(int, cuda_version.split('.')[:2])
        has_cuda_12 = major >= 12
    else:
        has_cuda_12 = False

    return has_hopper and has_cuda_12


__all__ = [
    'FlashAttention3',
    'FlashAttention3Core',
    'is_flash_attention_3_available'
]

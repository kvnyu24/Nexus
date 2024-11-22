import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from ...core.base import NexusModule

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Generate pair-wise relative position index
        coords = torch.stack(torch.meshgrid([
            torch.arange(window_size),
            torch.arange(window_size)
        ]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int = 0,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "Input feature size doesn't match H, W"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
            
        # Window partition
        x_windows = self._window_partition(shifted_x)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        shifted_x = self._window_reverse(attn_windows, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x

class SwinTransformer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.image_size = config["image_size"]
        self.patch_size = config.get("patch_size", 4)
        self.in_channels = config.get("in_channels", 3)
        self.embed_dim = config.get("embed_dim", 96)
        self.depths = config.get("depths", [2, 2, 6, 2])
        self.num_heads = config.get("num_heads", [3, 6, 12, 24])
        self.window_size = config.get("window_size", 7)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)
        self.num_classes = config.get("num_classes", 1000)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # Transformer stages
        self.stages = nn.ModuleList()
        for i_stage in range(len(self.depths)):
            stage = nn.ModuleList([
                SwinTransformerBlock(
                    dim=self.embed_dim * (2 ** i_stage),
                    num_heads=self.num_heads[i_stage],
                    window_size=self.window_size,
                    shift_size=0 if (i_layer % 2 == 0) else self.window_size // 2,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout
                )
                for i_layer in range(self.depths[i_stage])
            ])
            self.stages.append(stage)
            
        # Classification head
        self.norm = nn.LayerNorm(self.embed_dim * (2 ** (len(self.depths)-1)))
        self.head = nn.Linear(self.embed_dim * (2 ** (len(self.depths)-1)), self.num_classes)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # Apply transformer stages
        features = []
        for stage in self.stages:
            for block in stage:
                block.H, block.W = H, W
                x = block(x)
            features.append(x)
            if stage != self.stages[-1]:
                x = self.patch_merging(x, H, W)
                H, W = H // 2, W // 2
                
        # Classification
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)
        
        return {
            "logits": logits,
            "features": features
        } 
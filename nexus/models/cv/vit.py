import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...components.attention import UnifiedAttention
from ...core.base import NexusModule

class TransformerBlock(NexusModule):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_flash_attention: bool = False,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        
        # Multi-head self attention with optional flash attention
        self.attention = UnifiedAttention(
            hidden_size=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention
        )
        self.norm1 = nn.LayerNorm(dim)
        
        # MLP block with improved architecture
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
        
        # Layer scale parameters
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.attention(self.norm1(x))
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x

class PatchEmbedding(NexusModule):
    def __init__(
        self, 
        image_size: int, 
        patch_size: int, 
        in_channels: int, 
        embed_dim: int,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size}*{self.image_size})"
        
        x = self.projection(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        x = self.norm(x)
        return x

class VisionTransformer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        use_flash_attention = config.get("use_flash_attention", False)
        self.distillation = config.get("distillation", False)
        self.layer_scale_init_value = config.get("layer_scale_init_value", 1e-6)
        
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.in_channels = config["in_channels"]
        self.num_classes = config["num_classes"]
        self.embed_dim = config["embed_dim"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.mlp_ratio = config["mlp_ratio"]
        self.dropout = config["dropout"]
        
        # Patch embedding with normalization
        self.patch_embed = PatchEmbedding(
            self.image_size, self.patch_size,
            self.in_channels, self.embed_dim,
            norm_layer=nn.LayerNorm
        )
        
        # Position embedding with interpolation support
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        if self.distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        self.pos_drop = nn.Dropout(p=self.dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                use_flash_attention=use_flash_attention,
                layer_scale_init_value=self.layer_scale_init_value
            ) for _ in range(self.num_layers)
        ])
        
        # Classification head(s)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        
        if self.distillation:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes)
            
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.distillation:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(N ** 0.5), int(N ** 0.5), dim).permute(0, 3, 1, 2),
            size=(h, w),
            mode='bicubic',
            align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((self.pos_embed[:, :1], pos_embed), dim=1)
        
    def forward(self, image: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Ensure input is float32 for MPS compatibility
        if image.device.type == 'mps':
            image = image.to(torch.float32)
        
        B = image.shape[0]
        
        # Patch embedding
        x = self.patch_embed(image)
        
        # Add cls token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            
        # Add position embedding with interpolation
        x = x + self.interpolate_pos_encoding(x, self.patch_embed.grid_size, self.patch_embed.grid_size)
        x = self.pos_drop(x)
        
        # Apply transformer layers
        features = []
        for layer in self.transformer_layers:
            x = layer(x)
            features.append(x)
        
        # Classification
        x = self.norm(x)
        
        if self.distillation:
            cls_output = self.head(x[:, 0])
            dist_output = self.head_dist(x[:, 1])
            # During inference, return the average of both classifier predictions
            if not self.training:
                return {
                    "logits": (cls_output + dist_output) / 2,
                    "embeddings": x[:, 0],
                    "features": features
                }
            else:
                return {
                    "logits": cls_output,
                    "dist_logits": dist_output,
                    "embeddings": x[:, 0],
                    "features": features
                }
        else:
            return {
                "logits": self.head(x[:, 0]),
                "embeddings": x[:, 0],
                "features": features
            }
"""
SAM: Segment Anything Model

Implementation of the Segment Anything Model (SAM), a promptable segmentation
model that can generate high-quality masks for any object given various types
of input prompts (points, boxes, masks, or text).

Reference:
    Kirillov, A., Mintun, E., Ravi, N., et al. (2023).
    "Segment Anything."
    arXiv:2304.02643

Key Components:
    - ImageEncoder: ViT-based encoder producing image embeddings with windowed attention
    - PromptEncoder: Encodes points, boxes, masks, and text prompts into embeddings
    - MaskDecoder: Lightweight transformer decoder producing masks and IoU scores
    - SAM: Full model combining encoder, prompt encoder, and mask decoder

Architecture Details:
    - Image encoder runs once per image, amortizing cost across prompts
    - Prompt encoder uses learned positional embeddings for geometric prompts
    - Mask decoder uses two-way attention between image and prompt tokens
    - Supports multiple mask outputs with IoU-based ranking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, List, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class ImageEncoder(WeightInitMixin, NexusModule):
    """ViT-based image encoder for SAM.

    Encodes input images into dense feature maps using a Vision Transformer
    with windowed attention for efficiency. The output is a spatially downsampled
    feature map suitable for the mask decoder.

    Args:
        config: Configuration dictionary with keys:
            img_size (int): Input image resolution. Default: 1024.
            patch_size (int): Patch size. Default: 16.
            in_channels (int): Input channels. Default: 3.
            encoder_embed_dim (int): Embedding dimension. Default: 768.
            encoder_depth (int): Number of transformer layers. Default: 12.
            encoder_num_heads (int): Number of attention heads. Default: 12.
            mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
            dropout (float): Dropout rate. Default: 0.0.
            window_size (int): Window size for windowed attention. Default: 14.
            out_channels (int): Output channel dimension after neck. Default: 256.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 1024)
        self.patch_size = config.get("patch_size", 16)
        self.in_channels = config.get("in_channels", 3)
        self.embed_dim = config.get("encoder_embed_dim", 768)
        self.depth = config.get("encoder_depth", 12)
        self.num_heads = config.get("encoder_num_heads", 12)
        self.mlp_ratio = config.get("mlp_ratio", 4.0)
        self.dropout = config.get("dropout", 0.0)
        self.window_size = config.get("window_size", 14)
        self.out_channels = config.get("out_channels", 256)

        self.grid_size = self.img_size // self.patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.in_channels, self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

        # Absolute positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size, self.grid_size, self.embed_dim)
        )

        # Transformer blocks with windowed and global attention
        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            use_global = (i + 1) % 4 == 0  # Global attention every 4th block
            self.blocks.append(
                SAMBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    window_size=0 if use_global else self.window_size,
                    input_size=(self.grid_size, self.grid_size),
                )
            )

        # Neck to reduce channel dimension
        self.neck = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=1, bias=False),
            nn.LayerNorm([self.out_channels, self.grid_size, self.grid_size]),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([self.out_channels, self.grid_size, self.grid_size]),
        )

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.init_weights_vit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image into feature map.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Image embeddings of shape (B, out_channels, H/patch_size, W/patch_size).
        """
        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H', W')
        x = self.patch_embed(x)
        # Reshape to (B, H', W', embed_dim) for transformer
        x = x.permute(0, 2, 3, 1)
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Reshape to (B, embed_dim, H', W') for convolution neck
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)

        return x


class SAMBlock(NexusModule):
    """Transformer block for SAM image encoder with optional windowed attention.

    Supports both windowed attention (for efficiency) and global attention
    (for cross-window information exchange). Uses pre-normalization and
    relative position bias.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim expansion ratio.
        dropout: Dropout rate.
        window_size: Window size for windowed attention (0 for global).
        input_size: Spatial size of the input feature map (H, W).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        window_size: int = 0,
        input_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.input_size = input_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SAMAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (B, H, W, C).

        Returns:
            Output of shape (B, H, W, C).
        """
        shortcut = x
        x = self.norm1(x)

        # Windowed attention if applicable
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = self._window_partition(x)

        x = self.attn(x)

        if self.window_size > 0:
            x = self._window_unpartition(x, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _window_partition(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Partition input into non-overlapping windows.

        Args:
            x: Input of shape (B, H, W, C).

        Returns:
            Tuple of (windowed tensor of shape (B*nW, ws, ws, C), padded size).
        """
        B, H, W, C = x.shape
        ws = self.window_size

        # Pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # Reshape into windows
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, ws, ws, C)

        return x, (Hp, Wp)

    def _window_unpartition(
        self,
        x: torch.Tensor,
        pad_hw: Tuple[int, int],
        orig_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Reverse window partition.

        Args:
            x: Windowed tensor of shape (B*nW, ws, ws, C).
            pad_hw: Padded spatial size (Hp, Wp).
            orig_hw: Original spatial size (H, W).

        Returns:
            Unpartitioned tensor of shape (B, H, W, C).
        """
        Hp, Wp = pad_hw
        H, W = orig_hw
        ws = self.window_size
        B = x.shape[0] // ((Hp // ws) * (Wp // ws))

        x = x.view(B, Hp // ws, Wp // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]
        return x


class SAMAttention(NexusModule):
    """Multi-head attention with decomposed relative positional encoding.

    Uses separate relative position encodings for horizontal and vertical
    axes, which is more parameter-efficient than full 2D position encoding.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        input_size: Spatial size of the input for relative position bias.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        input_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Decomposed relative position bias
        self.rel_pos_h = nn.Parameter(
            torch.zeros(2 * input_size[0] - 1, self.head_dim)
        )
        self.rel_pos_w = nn.Parameter(
            torch.zeros(2 * input_size[1] - 1, self.head_dim)
        )
        nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
        nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Output tensor of shape (B, H, W, C).
        """
        B, H, W, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Attention with relative position bias
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self._add_decomposed_rel_pos(attn, q, H, W)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, H, W, C)

        return x

    def _add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        q: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Add decomposed relative position bias to attention weights.

        Args:
            attn: Attention weights of shape (B, heads, H*W, H*W).
            q: Query tensor of shape (B, heads, H*W, head_dim).
            H: Spatial height.
            W: Spatial width.

        Returns:
            Attention weights with added relative position bias.
        """
        # Compute relative position indices
        coords_h = torch.arange(H, device=q.device)
        coords_w = torch.arange(W, device=q.device)

        rel_h = coords_h[:, None] - coords_h[None, :] + (self.input_size[0] - 1)
        rel_w = coords_w[:, None] - coords_w[None, :] + (self.input_size[1] - 1)

        rel_h = rel_h.clamp(0, 2 * self.input_size[0] - 2)
        rel_w = rel_w.clamp(0, 2 * self.input_size[1] - 2)

        # Compute relative position attention
        Rh = self.rel_pos_h[rel_h.long()]  # (H, H, head_dim)
        Rw = self.rel_pos_w[rel_w.long()]  # (W, W, head_dim)

        B, nH, N, _ = q.shape
        q_h = q.reshape(B, nH, H, W, self.head_dim)

        # Height component: (B, nH, H, W, head_dim) x (H, H, head_dim) -> (B, nH, H, W, H)
        rel_h_attn = torch.einsum("bnhwc,hpc->bnhwp", q_h, Rh)
        # Width component
        rel_w_attn = torch.einsum("bnhwc,wpc->bnhwp", q_h, Rw)

        # Expand and add to attention
        # rel_h: (B, nH, H, W, H) -> broadcast over W for target
        # rel_w: (B, nH, H, W, W) -> broadcast over H for target
        attn = attn.reshape(B, nH, H, W, H, W)
        attn = attn + rel_h_attn.unsqueeze(-1) + rel_w_attn.unsqueeze(-2)
        attn = attn.reshape(B, nH, H * W, H * W)

        return attn


class PromptEncoder(NexusModule):
    """Encodes various prompts (points, boxes, masks, text) into embeddings.

    Supports multiple prompt types:
    - Points: encoded as positional embeddings with foreground/background labels
    - Boxes: encoded as two corner points (top-left, bottom-right)
    - Masks: downsampled and projected via small ConvNet
    - Text: projected through a linear layer (for open-vocabulary variants)

    The output sparse embeddings (from points/boxes/text) and dense embeddings
    (from masks) are provided to the mask decoder.

    Args:
        config: Configuration dictionary with keys:
            embed_dim (int): Embedding dimension. Default: 256.
            img_size (int): Input image resolution. Default: 1024.
            patch_size (int): Patch size (for mask downsampling). Default: 16.
            mask_in_channels (int): Number of channels in input masks. Default: 1.
            num_point_embeddings (int): Number of point label types. Default: 4.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.embed_dim = config.get("embed_dim", 256)
        self.img_size = config.get("img_size", 1024)
        self.patch_size = config.get("patch_size", 16)
        self.grid_size = self.img_size // self.patch_size
        self.mask_in_channels = config.get("mask_in_channels", 1)

        # Positional encoding for point and box prompts
        self.pe_layer = PositionalEncoding2D(self.embed_dim // 2)

        # Point embeddings (foreground, background, top-left corner, bottom-right corner)
        num_point_embeddings = config.get("num_point_embeddings", 4)
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, self.embed_dim) for _ in range(num_point_embeddings)
        ])

        # Not-a-point embedding (for padding)
        self.not_a_point_embed = nn.Embedding(1, self.embed_dim)

        # Mask downsampling network
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(self.mask_in_channels, self.embed_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm([self.embed_dim // 4, self.grid_size * 2, self.grid_size * 2]),
            nn.GELU(),
            nn.Conv2d(self.embed_dim // 4, self.embed_dim, kernel_size=2, stride=2),
            nn.LayerNorm([self.embed_dim, self.grid_size, self.grid_size]),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
        )

        # No-mask embedding (when no mask prompt is provided)
        self.no_mask_embed = nn.Embedding(1, self.embed_dim)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts into sparse and dense embeddings.

        Args:
            points: Tuple of (point_coords, point_labels).
                point_coords: (B, N, 2) in [0, img_size] coordinates.
                point_labels: (B, N) with 0=background, 1=foreground.
            boxes: Box coordinates (B, num_boxes, 4) as [x1, y1, x2, y2].
            masks: Mask prompts (B, 1, H, W) at full image resolution.

        Returns:
            Tuple of:
                sparse_embeddings: (B, N_sparse, embed_dim)
                dense_embeddings: (B, embed_dim, H/patch_size, W/patch_size)
        """
        B = self._get_batch_size(points, boxes, masks)
        device = self._get_device(points, boxes, masks)

        sparse_embeddings = torch.empty(B, 0, self.embed_dim, device=device)

        # Encode points
        if points is not None:
            point_coords, point_labels = points
            point_embeds = self._encode_points(point_coords, point_labels)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeds], dim=1)

        # Encode boxes
        if boxes is not None:
            box_embeds = self._encode_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeds], dim=1)

        # Encode masks
        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                B, -1, self.grid_size, self.grid_size
            )

        return sparse_embeddings, dense_embeddings

    def _encode_points(
        self,
        coords: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Encode point prompts.

        Args:
            coords: Point coordinates (B, N, 2) in pixel space.
            labels: Point labels (B, N), 0=background, 1=foreground.

        Returns:
            Point embeddings of shape (B, N, embed_dim).
        """
        # Normalize coordinates to [0, 1]
        coords = coords / self.img_size

        # Get positional encoding
        point_embeds = self.pe_layer.forward_with_coords(coords)

        # Add label embeddings
        # Label 0 -> background, Label 1 -> foreground
        for i in range(labels.shape[1]):
            mask_fg = (labels[:, i] == 1).unsqueeze(-1)
            mask_bg = (labels[:, i] == 0).unsqueeze(-1)
            label_embed = (
                mask_fg.float() * self.point_embeddings[1].weight
                + mask_bg.float() * self.point_embeddings[0].weight
            )
            point_embeds[:, i] = point_embeds[:, i] + label_embed.squeeze(1)

        return point_embeds

    def _encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode box prompts as corner point pairs.

        Args:
            boxes: Box coordinates (B, num_boxes, 4) as [x1, y1, x2, y2].

        Returns:
            Box embeddings of shape (B, num_boxes*2, embed_dim).
        """
        # Normalize coordinates
        boxes = boxes / self.img_size

        # Split into corner points
        corner1 = boxes[..., :2]  # top-left
        corner2 = boxes[..., 2:]  # bottom-right

        embed1 = self.pe_layer.forward_with_coords(corner1) + self.point_embeddings[2].weight
        embed2 = self.pe_layer.forward_with_coords(corner2) + self.point_embeddings[3].weight

        # Interleave corners
        B, N, D = embed1.shape
        box_embeds = torch.stack([embed1, embed2], dim=2).reshape(B, N * 2, D)
        return box_embeds

    def _get_batch_size(self, points, boxes, masks) -> int:
        if points is not None:
            return points[0].shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        return 1

    def _get_device(self, points, boxes, masks) -> torch.device:
        if points is not None:
            return points[0].device
        if boxes is not None:
            return boxes.device
        if masks is not None:
            return masks.device
        return torch.device("cpu")


class PositionalEncoding2D(nn.Module):
    """2D positional encoding using sinusoidal functions.

    Generates positional embeddings for 2D coordinates using a combination
    of sine and cosine functions at different frequencies.

    Args:
        num_pos_feats: Number of positional features per spatial dimension
            (total output dim is 2 * num_pos_feats).
        temperature: Temperature parameter controlling frequency range.
    """

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward_with_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding for given coordinates.

        Args:
            coords: Normalized coordinates in [0, 1] of shape (..., 2).

        Returns:
            Positional encodings of shape (..., 2 * num_pos_feats).
        """
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=coords.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = coords[..., 0:1] * 2 * math.pi / dim_t
        pos_y = coords[..., 1:2] * 2 * math.pi / dim_t

        pos_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)
        pos_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)

        return torch.cat([pos_x, pos_y], dim=-1)


class TwoWayAttentionBlock(NexusModule):
    """Two-way attention block for the SAM mask decoder.

    Performs self-attention on tokens, cross-attention from tokens to image,
    an MLP on tokens, and cross-attention from image to tokens. This enables
    bidirectional information flow between prompt tokens and image features.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension.
        dropout: Dropout rate.
        skip_first_layer_pe: Skip adding PE in the first cross-attention
            (used for the first block where tokens don't yet have context).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        dropout: float = 0.0,
        skip_first_layer_pe: bool = False,
    ):
        super().__init__()

        self.skip_first_layer_pe = skip_first_layer_pe

        # Self-attention on tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Cross-attention: tokens attend to image
        self.cross_attn_token_to_image = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Cross-attention: image attends to tokens
        self.cross_attn_image_to_token = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            queries: Token (prompt) embeddings (B, N_tokens, embed_dim).
            keys: Image embeddings (B, N_image, embed_dim).
            query_pe: Positional encoding for tokens (B, N_tokens, embed_dim).
            key_pe: Positional encoding for image (B, N_image, embed_dim).

        Returns:
            Tuple of (updated queries, updated keys).
        """
        # Self-attention on tokens
        if self.skip_first_layer_pe:
            q = k = queries
        else:
            q = k = queries + query_pe
        attn_out, _ = self.self_attn(q, k, queries)
        queries = self.norm1(queries + attn_out)

        # Cross-attention: tokens -> image
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_token_to_image(q, k, keys)
        queries = self.norm2(queries + attn_out)

        # MLP
        queries = self.norm3(queries + self.mlp(queries))

        # Cross-attention: image -> tokens
        q = keys + key_pe
        k = queries + query_pe
        attn_out, _ = self.cross_attn_image_to_token(q, k, queries)
        keys = self.norm4(keys + attn_out)

        return queries, keys


class MaskDecoder(WeightInitMixin, NexusModule):
    """Lightweight transformer decoder that produces masks and IoU scores.

    Takes image embeddings from the encoder and sparse/dense prompt embeddings
    from the prompt encoder, and produces segmentation masks with associated
    IoU (Intersection over Union) confidence scores.

    Args:
        config: Configuration dictionary with keys:
            decoder_dim (int): Decoder transformer dimension. Default: 256.
            decoder_num_heads (int): Number of attention heads. Default: 8.
            decoder_depth (int): Number of two-way attention blocks. Default: 2.
            decoder_mlp_dim (int): MLP hidden dimension. Default: 2048.
            num_mask_tokens (int): Number of mask output tokens. Default: 4.
            iou_head_depth (int): Number of MLP layers for IoU prediction. Default: 3.
            iou_head_hidden_dim (int): IoU MLP hidden dimension. Default: 256.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.decoder_dim = config.get("decoder_dim", 256)
        self.num_heads = config.get("decoder_num_heads", 8)
        self.depth = config.get("decoder_depth", 2)
        self.mlp_dim = config.get("decoder_mlp_dim", 2048)
        self.num_mask_tokens = config.get("num_mask_tokens", 4)

        # IoU prediction head config
        iou_head_depth = config.get("iou_head_depth", 3)
        iou_head_hidden_dim = config.get("iou_head_hidden_dim", 256)

        # Learnable mask tokens and IoU token
        self.iou_token = nn.Embedding(1, self.decoder_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.decoder_dim)

        # Two-way transformer decoder
        self.transformer_blocks = nn.ModuleList([
            TwoWayAttentionBlock(
                embed_dim=self.decoder_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                skip_first_layer_pe=(i == 0),
            )
            for i in range(self.depth)
        ])
        self.final_attn_token_to_image = nn.MultiheadAttention(
            self.decoder_dim, self.num_heads, batch_first=True,
        )
        self.final_norm = nn.LayerNorm(self.decoder_dim)

        # Mask prediction heads (one per mask token)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm([self.decoder_dim // 4]),
            nn.GELU(),
            nn.ConvTranspose2d(self.decoder_dim // 4, self.decoder_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.output_hypernetworks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.decoder_dim, self.decoder_dim),
                nn.GELU(),
                nn.Linear(self.decoder_dim, self.decoder_dim // 8),
            )
            for _ in range(self.num_mask_tokens)
        ])

        # IoU prediction head
        iou_layers = []
        for i in range(iou_head_depth - 1):
            in_dim = self.decoder_dim if i == 0 else iou_head_hidden_dim
            iou_layers.extend([nn.Linear(in_dim, iou_head_hidden_dim), nn.ReLU()])
        iou_layers.append(nn.Linear(iou_head_hidden_dim, self.num_mask_tokens))
        self.iou_prediction_head = nn.Sequential(*iou_layers)

        self.init_weights_vit()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict masks and IoU scores from embeddings.

        Args:
            image_embeddings: Encoder output (B, C, H, W).
            image_pe: Positional encoding for image (B, C, H, W).
            sparse_prompt_embeddings: From prompt encoder (B, N_sparse, C).
            dense_prompt_embeddings: From prompt encoder (B, C, H, W).

        Returns:
            Dictionary with:
                masks: Predicted masks (B, num_mask_tokens, 4*H, 4*W).
                iou_predictions: IoU scores (B, num_mask_tokens).
        """
        B, C, H, W = image_embeddings.shape

        # Prepare tokens: IoU token + mask tokens + prompt tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        ).unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # Combine image embeddings with dense prompt
        src = image_embeddings.flatten(2).transpose(1, 2)  # (B, H*W, C)
        src = src + dense_prompt_embeddings.flatten(2).transpose(1, 2)
        pos_src = image_pe.flatten(2).transpose(1, 2)

        # Token positional encoding
        token_pe = torch.zeros_like(tokens)

        # Run two-way transformer
        for block in self.transformer_blocks:
            tokens, src = block(tokens, src, token_pe, pos_src)

        # Final cross-attention from tokens to image
        q = tokens + token_pe
        k = src + pos_src
        attn_out, _ = self.final_attn_token_to_image(q, k, src)
        tokens = self.final_norm(tokens + attn_out)

        # Split output tokens
        iou_token_out = tokens[:, 0]
        mask_tokens_out = tokens[:, 1:1 + self.num_mask_tokens]

        # Upsample image features
        src = src.transpose(1, 2).view(B, C, H, W)
        upscaled = self.output_upscaling(src)

        # Generate masks via hypernetworks (dot product between tokens and upscaled features)
        masks = []
        for i in range(self.num_mask_tokens):
            hyper_in = mask_tokens_out[:, i]
            hyper_out = self.output_hypernetworks[i](hyper_in)  # (B, C//8)
            mask = torch.einsum("bc,bchw->bhw", hyper_out, upscaled)
            masks.append(mask)
        masks = torch.stack(masks, dim=1)  # (B, num_mask_tokens, H_up, W_up)

        # Predict IoU scores
        iou_predictions = self.iou_prediction_head(iou_token_out)

        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
        }


class SAM(WeightInitMixin, NexusModule):
    """Segment Anything Model (SAM).

    Full SAM model combining the image encoder, prompt encoder, and mask decoder.
    Designed for promptable segmentation: given an image and prompts (points,
    boxes, or masks), produces segmentation masks with quality scores.

    Config:
        img_size (int): Input image resolution. Default: 1024.
        patch_size (int): Patch size for image encoder. Default: 16.
        in_channels (int): Input channels. Default: 3.
        encoder_embed_dim (int): Image encoder embedding dim. Default: 768.
        encoder_depth (int): Image encoder depth. Default: 12.
        encoder_num_heads (int): Image encoder attention heads. Default: 12.
        decoder_dim (int): Mask decoder dimension. Default: 256.
        num_mask_tokens (int): Number of mask predictions. Default: 4.
        out_channels (int): Encoder output channels. Default: 256.

    Example:
        >>> config = {"img_size": 1024, "encoder_embed_dim": 768}
        >>> model = SAM(config)
        >>> images = torch.randn(1, 3, 1024, 1024)
        >>> points = (torch.tensor([[[512.0, 512.0]]]), torch.tensor([[1]]))
        >>> output = model(images, points=points)
        >>> output["masks"].shape
        torch.Size([1, 4, 256, 256])
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.img_size = config.get("img_size", 1024)
        self.out_channels = config.get("out_channels", 256)

        # Image encoder
        self.image_encoder = ImageEncoder(config)

        # Prompt encoder
        prompt_config = {
            "embed_dim": self.out_channels,
            "img_size": self.img_size,
            "patch_size": config.get("patch_size", 16),
        }
        self.prompt_encoder = PromptEncoder(prompt_config)

        # Mask decoder
        decoder_config = {
            "decoder_dim": config.get("decoder_dim", 256),
            "decoder_num_heads": config.get("decoder_num_heads", 8),
            "decoder_depth": config.get("decoder_depth", 2),
            "num_mask_tokens": config.get("num_mask_tokens", 4),
        }
        self.mask_decoder = MaskDecoder(decoder_config)

        # Image positional encoding
        grid_size = self.img_size // config.get("patch_size", 16)
        self.image_pe = nn.Parameter(
            torch.zeros(1, self.out_channels, grid_size, grid_size)
        )
        nn.init.trunc_normal_(self.image_pe, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: encode image, encode prompts, decode masks.

        Args:
            images: Input images (B, C, H, W).
            points: Optional point prompts (coords, labels).
            boxes: Optional box prompts (B, N, 4).
            masks: Optional mask prompts (B, 1, H, W).
            multimask_output: If True, return all mask predictions; if False,
                return only the best mask.

        Returns:
            Dictionary with:
                masks: Predicted masks.
                iou_predictions: Per-mask IoU scores.
                image_embeddings: Cached image features.
        """
        # Encode image (can be cached for multiple prompts)
        image_embeddings = self.image_encoder(images)

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points, boxes=boxes, masks=masks,
        )

        # Decode masks
        decoder_output = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.image_pe.expand(images.shape[0], -1, -1, -1),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )

        pred_masks = decoder_output["masks"]
        iou_predictions = decoder_output["iou_predictions"]

        # Select best mask if not multi-mask output
        if not multimask_output:
            best_idx = iou_predictions.argmax(dim=1)
            batch_idx = torch.arange(pred_masks.shape[0], device=pred_masks.device)
            pred_masks = pred_masks[batch_idx, best_idx].unsqueeze(1)
            iou_predictions = iou_predictions[batch_idx, best_idx].unsqueeze(1)

        return {
            "masks": pred_masks,
            "iou_predictions": iou_predictions,
            "image_embeddings": image_embeddings,
        }

    @torch.no_grad()
    def generate_masks(
        self,
        images: torch.Tensor,
        points_per_side: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """Automatic mask generation by sampling a grid of point prompts.

        Generates masks for the entire image by placing a uniform grid of
        foreground points and running the model for each.

        Args:
            images: Input images (B, C, H, W).
            points_per_side: Number of points along each side of the grid.

        Returns:
            Dictionary with all generated masks and scores.
        """
        B = images.shape[0]
        device = images.device

        # Generate point grid
        grid = torch.linspace(0, self.img_size, points_per_side + 2, device=device)[1:-1]
        grid_x, grid_y = torch.meshgrid(grid, grid, indexing="xy")
        point_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        point_coords = point_coords.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        point_labels = torch.ones(B, point_coords.shape[1], device=device, dtype=torch.long)

        # Encode image once
        image_embeddings = self.image_encoder(images)

        # Process each point individually
        all_masks = []
        all_ious = []

        for i in range(point_coords.shape[1]):
            pc = point_coords[:, i:i+1, :]
            pl = point_labels[:, i:i+1]

            sparse_emb, dense_emb = self.prompt_encoder(points=(pc, pl))
            out = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.image_pe.expand(B, -1, -1, -1),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
            )
            # Take the best mask for each point
            best_idx = out["iou_predictions"].argmax(dim=1)
            batch_idx = torch.arange(B, device=device)
            all_masks.append(out["masks"][batch_idx, best_idx])
            all_ious.append(out["iou_predictions"][batch_idx, best_idx])

        return {
            "masks": torch.stack(all_masks, dim=1),
            "iou_predictions": torch.stack(all_ious, dim=1),
        }


__all__ = [
    "SAM",
    "ImageEncoder",
    "PromptEncoder",
    "MaskDecoder",
    "TwoWayAttentionBlock",
    "SAMBlock",
    "SAMAttention",
]

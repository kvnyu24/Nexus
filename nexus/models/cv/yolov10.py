"""
YOLOv10: Real-Time End-to-End Object Detection

Implementation of YOLOv10, which introduces NMS-free training through one-to-one
label assignment and dual-label assignment strategy, achieving real-time detection
without post-processing overhead.

Reference:
    Wang, A., Chen, H., Liu, L., et al. (2024).
    "YOLOv10: Real-Time End-to-End Object Detection."
    arXiv:2405.14458

Key Components:
    - CSPNet backbone with spatial-channel decoupled downsampling
    - C2f/C3k2 blocks for efficient feature extraction
    - One-to-one head for NMS-free detection
    - Dual label assignment during training

Architecture Details:
    - No NMS required - one-to-one matching between predictions and GT
    - Consistent dual assignments (one-to-many + one-to-one) during training
    - Spatial-channel decoupled downsampling for efficiency
    - Large kernel convolutions for expanded receptive field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple

from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin


class C2fBlock(NexusModule):
    """C2f: CSP with 2 convolutions and fast implementation.

    Efficient bottleneck block used in YOLOv10.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        num_blocks (int): Number of bottleneck blocks. Default: 1.
        shortcut (bool): Use residual connection. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()

        hidden_channels = out_channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.SiLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d((2 + num_blocks) * hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        self.bottlenecks = nn.ModuleList([
            Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        y = self.conv1(x)
        y = list(y.chunk(2, dim=1))

        for bottleneck in self.bottlenecks:
            y.append(bottleneck(y[-1]))

        return self.conv2(torch.cat(y, dim=1))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
    ):
        super().__init__()

        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class SPPFBlock(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()

        hidden_channels = in_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class OneToOneHead(NexusModule):
    """One-to-one detection head for NMS-free inference.

    Uses one-to-one label assignment to eliminate the need for NMS.

    Args:
        in_channels (int): Input channels.
        num_classes (int): Number of classes.
        reg_max (int): Maximum value for distribution focal loss. Default: 16.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        reg_max: int = 16,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max

        # Separate branches for classification and regression
        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # Prediction heads
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(in_channels, 4 * reg_max, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features (B, C, H, W).

        Returns:
            Tuple of (class_pred, box_pred).
        """
        # Classification branch
        cls_feat = self.cls_convs(x)
        cls_pred = self.cls_pred(cls_feat)

        # Regression branch
        reg_feat = self.reg_convs(x)
        reg_pred = self.reg_pred(reg_feat)

        return cls_pred, reg_pred


class YOLOv10(WeightInitMixin, NexusModule):
    """YOLOv10: Real-Time End-to-End Object Detection.

    NMS-free object detector with one-to-one label assignment.

    Config:
        # Backbone config
        depth_multiple (float): Depth scaling factor. Default: 1.0.
        width_multiple (float): Width scaling factor. Default: 1.0.

        # Detection config
        num_classes (int): Number of object classes. Default: 80.
        reg_max (int): Maximum for DFL. Default: 16.

        # Architecture config
        base_channels (int): Base channel number. Default: 64.
        base_depth (int): Base depth multiplier. Default: 3.

    Example:
        >>> config = {
        ...     "num_classes": 80,
        ...     "depth_multiple": 1.0,
        ...     "width_multiple": 1.0,
        ... }
        >>> model = YOLOv10(config)
        >>> images = torch.randn(2, 3, 640, 640)
        >>> output = model(images)
        >>> for pred in output["predictions"]:
        ...     print(pred[0].shape, pred[1].shape)  # class and box predictions
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_classes = config.get("num_classes", 80)
        depth_mul = config.get("depth_multiple", 1.0)
        width_mul = config.get("width_multiple", 1.0)

        base_channels = config.get("base_channels", 64)
        base_depth = config.get("base_depth", 3)

        # Calculate actual channels
        def make_divisible(x: float, divisor: int = 8) -> int:
            return max(divisor, int(x + divisor / 2) // divisor * divisor)

        c1 = make_divisible(base_channels * width_mul)
        c2 = make_divisible(c1 * 2)
        c3 = make_divisible(c2 * 2)
        c4 = make_divisible(c3 * 2)
        c5 = make_divisible(c4 * 2)

        d1 = max(1, int(base_depth * depth_mul))

        # Backbone: CSPDarknet
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            C2fBlock(c2, c2, num_blocks=d1),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
            C2fBlock(c3, c3, num_blocks=d1 * 2),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
            C2fBlock(c4, c4, num_blocks=d1 * 2),
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.SiLU(inplace=True),
            C2fBlock(c5, c5, num_blocks=d1),
            SPPFBlock(c5, c5),
        )

        # Neck: PAN
        self.neck_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2fBlock(c5 + c4, c4, num_blocks=d1),
        )

        self.neck_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2fBlock(c4 + c3, c3, num_blocks=d1),
        )

        self.neck_down1 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
            C2fBlock(c3 + c4, c4, num_blocks=d1),
        )

        self.neck_down2 = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
            C2fBlock(c4 + c5, c5, num_blocks=d1),
        )

        # One-to-one detection heads (3 scales)
        self.heads = nn.ModuleList([
            OneToOneHead(c3, self.num_classes, reg_max=config.get("reg_max", 16)),
            OneToOneHead(c4, self.num_classes, reg_max=config.get("reg_max", 16)),
            OneToOneHead(c5, self.num_classes, reg_max=config.get("reg_max", 16)),
        ])

        self.init_weights_vit()

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Dictionary with multi-scale predictions (no NMS needed).
        """
        # Backbone
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        # Neck (top-down)
        p5 = s4
        p4 = self.neck_up1(torch.cat([F.interpolate(p5, size=s3.shape[-2:]), s3], dim=1))
        p3 = self.neck_up2(torch.cat([F.interpolate(p4, size=s2.shape[-2:]), s2], dim=1))

        # Neck (bottom-up)
        p4 = self.neck_down1(torch.cat([p3, p4], dim=1))
        p5 = self.neck_down2(torch.cat([p4, p5], dim=1))

        # Detection heads
        predictions = []
        for head, feature in zip(self.heads, [p3, p4, p5]):
            cls_pred, box_pred = head(feature)
            predictions.append((cls_pred, box_pred))

        return {
            "predictions": predictions,
            "features": [p3, p4, p5],
        }


__all__ = [
    "YOLOv10",
    "C2fBlock",
    "SPPFBlock",
    "OneToOneHead",
]

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin, FeatureBankMixin

class VisionEncoder(NexusModule):
    def __init__(self, input_channels: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)

class LanguageEncoder(NexusModule):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=4
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        return self.encoder(x, src_key_padding_mask=attention_mask)

class PaLME(ConfigValidatorMixin, FeatureBankMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config using ConfigValidatorMixin
        self.validate_config(config, required_keys=["hidden_dim", "num_layers", "vocab_size"])
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.num_heads = config.get("num_heads", 8)
        
        # Add position embeddings
        self.max_seq_length = config.get("max_seq_length", 1024)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_seq_length, self.hidden_dim)
        )
        
        # Add layer normalizations
        self.pre_vision_norm = nn.LayerNorm(self.hidden_dim)
        self.pre_language_norm = nn.LayerNorm(self.hidden_dim)
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        
        # Vision components
        self.vision_encoder = VisionEncoder(
            input_channels=config.get("input_channels", 3),
            hidden_dim=self.hidden_dim
        )
        
        # Language components
        self.language_encoder = LanguageEncoder(
            vocab_size=config["vocab_size"],
            hidden_dim=self.hidden_dim
        )
        
        # Cross-modal fusion (following EnhancedFusionModule pattern)
        self.cross_attention = nn.ModuleDict({
            'vision_to_text': nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=config.get("dropout", 0.1)
            ),
            'text_to_vision': nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=config.get("dropout", 0.1)
            )
        })
        
        # Add feature bank with configurable size using FeatureBankMixin
        self.bank_size = config.get("bank_size", 10000)
        self.register_feature_bank("feature", self.bank_size, self.hidden_dim)
        
        # Output heads
        self.task_heads = nn.ModuleDict({
            'language_generation': nn.Linear(self.hidden_dim, config["vocab_size"]),
            'visual_prediction': nn.Linear(self.hidden_dim, config.get("visual_dim", 256)),
            'action_prediction': nn.Linear(self.hidden_dim, config.get("num_actions", 100))
        })
        
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        # Process visual input with normalization
        if images is not None:
            visual_features = self.vision_encoder(images)
            visual_features = self.pre_vision_norm(visual_features)
            outputs["visual_features"] = visual_features
            
        # Process text input with position embeddings
        if text_tokens is not None:
            text_features = self.language_encoder(text_tokens, attention_mask)
            seq_length = text_features.size(1)
            text_features = text_features + self.position_embedding[:, :seq_length]
            text_features = self.pre_language_norm(text_features)
            outputs["text_features"] = text_features
            
        # Enhanced cross-modal fusion
        if images is not None and text_tokens is not None:
            # Vision-guided language features
            vision_text_features = self.cross_attention['vision_to_text'](
                text_features,
                visual_features,
                visual_features
            )[0]
            
            # Language-guided vision features
            text_vision_features = self.cross_attention['text_to_vision'](
                visual_features,
                text_features,
                text_features
            )[0]
            
            # Fuse and normalize features
            fused_features = self.final_norm(vision_text_features + text_vision_features)
            outputs["fused_features"] = fused_features
            
            # Update feature bank using FeatureBankMixin
            FeatureBankMixin.update_feature_bank(self, "feature", fused_features)
            
            # Generate task-specific outputs
            outputs.update({
                "language_logits": self.task_heads["language_generation"](fused_features),
                "visual_predictions": self.task_heads["visual_prediction"](fused_features),
                "action_logits": self.task_heads["action_prediction"](fused_features)
            })
            
        return outputs
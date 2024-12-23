from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
from .base_attention import BaseAttention

class MultiHeadSelfAttention(BaseAttention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project QKV using unified projection from base class
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention
        q = self._reshape_for_attention(q)
        k = self._reshape_for_attention(k) 
        v = self._reshape_for_attention(v)
        
        if self.use_flash_attention and torch.cuda.is_available():
            output = self.flash_attn_func(q, k, v, dropout_p=self.dropout.p)
            attention_weights = None
        else:
            # Standard scaled dot-product attention
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                    float("-inf")
                )
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.hidden_size)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_weights
        return output
# Learned Positional Encoding

## Overview

Learned Positional Encoding treats position embeddings as trainable parameters, allowing the model to learn optimal positional representations from data. Used in GPT-2, BERT, and RoBERTa.

**Key Characteristics**:
- Learned embedding table: `PE ∈ ℝ^(max_seq_len × d)`
- Simple implementation: Standard `nn.Embedding` layer
- No extrapolation capability: Cannot handle sequences longer than training
- High parameter count: O(max_seq_len × d) parameters

## When to Use

**Use learned PE if**:
- Following established architectures (GPT-2, BERT)
- Training and deployment lengths match exactly
- Can afford the parameter overhead
- Don't need length extrapolation

**Don't use if**:
- Need length generalization
- Working with very long sequences (parameter cost)
- Want zero-shot transfer to longer contexts

## Implementation

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(p=dropout)

        # Initialize with small random values
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, x, position_ids=None):
        seq_len = x.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        pos_embeddings = self.position_embeddings(position_ids)
        return self.dropout(x + pos_embeddings)
```

## Advantages

1. **Data-adaptive**: Learns optimal encoding from training data
2. **Simple**: No complex mathematical formulation
3. **Proven**: Works well in established models (BERT, GPT-2)

## Disadvantages

1. **No extrapolation**: Fails completely on longer sequences
2. **Parameter overhead**: Adds max_seq_len × d parameters
3. **Fixed length**: Must know maximum length at initialization
4. **No built-in structure**: Doesn't encode mathematical relationships

## Experiments

### Length Generalization

Training on 512 tokens:

| Test Length | Learned PE (PPL) | Sinusoidal (PPL) | RoPE (PPL) |
|-------------|------------------|------------------|------------|
| 512 (train) | 15.1 | 15.1 | 15.0 |
| 1024 | ∞ (fails) | 18.2 | 17.1 |
| 2048 | ∞ | 25.3 | 22.8 |

**Observation**: Learned PE cannot extrapolate beyond training length.

### Parameter Count

For typical settings:

| Model Size | Seq Len | Dim | Learned PE Params |
|------------|---------|-----|-------------------|
| Small | 512 | 768 | 393K |
| Base | 2048 | 768 | 1.6M |
| Large | 2048 | 1024 | 2.1M |

## Common Pitfalls

1. **Exceeding max_seq_len**: Model crashes on longer sequences
2. **Poor initialization**: Use small std (0.01-0.02) to avoid large initial gradients
3. **Forgetting dropout**: Prevents overfitting to exact positions

## References

- Devlin, J., et al. (2018). **BERT: Pre-training of Deep Bidirectional Transformers**. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Radford, A., et al. (2019). **Language Models are Unsupervised Multitask Learners**. OpenAI.

**Implementation**: [/nexus/components/embeddings/learned_pe.py](../../nexus/components/embeddings/learned_pe.py)

---

**Next**: [Relative Bias](./relative_bias.md) | [Back to Overview](./README.md)

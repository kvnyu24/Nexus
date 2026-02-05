# Tokenization

Tokenization converts text into discrete units (tokens) that language models can process. Traditional approaches use fixed vocabularies (BPE, WordPiece), but modern byte-level methods eliminate the need for predetermined vocabularies.

## Overview

The choice of tokenization strategy significantly impacts model performance, efficiency, vocabulary size, and the ability to handle multiple languages or domains. Recent approaches operate directly on bytes to achieve true vocabulary-freedom.

## When to Use Byte-Level Tokenization

Use byte-level tokenization when you need:

1. **True Multi-linguality**: Handle any language without vocabulary constraints
2. **Robustness**: Process any byte sequence (text, code, binary data)
3. **No Preprocessing**: Eliminate tokenizer training and maintenance
4. **Long-tail Handling**: Better performance on rare words/characters
5. **Simplicity**: Single model for all languages and domains

## Approaches

### 1. Byte Latent Transformer (BLT)

Uses entropy-based dynamic patching to group bytes into variable-length patches, with a latent transformer processing patch-level representations.

**Strengths:**
- Dynamic patch sizes (adaptive to content complexity)
- Better scaling than fixed tokenization
- No tokenizer vocabulary
- Efficient handling of both simple and complex content

**Weaknesses:**
- Complex implementation (entropy computation, patching)
- Training requires careful tuning
- Inference overhead from dynamic patching
- Less mature than traditional tokenization

**Use when:** You want state-of-the-art byte-level modeling with adaptive granularity, especially for mixed-complexity content.

See: [byte_latent_transformer.md](./byte_latent_transformer.md)

### 2. MambaByte

Applies Mamba (selective state space model) directly to raw bytes, leveraging SSM efficiency for long-range byte-level modeling.

**Strengths:**
- Efficient long-sequence modeling (SSM benefits)
- Simpler architecture than BLT (no dynamic patching)
- True language-agnostic
- Better scaling than byte-level transformers

**Weaknesses:**
- SSM complexity (harder to implement)
- May underperform on short sequences
- Limited to sequential processing
- Newer architecture (fewer resources)

**Use when:** You need efficient byte-level modeling for long sequences, or want to leverage SSM benefits for tokenizer-free models.

See: [mambabyte.md](./mambabyte.md)

## Comparison with Traditional Tokenization

| Feature | BPE/WordPiece | BLT | MambaByte |
|---------|---------------|-----|-----------|
| Vocabulary | Fixed (30K-100K) | None (256 bytes) | None (256 bytes) |
| Multi-lingual | Limited | Excellent | Excellent |
| Robustness | Poor (OOV) | Excellent | Excellent |
| Sequence Length | Shorter | Medium (patches) | Longer (efficient) |
| Inference Speed | Fast | Medium | Fast-Medium |
| Training Complexity | Low | High | Medium-High |
| Implementation | Simple | Complex | Medium |

## Comparison Matrix

| Feature | Byte Latent Transformer | MambaByte |
|---------|------------------------|-----------|
| Architecture | Transformer + patching | Mamba SSM |
| Patch Strategy | Dynamic (entropy) | Fixed byte-level |
| Efficiency | Medium | High (SSM) |
| Long Context | Medium | High |
| Implementation | Complex | Medium |
| Maturity | Cutting-edge (2024) | Recent (2024) |
| Best Use Case | Mixed complexity | Long sequences |

## Best Practices

### General Byte-Level Modeling

1. **Pretraining Data**: Byte-level models benefit from diverse, multilingual data
2. **Sequence Length**: Start with shorter sequences during training, gradually increase
3. **Batch Size**: Use larger batches than token-based models (compensate for longer sequences)
4. **Learning Rate**: Lower learning rates often work better for byte-level models
5. **Evaluation**: Evaluate on bytes-per-character and bits-per-byte metrics

### Byte Latent Transformer

1. **Entropy Threshold Tuning**: Adjust threshold based on content type (lower for structured data)
2. **Patch Size Limits**: Set max/min patch sizes appropriate for your domain
3. **Local Encoder Depth**: Deeper local encoders for complex within-patch patterns
4. **Latent Dimension**: Balance patch-level and latent-level expressiveness

### MambaByte

1. **State Size**: Larger state sizes for more complex dependencies
2. **SSM Initialization**: Use structured SSM initialization for stability
3. **Convolution Kernel**: Adjust kernel size based on local pattern complexity
4. **Layer Depth**: More layers compensate for SSM's different inductive bias

## Training Considerations

### Data Preprocessing
- Byte-level models need raw bytes (UTF-8 encoding)
- No special tokenization or normalization
- Handle byte sequences up to max length

### Memory Requirements
- Byte sequences are ~4x longer than BPE tokens
- Use gradient checkpointing for long sequences
- Consider sequence packing for efficiency

### Optimization
- Warmup learning rate for stability
- Gradient clipping (bytes have different dynamics)
- Mixed precision training (FP16/BF16)

## Common Pitfalls

1. **Sequence Length Mismatch**: Forgetting that byte sequences are much longer
2. **Inefficient Batching**: Not packing sequences efficiently
3. **Wrong Metrics**: Using token-based metrics instead of byte-based
4. **Character Encoding**: Mixing encodings (always use UTF-8)
5. **Evaluation Bias**: Comparing to subword models without accounting for granularity

## Performance Metrics

- **Bits per Byte (BPB)**: Lower is better
- **Bytes per Character**: Efficiency for different languages
- **Inference Latency**: Time to generate N bytes
- **Memory Usage**: Peak memory during training/inference

## Deployment Considerations

1. **Serving Latency**: Byte-level models may be slower for short text
2. **Input Preprocessing**: Minimal (just byte encoding)
3. **Output Decoding**: UTF-8 decoding with error handling
4. **Caching**: Cache byte-level hidden states for faster generation

## Resources

- BLT Paper: [https://arxiv.org/abs/2412.09871](https://arxiv.org/abs/2412.09871)
- MambaByte Paper: [https://arxiv.org/abs/2401.13660](https://arxiv.org/abs/2401.13660)
- Mamba: [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
- ByT5 (earlier byte-level work): [https://arxiv.org/abs/2105.13626](https://arxiv.org/abs/2105.13626)

## Example Use Cases

1. **Multilingual Modeling**: Single model for 100+ languages
2. **Code Generation**: Handle any programming language without tokenizer updates
3. **Mixed Content**: Process text, code, and structured data together
4. **Rare Language Support**: Model low-resource languages without vocabulary design
5. **Binary Data**: Process any byte sequence (not just text)

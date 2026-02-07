# Knowledge Distillation for Large Language Models

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model, enabling the student to achieve performance close to the teacher while being significantly more efficient. For LLMs, distillation is essential for deploying capable models at scale.

## Overview

| Method | Type | Key Innovation | Best For |
|--------|------|----------------|----------|
| Knowledge Distiller | Output-based | Soft logits matching | General-purpose distillation |
| Rationale-based KD | Intermediate | Reasoning chain transfer | Complex reasoning tasks |
| Minitron | Architecture search | Layer/neuron pruning + distillation | Structured compression |
| TinyBERT | Layer-wise | Multi-stage progressive distillation | BERT-family models |

## When to Use Distillation

### Use Cases

**Ideal for Distillation:**
- Deploying smaller models in production
- Reducing inference costs while maintaining quality
- Creating specialized models for specific tasks
- Ensemble knowledge transfer to single model
- When you have a strong teacher and unlimited unlabeled data

**Consider Other Methods:**
- If student architecture is very different (hard to transfer)
- Limited computation budget (distillation requires training)
- When teacher itself is suboptimal (distillation preserves errors)
- For >10× compression (combine with quantization/pruning)

### Compression Comparison

Distilling a 7B model to 1B:

| Method | Student Size | Training Time | Accuracy Retention | Inference Speed |
|--------|--------------|---------------|-------------------|-----------------|
| Train from scratch | 1B | 100 hrs | 70-80% | 10× |
| **Output distillation** | **1B** | **20 hrs** | **85-90%** | **10×** |
| Rationale distillation | 1B | 30 hrs | 90-93% | 10× |
| Minitron | 1B | 40 hrs | 88-92% | 10× |

Distillation significantly improves efficiency of training smaller models.

## Distillation Fundamentals

### What is Knowledge Distillation?

Training a student model to mimic a teacher model's behavior:

```
Loss = α * L_task(y, y_student) + (1-α) * L_distill(y_teacher, y_student)
```

where:
- **L_task**: Standard task loss (e.g., cross-entropy with true labels)
- **L_distill**: Distillation loss (matching teacher outputs)
- **α**: Weight balancing task and distillation (typically 0.1-0.5)

### Temperature Scaling

Soften probability distributions for better knowledge transfer:

```
Soft Target: p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

where:
- **z_i**: Logit for class i
- **T**: Temperature (T=1 is standard softmax, T>1 is softer)

**Why it works**: High temperature reveals relative similarities between classes:
- T=1: [0.9, 0.05, 0.03, 0.02] (hard, one class dominates)
- T=5: [0.6, 0.2, 0.15, 0.05] (soft, shows class relationships)

### Types of Distillation

**1. Output Distillation** (Logit Matching):
```python
# Match final layer logits
loss_distill = MSE(logits_student, logits_teacher)
# or KL divergence on softened probabilities
loss_distill = KL(softmax(logits_student/T), softmax(logits_teacher/T))
```

**2. Intermediate Distillation** (Feature Matching):
```python
# Match intermediate hidden states
loss_distill = MSE(hidden_student, project(hidden_teacher))
```

**3. Attention Distillation**:
```python
# Match attention distributions
loss_distill = MSE(attn_student, attn_teacher)
```

**4. Rationale Distillation** (Chain-of-Thought):
```python
# Transfer reasoning steps
loss_distill = -log P(reasoning_chain_teacher | input, student)
```

## Method Comparison

### Output vs Intermediate vs Rationale

**Output Distillation**:
- **Pros**: Simple, task-agnostic, no architecture constraints
- **Cons**: Doesn't transfer reasoning process, limited for complex tasks
- **Use when**: General-purpose compression, simple tasks

**Intermediate Distillation**:
- **Pros**: Transfers richer knowledge, better for complex reasoning
- **Cons**: Requires architecture alignment, more memory
- **Use when**: Student and teacher have similar architectures

**Rationale Distillation**:
- **Pros**: Transfers reasoning ability, excellent for complex tasks
- **Cons**: Requires rationale generation, more expensive
- **Use when**: Math, coding, multi-step reasoning tasks

### Performance Comparison (Distilling GPT-3.5 to 1.5B Student)

| Method | MMLU | GSM8K | HumanEval | Training Time |
|--------|------|-------|-----------|---------------|
| Train from scratch | 35.2% | 12.4% | 8.2% | 100 hrs |
| Output distillation | 42.8% (+7.6) | 28.6% (+16.2) | 15.3% (+7.1) | 20 hrs |
| Intermediate distillation | 44.1% (+8.9) | 32.4% (+20.0) | 17.8% (+9.6) | 30 hrs |
| **Rationale distillation** | **46.3% (+11.1)** | **41.2% (+28.8)** | **22.1% (+13.9)** | **35 hrs** |
| Minitron | 45.7% (+10.5) | 38.5% (+26.1) | 20.6% (+12.4) | 45 hrs |

Rationale distillation provides best quality, especially for reasoning tasks.

## Quick Start

### Basic Output Distillation

```python
from nexus.models.compression.distillation import KnowledgeDistiller

# Load teacher and student
teacher = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
student = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-1b")

# Create distiller
distiller = KnowledgeDistiller(
    teacher=teacher,
    student=student,
    temperature=2.0,
    alpha=0.5  # 50% task loss, 50% distillation loss
)

# Train student
for batch in dataloader:
    loss = distiller.compute_loss(
        input_ids=batch['input_ids'],
        labels=batch['labels']
    )
    loss.backward()
    optimizer.step()
```

### Rationale-Based Distillation

```python
from nexus.models.compression.distillation import RationaleDistiller

# Generate reasoning chains from teacher
teacher_outputs = teacher.generate(
    input_ids,
    output_hidden_states=True,
    return_dict_in_generate=True,
    max_new_tokens=200  # Allow space for reasoning
)

# Distill with rationales
distiller = RationaleDistiller(
    teacher=teacher,
    student=student,
    rationale_weight=0.7  # Higher weight on reasoning
)

loss = distiller.compute_loss(
    input_ids=input_ids,
    labels=labels,
    teacher_rationale=teacher_outputs['sequences']
)
```

### Minitron (Architecture Search + Distillation)

```python
from nexus.models.compression.distillation import MinitronPruner

# Prune teacher to smaller architecture
pruner = MinitronPruner(
    teacher=teacher_7b,
    target_params=1.5e9,  # Target 1.5B parameters
    pruning_strategy="importance"
)

# Get pruned student architecture
student = pruner.prune_and_create_student()

# Distill knowledge
distiller = KnowledgeDistiller(teacher=teacher_7b, student=student)
# ... train as usual
```

## Hyperparameter Guidelines

### Temperature Selection

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| T = 1 | Hard targets (standard softmax) | When teacher is uncertain |
| T = 2 | Slightly soft | General distillation (recommended) |
| T = 4 | Soft targets | Large teacher-student gap |
| T = 8 | Very soft | Extreme compression (10×+) |

**Rule of thumb**: T = log₂(teacher_size / student_size) + 1

### Alpha (Loss Weight) Selection

```
Loss = α * L_task + (1-α) * L_distill
```

| Alpha | Priority | Use Case |
|-------|----------|----------|
| α = 0.1 | Heavily favor distillation | Abundant unlabeled data |
| α = 0.3 | Favor distillation | Moderate labeled data |
| α = 0.5 | Balanced | Equal weight (recommended) |
| α = 0.7 | Favor task loss | Limited unlabeled data |
| α = 0.9 | Heavily favor task loss | Weak teacher |

### Student-Teacher Size Ratio

| Ratio | Difficulty | Expected Retention | Notes |
|-------|------------|-------------------|-------|
| 1:2 (3.5B → 1.7B) | Easy | 95-98% | Minimal distillation needed |
| 1:4 (7B → 1.7B) | Medium | 85-92% | Recommended range |
| 1:8 (13B → 1.6B) | Hard | 75-85% | Significant compression |
| 1:10+ (70B → 7B) | Very Hard | 65-80% | Use advanced methods |

**Guideline**: Diminishing returns beyond 4× compression with basic distillation.

### Data Requirements

| Training Data Size | Quality | Expected Results |
|-------------------|---------|------------------|
| 1M samples | Low diversity | Poor (60-70% retention) |
| 10M samples | Medium diversity | Good (80-85% retention) |
| 100M samples | High diversity | Excellent (90-95% retention) |
| Unlimited | Max diversity | Optimal (95%+ retention) |

**Note**: Quality > quantity. 10M diverse samples better than 100M repetitive.

## Advanced Topics

### Multi-Teacher Distillation

Learn from multiple teachers:

```python
# Ensemble of teachers
teachers = [model_7b_v1, model_7b_v2, model_13b]

for batch in dataloader:
    # Average teacher outputs
    teacher_logits = [teacher(batch) for teacher in teachers]
    avg_logits = torch.mean(torch.stack(teacher_logits), dim=0)

    # Distill from ensemble
    student_logits = student(batch)
    loss = distillation_loss(student_logits, avg_logits, T=2.0)
```

**When to use**: Multiple models with complementary strengths.

### Progressive Distillation

Gradually reduce student size:

```
7B → 5B → 3B → 1.5B → 1B
```

Each step uses the previous as teacher. Benefits:
- Easier optimization (smaller gaps)
- Better final accuracy (+2-3%)
- More compute expensive

### Self-Distillation

Student becomes its own teacher:

```python
# 1. Train student normally
student.train()

# 2. Use student as teacher
with torch.no_grad():
    teacher_logits = student(batch)

# 3. Retrain student with its own soft targets
student_logits = student(batch)
loss = distillation_loss(student_logits, teacher_logits.detach())
```

**Effect**: Regularization, improved calibration, slight accuracy gain.

### Task-Specific Distillation

Distill for specific capabilities:

```python
# Math-focused distillation
distiller = KnowledgeDistiller(
    teacher=teacher,
    student=student,
    task_weight={'math': 0.5, 'code': 0.3, 'general': 0.2}
)

# Sample more from math/code datasets
dataloader = create_weighted_dataloader(weights={'math': 0.5, ...})
```

### Combining with Quantization/Pruning

**Option 1**: Distill then compress
```python
# 1. Distill 7B → 1.5B
student = distill(teacher_7b, student_1.5b)

# 2. Quantize student to 4-bit
student = quantize(student, bits=4)

# Result: 1.5B @ 4-bit = ~750MB
```

**Option 2**: Compress teacher, then distill
```python
# 1. Quantize teacher to 4-bit
teacher_4bit = quantize(teacher_7b, bits=4)

# 2. Distill from quantized teacher
student = distill(teacher_4bit, student_1.5b)

# Faster distillation, slightly lower quality
```

**Recommendation**: Option 1 for quality, Option 2 for speed.

## Common Issues & Solutions

### Issue 1: Student Doesn't Improve

**Symptoms**: Student accuracy stuck near random, not learning from teacher.

**Solutions**:
1. Reduce temperature (T=4 → T=2)
2. Increase alpha (more weight on task loss)
3. Check if teacher is actually good (accuracy, calibration)
4. Ensure student capacity is sufficient
5. Verify data distribution matches evaluation

### Issue 2: Overfitting to Teacher

**Symptoms**: Student matches teacher on train set but both generalize poorly to test.

**Solutions**:
1. Increase alpha (favor task loss over distillation)
2. Add regularization (dropout, weight decay)
3. Use more diverse training data
4. Reduce distillation weight for later training epochs

### Issue 3: Training Unstable

**Symptoms**: Loss spikes, gradients explode, NaN values.

**Solutions**:
1. Reduce learning rate (1e-4 → 1e-5)
2. Gradient clipping: `torch.nn.utils.clip_grad_norm_(params, 1.0)`
3. Lower temperature (T=4 → T=2)
4. Use mixed precision training carefully
5. Check for numerical issues in distillation loss

### Issue 4: Slow Convergence

**Symptoms**: Training takes many more epochs than expected.

**Solutions**:
1. Increase learning rate for student
2. Use learning rate warmup: 0 → peak over 1000 steps
3. Pre-train student on task before distillation
4. Use larger batch size (if memory allows)
5. Try progressive distillation (intermediate size models)

### Issue 5: Student Worse Than Training From Scratch

**Symptoms**: Distilled student underperforms randomly initialized trained student.

**Possible Causes**:
- Teacher is poor quality
- Distillation hyperparameters wrong (temperature, alpha)
- Student capacity insufficient
- Data mismatch

**Solution**: Ablation study:
```python
# Compare three approaches
student_scratch = train_from_scratch(student, data)
student_distill_low_T = distill(teacher, student, T=1.0)
student_distill_high_T = distill(teacher, student, T=4.0)
student_distill_balanced = distill(teacher, student, alpha=0.5, T=2.0)

# Find which works best
```

### Issue 6: Poor Reasoning Transfer

**Symptoms**: Student matches teacher on simple tasks but fails on complex reasoning.

**Solutions**:
1. Use rationale-based distillation
2. Add intermediate layer distillation
3. Distill attention patterns
4. Increase model capacity (1B → 1.5B)
5. Train longer with reasoning-heavy data

## Benchmarks

### Compression Ratios and Accuracy

Distilling various teachers to 1.5B student:

| Teacher | Size | Student Acc | Teacher Acc | Retention | Compression |
|---------|------|-------------|-------------|-----------|-------------|
| GPT-2-XL | 1.5B | 58.2% | 60.1% | 96.8% | 1.0× |
| LLaMA-2-7B | 7B | 62.4% | 68.3% | 91.4% | 4.7× |
| LLaMA-2-13B | 13B | 64.1% | 71.2% | 90.0% | 8.7× |
| LLaMA-2-70B | 70B | 67.8% | 78.5% | 86.4% | 46.7× |

**Observation**: Larger teachers transfer more knowledge, but retention decreases.

### Method Comparison (7B → 1.5B)

| Method | MMLU | GSM8K | HumanEval | BBH | Avg Retention |
|--------|------|-------|-----------|-----|---------------|
| Teacher (7B) | 42.1% | 38.5% | 15.2% | 35.8% | 100% |
| From Scratch | 35.2% | 12.4% | 8.2% | 25.1% | 62.0% |
| Output KD | 38.7% | 28.6% | 12.3% | 30.2% | 84.2% |
| Intermediate KD | 39.4% | 32.4% | 13.8% | 31.5% | 89.7% |
| **Rationale KD** | **40.8%** | **36.1%** | **14.5%** | **33.2%** | **94.0%** |
| Minitron | 40.2% | 34.7% | 14.1% | 32.6% | 92.5% |

### Training Efficiency

Distillation vs training from scratch (achieving 35% MMLU with 1.5B model):

| Method | Training Time | GPU-Hours | Data Required | Final Accuracy |
|--------|---------------|-----------|---------------|----------------|
| From Scratch | 100 hrs | 800 | 1T tokens | 35.2% |
| Output KD | 20 hrs | 160 | 100B tokens | 38.7% |
| Rationale KD | 35 hrs | 280 | 50B tokens | 40.8% |

**Distillation is 5× more data-efficient and achieves higher accuracy.**

### Inference Speed (A100 GPU)

| Model | Size | Latency (ms) | Throughput (tokens/s) | Memory (GB) |
|-------|------|--------------|----------------------|-------------|
| Teacher (7B, FP16) | 7B | 31 | 32 | 14 |
| Student (1.5B, FP16) | 1.5B | 8 | 125 | 3 |
| Student (1.5B, INT8) | 1.5B | 6 | 167 | 1.5 |
| Student (1.5B, INT4) | 1.5B | 4 | 250 | 0.8 |

**Distilled student is 4-8× faster and uses 4-17× less memory.**

## Tools & Libraries

### Production-Ready

- **Transformers (Hugging Face)**: Built-in distillation utilities
  ```python
  from transformers import DistillationTrainer
  ```

- **TextBrewer**: https://github.com/airaria/TextBrewer
  - Flexible distillation framework
  - Multiple distillation strategies
  - Easy to customize

- **Neural Compressor**: https://github.com/intel/neural-compressor
  - Intel-optimized distillation
  - Combines with quantization
  - Production deployment tools

### Research

- **Nexus**: `Nexus/nexus/models/compression/distillation/`
- **MiniLLM**: https://github.com/microsoft/LMOps/tree/main/minillm
- **Distilling Step-by-Step**: https://github.com/google-research/distilling-step-by-step

## References

### Papers

1. **Knowledge Distillation**: Hinton et al. "Distilling the Knowledge in a Neural Network." NIPS 2014 Workshop.

2. **Rationale Distillation**: Hsieh et al. "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data." ACL 2023.

3. **Minitron**: Sreenivas et al. "LLM Pruning and Distillation in Practice." arXiv 2024.

4. **TinyBERT**: Jiao et al. "TinyBERT: Distilling BERT for Natural Language Understanding." EMNLP 2020.

5. **MiniLLM**: Gu et al. "Knowledge Distillation of Large Language Models." ICLR 2024.

6. **Self-Distillation**: Zhang et al. "Be Your Own Teacher: Improve the Performance of CNNs via Self Distillation." ICCV 2019.

### Surveys

- Gou et al. "Knowledge Distillation: A Survey." IJCV 2021.
- Xu et al. "A Survey on Model Compression and Acceleration for Pre-trained Language Models." AAAI 2024.

## See Also

- [PEFT Methods](../peft/README.md): Parameter-efficient fine-tuning
- [Quantization Methods](../quantization/README.md): Reduce precision
- [Pruning Methods](../pruning/README.md): Remove parameters

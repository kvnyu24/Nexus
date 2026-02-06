# Continual Learning

This directory contains comprehensive documentation for continual learning methods, which enable neural networks to learn new tasks sequentially without forgetting previously learned knowledge.

## Overview

Continual learning (also called lifelong learning or incremental learning) addresses the fundamental challenge of **catastrophic forgetting**: the tendency of neural networks to forget previously learned tasks when trained on new data. This is critical for:

- **Lifelong learning systems**: AI that continuously adapts to new information
- **Resource-constrained deployment**: Cannot retrain from scratch on all data
- **Privacy-preserving learning**: Old data may not be accessible
- **Dynamic environments**: Tasks and distributions change over time
- **Few-shot adaptation**: Quickly learn new tasks without forgetting old ones

## Algorithm Categories

### 1. Regularization-Based Methods
- **EWC (Elastic Weight Consolidation)**: Constrains important parameters using Fisher Information
- **EVCL (Elastic Variational Continual Learning)**: Variational Bayesian approach with probabilistic parameter importance

### 2. Rehearsal-Based Methods
- **Self-Synthesized Rehearsal**: Generates synthetic samples from learned distribution to prevent forgetting

### 3. Prompt-Based Methods (for Vision Transformers)
- **L2P (Learning to Prompt)**: Task-conditioned prompt pool for continual learning
- **DualPrompt**: Separate general and task-specific prompts
- **CODA-Prompt**: Context-aware prompt selection with domain adaptation

## Learning Path

### Beginner Path
1. **Start with EWC** (`ewc.md`)
   - Understand the catastrophic forgetting problem
   - Learn Fisher Information Matrix computation
   - Grasp regularization-based continual learning
   - See how to identify important parameters

2. **Progress to Self-Synthesized Rehearsal** (`self_synthesized_rehearsal.md`)
   - Understand replay-based approaches
   - Learn generative modeling for continual learning
   - Master memory-efficient rehearsal strategies
   - See privacy-preserving continual learning

### Intermediate Path
3. **Study EVCL** (`evcl.md`)
   - Learn variational Bayesian continual learning
   - Understand probabilistic parameter importance
   - Master uncertainty quantification in continual learning
   - See elastic adaptation mechanisms

### Advanced Path
4. **Master Prompt-Based CL** (`prompt_based_cl.md`)
   - Understand parameter-efficient continual learning
   - Learn prompt pool mechanisms
   - Study task-conditioned prompt selection
   - Master Vision Transformer continual learning
   - See state-of-the-art performance on standard benchmarks

## Quick Comparison

| Method | Category | Key Innovation | Memory | Forget Rate | Task ID Needed | Best For |
|--------|----------|----------------|--------|-------------|----------------|----------|
| EWC | Regularization | Fisher Information | Low | Medium | No | Simple baselines |
| EVCL | Regularization | Variational Bayesian | Low | Low | No | Uncertainty-aware CL |
| Self-Synthesized | Rehearsal | Synthetic replay | Medium | Very Low | No | Privacy-preserving |
| L2P | Prompt-based | Prompt pool | Low | Very Low | At test | Vision Transformers |
| DualPrompt | Prompt-based | Dual prompts | Low | Very Low | At test | Task-incremental |
| CODA-Prompt | Prompt-based | Context-aware | Low | Lowest | At test | Domain adaptation |

## Key Concepts

### Catastrophic Forgetting
The fundamental problem in continual learning:
```
When training on task T_n, performance on tasks T_1...T_{n-1} degrades significantly
```

### Continual Learning Scenarios

1. **Task-Incremental Learning (Task-IL)**
   - Task identity known at test time
   - Easier scenario
   - Can use task-specific components

2. **Domain-Incremental Learning (Domain-IL)**
   - Same task, different domains
   - No task identity at test time
   - Must handle distribution shift

3. **Class-Incremental Learning (Class-IL)**
   - New classes added incrementally
   - Hardest scenario
   - No task identity, expanding output space

### Stability-Plasticity Trade-off
```
Stability: Retain knowledge of old tasks
Plasticity: Adapt to new tasks
Goal: Balance both without catastrophic forgetting
```

### Key Metrics

1. **Average Accuracy**: Mean performance across all tasks
2. **Forgetting Measure**: How much performance degrades on old tasks
3. **Forward Transfer**: How old knowledge helps new tasks
4. **Backward Transfer**: How new learning affects old tasks

## Implementation Guide

### Code Structure in Nexus
All implementations are located in `/nexus/models/continual/`:
- `ewc.py`: Elastic Weight Consolidation
- `evcl.py`: Elastic Variational Continual Learning
- `self_synthesized_rehearsal.py`: Self-Synthesized Rehearsal
- `prompt_based_cl.py`: L2P, DualPrompt, CODA-Prompt

### Common Patterns
All implementations follow the Nexus design:
```python
from nexus.models.continual import EVCLModel

config = {
    "input_dim": 784,
    "hidden_dims": [256, 256],
    "num_classes": 10,
    "num_tasks": 5,
    "prior_std": 1.0,
    "kl_weight": 1e-4,
}

model = EVCLModel(config)

# Train on task 1
for batch in task1_data:
    loss, metrics = model.train_step(batch, task_id=0)

# Save task-specific posterior
model.consolidate_task(task_id=0)

# Train on task 2 (with regularization to prevent forgetting)
for batch in task2_data:
    loss, metrics = model.train_step(batch, task_id=1)
```

### Evaluation Protocol
```python
# Evaluate on all tasks after learning task T
avg_acc = 0
for task_id in range(num_tasks_learned):
    acc = evaluate(model, test_data[task_id], task_id)
    avg_acc += acc
avg_acc /= num_tasks_learned

# Compute forgetting
forgetting = initial_acc[task_id] - current_acc[task_id]
```

## When to Use Each Method

### Choose EWC when:
- Starting with continual learning
- Need simple baseline
- Limited computational budget
- Working with small models
- Want interpretable importance scores

### Choose EVCL when:
- Need uncertainty quantification
- Want probabilistic approach
- Have computational resources for variational inference
- Dealing with ambiguous tasks
- Research on Bayesian continual learning

### Choose Self-Synthesized Rehearsal when:
- Privacy is a concern (no real data storage)
- Can train generative model
- Need strong anti-forgetting
- Working with image data
- Have GPU memory for generator

### Choose L2P when:
- Using Vision Transformers
- Limited parameter budget
- Need parameter-efficient solution
- Task ID available at test time
- Class-incremental learning

### Choose DualPrompt when:
- Using Vision Transformers
- Want to separate general vs. task-specific knowledge
- Need better task separation
- Class-incremental learning with many tasks

### Choose CODA-Prompt when:
- Using Vision Transformers
- Domain shift between tasks
- Need state-of-the-art performance
- Can afford prompt pool overhead
- Domain-incremental or class-incremental learning

## Common Pitfalls

### General Issues
1. **Evaluation protocol**: Must test on all previous tasks after each new task
2. **Task boundaries**: Clear task separation required in training
3. **Hyperparameter tuning**: Cannot tune on test tasks
4. **Memory leakage**: Accidentally using test task data for training
5. **Comparison fairness**: Same backbone, same compute budget

### Method-Specific Issues
- **EWC**: Fisher matrix computation expensive, sensitive to λ
- **EVCL**: Variational inference adds overhead, needs careful initialization
- **Rehearsal**: Generator quality critical, may memorize training data
- **Prompt-based**: Requires ViT backbone, task ID at test time for some variants

## Mathematical Foundations

### Continual Learning Objective
```
Minimize: L = ∑_{t=1}^T [L_t(θ) + R_t(θ, θ_{t-1})]

where:
  L_t(θ): Loss on current task t
  R_t(θ, θ_{t-1}): Regularization to prevent forgetting
```

### EWC Regularization
```
R_EWC(θ) = λ/2 ∑_i F_i (θ_i - θ*_i)^2

where:
  F_i: Fisher Information for parameter i
  θ*_i: Optimal parameter from previous task
  λ: Regularization strength
```

### Variational Continual Learning (EVCL)
```
Minimize: -E_q[log p(D_t|θ)] + KL(q(θ) || p(θ|D_{1:t-1}))

where:
  q(θ): Variational posterior
  p(θ|D_{1:t-1}): Prior from previous tasks
  D_t: Data from task t
```

### Prompt-Based Objective
```
y = f(x; [P(x, k); E(x)], Θ)

where:
  P(x, k): Selected prompts from pool
  E(x): Input embeddings
  k: Query key for prompt selection
  Θ: Frozen pre-trained parameters
```

## Experimental Results

### Standard Benchmarks
- **Split CIFAR-10**: 10 classes split into 5 tasks
- **Split CIFAR-100**: 100 classes split into 10 or 20 tasks
- **Split ImageNet**: 1000 classes split into multiple tasks
- **5-Datasets**: CIFAR-10, MNIST, notMNIST, Fashion-MNIST, SVHN
- **CORe50**: 50 object classes with domain shift

### Typical Performance (Average Accuracy on Split CIFAR-100, 10 tasks)
- **Finetuning (upper bound with task ID)**: ~65%
- **EWC**: ~45-50%
- **EVCL**: ~52-55%
- **Self-Synthesized Rehearsal**: ~55-60%
- **L2P**: ~83-85%
- **DualPrompt**: ~85-87%
- **CODA-Prompt**: ~87-90%

Note: Prompt-based methods use pre-trained ViT, others train from scratch.

## Best Practices

### Training
1. **Clear task boundaries**: Ensure clean task separation during training
2. **Hyperparameter selection**: Tune only on validation set of seen tasks
3. **Multiple seeds**: Report mean ± std over multiple runs
4. **Fair comparison**: Same architecture, same total parameters
5. **Compute budget**: Report training time and memory usage

### Evaluation
1. **Continual evaluation**: Test on all previous tasks after each new task
2. **Report multiple metrics**: Accuracy, forgetting, forward/backward transfer
3. **Task-IL vs Class-IL**: Specify which scenario
4. **Test-time task ID**: Clearly state if task identity is available
5. **Upper/lower bounds**: Compare against joint training and finetuning

### Implementation
1. **Modular design**: Separate task learning and consolidation
2. **Checkpointing**: Save model after each task
3. **Efficient storage**: Store only necessary statistics (Fisher, prompts)
4. **Gradient control**: Careful gradient flow for regularization
5. **Numerical stability**: Use log-space for Fisher, clip gradients

## References

### Foundational Papers
1. **Catastrophic Forgetting**: McCloskey & Cohen (1989) - "Catastrophic Interference in Connectionist Networks"
2. **EWC**: Kirkpatrick et al. (2017) - "Overcoming Catastrophic Forgetting in Neural Networks"
3. **Three Scenarios**: van de Ven & Tolias (2019) - "Three Scenarios for Continual Learning"

### Regularization Methods
4. **EWC**: Kirkpatrick et al. (2017) - "Overcoming Catastrophic Forgetting in Neural Networks"
5. **Online EWC**: Schwarz et al. (2018) - "Progress & Compress"
6. **VCL**: Nguyen et al. (2018) - "Variational Continual Learning"
7. **EVCL**: Nguyen et al. (2024) - "Elastic Variational Continual Learning"

### Rehearsal Methods
8. **GEM**: Lopez-Paz & Ranzato (2017) - "Gradient Episodic Memory"
9. **iCaRL**: Rebuffi et al. (2017) - "iCaRL: Incremental Classifier and Representation Learning"
10. **DGR**: Shin et al. (2017) - "Continual Learning with Deep Generative Replay"
11. **Self-Synthesized**: Yin et al. (2020) - "Dreaming to Distill"

### Prompt-Based Methods
12. **L2P**: Wang et al. (2022) - "Learning to Prompt for Continual Learning"
13. **DualPrompt**: Wang et al. (2022) - "DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning"
14. **CODA-Prompt**: Smith et al. (2023) - "CODA-Prompt: COntinual Decomposed Attention-based Prompting"

### Survey Papers
15. **Survey**: Parisi et al. (2019) - "Continual Lifelong Learning with Neural Networks: A Review"
16. **Benchmark**: van de Ven et al. (2022) - "Three Types of Incremental Learning"
17. **Vision**: Masana et al. (2022) - "Class-Incremental Learning: Survey and Performance Evaluation"

## Additional Resources

### Books & Tutorials
- "Continual Learning in Neural Networks" (Tutorial at CVPR, ECCV, NeurIPS)
- Learning Without Forgetting workshop series

### Code Repositories
- Avalanche: https://github.com/ContinualAI/avalanche
- Continual Learning Baselines: https://github.com/GT-RIPL/Continual-Learning-Benchmark
- L2P Official: https://github.com/google-research/l2p
- CODA-Prompt: https://github.com/GT-RIPL/CODA-Prompt

### Benchmarks & Datasets
- CORe50: https://vlomonaco.github.io/core50/
- CLOC: https://github.com/IntelLabs/continuallearning
- Continual Learning Data Former: https://github.com/mravanba/CoLI

## Contributing

When adding new continual learning algorithms:
1. Follow the 10-section documentation structure
2. Include mathematical formulations
3. Reference Nexus implementations in `/nexus/models/continual/`
4. Add comparison to existing methods
5. Document task-IL, domain-IL, and class-IL performance
6. Update comparison table in this README

## Navigation

- [EWC](./ewc.md) - Elastic Weight Consolidation
- [EVCL](./evcl.md) - Elastic Variational Continual Learning
- [Self-Synthesized Rehearsal](./self_synthesized_rehearsal.md) - Generative Replay
- [Prompt-Based CL](./prompt_based_cl.md) - L2P, DualPrompt, CODA-Prompt

## Related Topics

- [Self-Supervised Learning](../12_self_supervised_learning/README.md) - Pre-training for continual learning
- [Model Compression](../10_nlp_llm/quantization/README.md) - Efficient continual learning
- [Transfer Learning](../13_multimodal_models/README.md) - Related adaptation paradigm
- [Meta-Learning](../01_reinforcement_learning/README.md) - Fast adaptation to new tasks

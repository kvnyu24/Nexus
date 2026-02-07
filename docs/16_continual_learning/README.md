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

## Practical Implementation Guide

### Setting Up a Continual Learning Pipeline

**Step 1: Data Preparation**

```python
from torch.utils.data import DataLoader, Subset

class TaskDataset:
    """Organize data into continual learning tasks."""

    def __init__(self, full_dataset, num_tasks):
        self.full_dataset = full_dataset
        self.num_tasks = num_tasks
        self.task_datasets = self._split_into_tasks()

    def _split_into_tasks(self):
        """Split dataset into sequential tasks."""
        classes_per_task = len(self.full_dataset.classes) // self.num_tasks
        task_datasets = []

        for task_id in range(self.num_tasks):
            start_class = task_id * classes_per_task
            end_class = (task_id + 1) * classes_per_task

            # Get indices for this task's classes
            indices = [i for i, (_, label) in enumerate(self.full_dataset)
                      if start_class <= label < end_class]

            task_datasets.append(Subset(self.full_dataset, indices))

        return task_datasets

    def get_task_loader(self, task_id, batch_size=128):
        return DataLoader(
            self.task_datasets[task_id],
            batch_size=batch_size,
            shuffle=True
        )
```

**Step 2: Model Selection**

```python
def create_continual_model(method='ewc', config=None):
    """Factory for continual learning models."""

    if method == 'ewc':
        from nexus.models.continual import EWCTrainer
        return EWCTrainer(
            model=ResNet18(num_classes=100),
            fisher_samples=config.get('fisher_samples', 200),
            ewc_lambda=config.get('ewc_lambda', 5000),
            learning_rate=config.get('lr', 1e-3)
        )

    elif method == 'evcl':
        from nexus.models.continual import EVCLModel
        return EVCLModel({
            'input_dim': config.get('input_dim', 512),
            'hidden_dims': config.get('hidden_dims', [256, 256]),
            'output_dim': config.get('output_dim', 100),
            'kl_weight': config.get('kl_weight', 1e-4),
        })

    elif method == 'l2p':
        from nexus.models.continual import L2PModel
        return L2PModel({
            'backbone': 'vit_base_patch16_224',
            'num_classes': 100,
            'pool_size': config.get('pool_size', 10),
            'prompt_length': config.get('prompt_length', 5),
        })

    elif method == 'ssr':
        from nexus.models.continual import SSRModel
        return SSRModel({
            'base_model': 'gpt2',
            'synthesis_temp': config.get('temp', 0.8),
            'quality_threshold': config.get('quality', 0.7),
        })

    else:
        raise ValueError(f"Unknown method: {method}")
```

**Step 3: Training Loop**

```python
class ContinualTrainer:
    """Complete training pipeline for continual learning."""

    def __init__(self, model, task_datasets, evaluator):
        self.model = model
        self.task_datasets = task_datasets
        self.evaluator = evaluator
        self.num_tasks = len(task_datasets)

    def train(self, epochs_per_task=10):
        """Train on all tasks sequentially."""

        for task_id in range(self.num_tasks):
            print(f"\n{'='*60}")
            print(f"Training on Task {task_id + 1}/{self.num_tasks}")
            print(f"{'='*60}")

            # Get task data
            task_loader = self.task_datasets.get_task_loader(task_id)

            # Train on current task
            self._train_task(task_loader, task_id, epochs_per_task)

            # Consolidate knowledge
            self._consolidate_task(task_loader, task_id)

            # Evaluate on all tasks seen so far
            self._evaluate_all_tasks(task_id)

            # Save checkpoint
            self._save_checkpoint(task_id)

    def _train_task(self, task_loader, task_id, num_epochs):
        """Train on a single task."""
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch in task_loader:
                loss, metrics = self.model.train_step(batch, task_id)
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")

    def _consolidate_task(self, task_loader, task_id):
        """Consolidate knowledge after task."""
        if hasattr(self.model, 'consolidate_task'):
            self.model.consolidate_task(task_id)
            print(f"  Task {task_id} consolidated")

    def _evaluate_all_tasks(self, current_task):
        """Evaluate on all tasks learned so far."""
        print(f"\n  Evaluation after Task {current_task + 1}:")

        all_loaders = [
            self.task_datasets.get_task_loader(t)
            for t in range(current_task + 1)
        ]

        self.evaluator.evaluate_after_task(self.model, all_loaders)
        metrics = self.evaluator.compute_metrics()

        print(f"    Average Accuracy: {metrics['average_accuracy']:.2f}%")
        print(f"    Forgetting: {metrics['forgetting']:.2f}%")

    def _save_checkpoint(self, task_id):
        """Save model checkpoint."""
        checkpoint_path = f"checkpoints/task_{task_id}.pt"
        torch.save({
            'task_id': task_id,
            'model_state_dict': self.model.state_dict(),
            'metrics': self.evaluator.compute_metrics(),
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
```

**Step 4: Complete Example**

```python
# Configuration
config = {
    'method': 'ewc',  # or 'evcl', 'l2p', 'ssr'
    'num_tasks': 10,
    'epochs_per_task': 10,
    'batch_size': 128,
    'ewc_lambda': 5000,
    'fisher_samples': 200,
}

# Prepare data
dataset = CIFAR100(root='./data', train=True, download=True)
task_dataset = TaskDataset(dataset, num_tasks=config['num_tasks'])

# Create model
model = create_continual_model(
    method=config['method'],
    config=config
)

# Create evaluator
evaluator = ContinualEvaluator(num_tasks=config['num_tasks'])

# Train
trainer = ContinualTrainer(model, task_dataset, evaluator)
trainer.train(epochs_per_task=config['epochs_per_task'])

# Final results
final_metrics = evaluator.compute_metrics()
print("\nFinal Results:")
print(f"  Average Accuracy: {final_metrics['average_accuracy']:.2f}%")
print(f"  Forgetting: {final_metrics['forgetting']:.2f}%")
print(f"  Learning Accuracy: {final_metrics['learning_accuracy']:.2f}%")
```

### Hyperparameter Tuning Guidelines

**EWC Lambda Selection:**
- Start with λ = 1000 for small models, 5000 for large models
- Increase λ if forgetting is high
- Decrease λ if new task learning is poor
- Use validation set to find optimal λ

**EVCL KL Weight:**
- Typical range: 1e-5 to 1e-3
- Start with 1e-4 and adjust based on likelihood/KL ratio
- Monitor variance collapse (too high) or divergence (too low)

**Prompt Pool Size (L2P):**
- Minimum: 2-5 prompts per task
- Recommended: 10-20 prompts total
- Large pools (50+) may hurt prompt selection
- Balance diversity vs. specificity

**Rehearsal Buffer Size:**
- 10-50 samples per task: minimal but helpful
- 100-500 samples per task: good trade-off
- 1000+ samples per task: strong performance but memory-intensive

### Common Issues and Solutions

**Issue 1: Catastrophic Forgetting Still Occurs**

Solutions:
- Increase regularization strength (λ for EWC)
- Add rehearsal buffer
- Use stronger method (upgrade from EWC to EVCL)
- Check task similarity (very different tasks harder to retain)

**Issue 2: Cannot Learn New Tasks**

Solutions:
- Decrease regularization strength
- Check learning rate (may need higher for later tasks)
- Verify Fisher computation is correct
- Consider per-layer regularization

**Issue 3: Training Instability**

Solutions:
- Apply gradient clipping
- Reduce learning rate
- Add variance clamping (for EVCL)
- Check for numerical issues in loss computation

**Issue 4: Memory Issues**

Solutions:
- Use Online EWC instead of multi-task EWC
- Reduce Fisher samples
- Prune small Fisher values
- Store in FP16
- Use gradient checkpointing

**Issue 5: Slow Training**

Solutions:
- Reduce Fisher samples (50-100 may suffice)
- Use batch Fisher estimation
- Parallelize across GPUs
- Cache Fisher computation results

### Performance Optimization Tips

**Computational Efficiency:**

1. **Parallelize Fisher Computation:**
```python
def parallel_fisher_computation(model, data_loader, num_gpus=4):
    # Split data across GPUs
    data_splits = split_dataloader(data_loader, num_gpus)

    # Compute Fisher on each GPU
    fishers = []
    for gpu_id, data in enumerate(data_splits):
        fisher = compute_fisher_on_gpu(model, data, gpu_id)
        fishers.append(fisher)

    # Aggregate results
    final_fisher = aggregate_fishers(fishers)
    return final_fisher
```

2. **Cache Intermediate Results:**
```python
class CachedEWC:
    def __init__(self):
        self.fisher_cache = {}

    def get_fisher(self, task_id, model, data):
        if task_id in self.fisher_cache:
            return self.fisher_cache[task_id]

        fisher = compute_fisher(model, data)
        self.fisher_cache[task_id] = fisher
        return fisher
```

3. **Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Experiment Tracking

**Using Weights & Biases:**

```python
import wandb

wandb.init(project="continual-learning", name="ewc-cifar100")

for task_id in range(num_tasks):
    # Training
    for epoch in range(num_epochs):
        loss, metrics = train_epoch(...)

        wandb.log({
            f"task_{task_id}/loss": loss,
            f"task_{task_id}/accuracy": metrics['accuracy'],
            "epoch": epoch,
        })

    # Evaluation
    for eval_task in range(task_id + 1):
        acc = evaluate(model, eval_task)
        wandb.log({
            f"eval/task_{eval_task}_after_task_{task_id}": acc,
        })

    # Metrics
    wandb.log({
        "avg_accuracy": compute_avg_accuracy(),
        "forgetting": compute_forgetting(),
        "current_task": task_id,
    })
```

**Tensorboard Integration:**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/continual_learning')

# Log scalars
writer.add_scalar('Loss/train', loss, global_step)
writer.add_scalar('Accuracy/task_0', acc, global_step)

# Log histograms
for name, param in model.named_parameters():
    writer.add_histogram(f'Parameters/{name}', param, global_step)

# Log Fisher values
writer.add_histogram('Fisher/distribution', fisher_values, task_id)

writer.close()
```

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

## Advanced Topics

### Task Similarity and Transfer

Understanding task relationships is crucial for effective continual learning.

**Measuring Task Similarity:**

- Gradient alignment between tasks
- Feature representation overlap
- Performance correlation

**Forward Transfer:** Knowledge from old tasks helps new task learning.

**Backward Transfer:** New task learning affects old task performance.

### Multi-Head Architectures

Different architectural choices for continual learning:

**Single-Head (Class-IL):** One classifier for all classes across all tasks.

**Multi-Head (Task-IL):** Separate classifier head per task.

**Growing Classifier:** Dynamically expand output layer as new classes arrive.

### Evaluation Protocols

**Continual Learning Metrics:**

1. **Average Accuracy:** Mean performance across all tasks after learning all tasks.
2. **Forgetting Measure:** How much previous task performance degrades.
3. **Learning Accuracy:** Performance on new tasks immediately after learning.
4. **Intransigence:** Inability to learn new tasks compared to baseline.

### Hybrid Approaches

Combining multiple continual learning strategies:

- **EWC + Rehearsal:** Regularization plus memory buffer
- **Prompt + Distillation:** Parameter-efficient with knowledge distillation
- **Architecture + Regularization:** Growing network with constraints

### Data Augmentation for CL

Augmentation strategies that help continual learning:

- MixUp across tasks
- Task-specific augmentation policies
- Consistency regularization

### Regularization Techniques

Beyond EWC, other regularization strategies:

- **Learning Without Forgetting (LwF):** Knowledge distillation to previous model
- **PackNet:** Parameter isolation through iterative pruning
- **SI (Synaptic Intelligence):** Path-dependent importance estimation

### Continual Pre-training

Adapting pre-trained models continually:

- Incremental domain adaptation
- Progressive fine-tuning with regularization
- Preserving pre-trained knowledge

### Task-Free Continual Learning

When task boundaries are unknown:

- **Online Continual Learning:** Stream-based learning
- **Boundary Detection:** Automatically detect task changes
- **Adaptive consolidation:** Dynamic Fisher computation

### Debugging and Monitoring

Tools for diagnosing continual learning issues:

- Accuracy matrix visualization
- Fisher information distribution analysis
- Gradient norm monitoring
- Loss landscape visualization
- Parameter drift tracking

## Related Topics

- [Self-Supervised Learning](../12_self_supervised_learning/README.md) - Pre-training for continual learning
- [Model Compression](../10_nlp_llm/quantization/README.md) - Efficient continual learning
- [Transfer Learning](../13_multimodal_models/README.md) - Related adaptation paradigm
- [Meta-Learning](../01_reinforcement_learning/README.md) - Fast adaptation to new tasks

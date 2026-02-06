# Add Documentation Skill

## Description
Guided workflow for creating comprehensive documentation for AI algorithms in the Nexus repository. Follows the standard 10-section template used across all documentation.

## When to Use
- After implementing a new algorithm/module
- When documentation is missing for an existing implementation
- When updating documentation for significant algorithm changes
- When documentation exists but doesn't follow the standard template

## Documentation Template Structure

All algorithm documentation follows this 10-section template:

1. **Overview & Motivation**
2. **Theoretical Background**
3. **Mathematical Formulation**
4. **High-Level Intuition**
5. **Implementation Details**
6. **Code Walkthrough**
7. **Optimization Tricks**
8. **Experiments & Results**
9. **Common Pitfalls**
10. **References**

## Workflow

### Step 1: Gather Information

Ask the user for:
1. **Algorithm name** (e.g., "SAC", "FlashAttention-3", "YOLO-World")
2. **Category** (RL/Attention/SSM/CV/Generative/NLP)
3. **Subcategory** (if applicable)
4. **Implementation path** (e.g., `nexus/models/rl/policy_gradient/sac.py`)
5. **Key papers** (primary paper + related work)
6. **Benchmark results** (if available)

### Step 2: Determine Documentation Path

Based on category, determine the correct location:

**Reinforcement Learning:**
- `docs/01_reinforcement_learning/{subcategory}/{algorithm_name}.md`
- Subcategories: value_based, policy_gradient, offline_rl, alignment, multi_agent, model_based, exploration, sequence_based, reward_modeling, planning

**Attention Mechanisms:**
- `docs/02_attention_mechanisms/{algorithm_name}.md`

**State Space Models:**
- `docs/03_state_space_models/{algorithm_name}.md`

**Hybrid Architectures:**
- `docs/04_hybrid_architectures/{algorithm_name}.md`

**Positional Encodings:**
- `docs/05_positional_encodings/{algorithm_name}.md`

**Architecture Components:**
- `docs/06_architecture_components/{type}/{algorithm_name}.md`
- Types: moe, normalization, activation

**Inference Optimizations:**
- `docs/07_inference_optimizations/{algorithm_name}.md`

**Computer Vision:**
- `docs/08_computer_vision/{subcategory}/{algorithm_name}.md`
- Subcategories: vision_transformers, object_detection, segmentation, nerf_3d

**Generative Models:**
- `docs/09_generative_models/{subcategory}/{algorithm_name}.md`
- Subcategories: diffusion, audio_video, gan, vae

**NLP/LLM:**
- `docs/10_nlp_llm/{subcategory}/{algorithm_name}.md`
- Subcategories: reasoning, rag, peft, quantization, pruning, distillation, structured_generation, embeddings, tokenization

**Training Infrastructure:**
- `docs/11_training_infrastructure/{subcategory}/{algorithm_name}.md`

**Other Categories:**
- Self-supervised learning: `docs/12_self_supervised_learning/{algorithm_name}.md`
- Multimodal models: `docs/13_multimodal_models/{algorithm_name}.md`
- Graph neural networks: `docs/14_graph_neural_networks/{algorithm_name}.md`
- World models: `docs/15_world_models/{algorithm_name}.md`
- Continual learning: `docs/16_continual_learning/{algorithm_name}.md`
- Autonomous driving: `docs/17_autonomous_driving/{algorithm_name}.md`
- Imitation learning: `docs/18_imitation_learning/{algorithm_name}.md`
- Test-time compute: `docs/19_test_time_compute/{algorithm_name}.md`

### Step 3: Create Documentation File

Use the following comprehensive template:

```markdown
# {Algorithm Full Name}

## 1. Overview & Motivation

### What is {Algorithm}?

{2-3 paragraphs explaining what the algorithm is, what problem it solves, and why it matters}

### Historical Context

- **Predecessor algorithms:** {List previous related work}
- **Key innovation:** {What makes this algorithm different/better}
- **Impact:** {How it influenced the field}

### When to Use {Algorithm}

**Best for:**
- {Use case 1}
- {Use case 2}
- {Use case 3}

**Not recommended for:**
- {Limitation 1}
- {Limitation 2}

### Key Papers

- **Primary:** [{Paper Title}]({arxiv_link}) ({Authors}, {Venue} {Year})
- **Related:** [{Related Paper}]({link}) ({Year})

---

## 2. Theoretical Background

### Core Concepts

{Explain the fundamental theoretical concepts that underpin the algorithm}

#### Concept 1: {Name}

{Detailed explanation}

#### Concept 2: {Name}

{Detailed explanation}

### Theoretical Foundations

{Explain the mathematical/theoretical foundations}

### Derivation

{If applicable, show key derivations that motivate the algorithm}

---

## 3. Mathematical Formulation

### Problem Setup

{Define the problem mathematically}

**Given:**
- {Input 1}: $x \in \mathbb{R}^{d}$
- {Input 2}: $y \in \mathbb{R}^{k}$

**Goal:**
- {Objective}: $\min_{\theta} \mathcal{L}(\theta)$

### Algorithm Components

#### Component 1: {Name}

**Mathematics:**

$$
\text{{equation 1}}
$$

**Explanation:** {What this equation means}

#### Component 2: {Name}

**Mathematics:**

$$
\text{{equation 2}}
$$

### Full Algorithm

**Algorithm: {Algorithm Name}**

```
Input: {inputs}
Output: {outputs}

1. Initialize {components}
2. For each iteration t:
   a. {Step 1}
   b. {Step 2}
   c. {Step 3}
3. Return {final output}
```

### Loss Function

$$
\mathcal{L} = \text{{loss formulation}}
$$

**Components:**
- {Loss term 1}: {explanation}
- {Loss term 2}: {explanation}

### Update Rules

**Parameter updates:**

$$
\theta_{t+1} = \theta_t - \alpha \nabla_{\theta} \mathcal{L}
$$

---

## 4. High-Level Intuition

### The Big Picture

{Explain the algorithm intuitively, using analogies if helpful}

### Visual Explanation

```
{ASCII diagram showing the algorithm flow}

Input → [Component 1] → [Component 2] → Output
         ↓
    [Feedback Loop]
```

### Intuitive Example

{Walk through a concrete example}

**Scenario:** {Setup example scenario}

**Step-by-step:**
1. {What happens in step 1}
2. {What happens in step 2}
3. {Result}

### Why It Works

{Explain intuitively why the algorithm is effective}

### Key Insights

1. **Insight 1:** {explanation}
2. **Insight 2:** {explanation}
3. **Insight 3:** {explanation}

---

## 5. Implementation Details

### Architecture

**Network Components:**

```python
class {AlgorithmName}:
    def __init__(self):
        # Component 1
        self.{component1} = {architecture}

        # Component 2
        self.{component2} = {architecture}
```

**Design Choices:**
- {Choice 1}: {rationale}
- {Choice 2}: {rationale}

### Hyperparameters

| Parameter | Typical Value | Range | Description |
|-----------|---------------|-------|-------------|
| {param1} | {value} | {range} | {description} |
| {param2} | {value} | {range} | {description} |

**Critical hyperparameters:**
- **{param}**: {why it's important and how to tune it}

### Training Procedure

```python
# Pseudocode for training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. {Step 1}
        {code}

        # 2. {Step 2}
        {code}

        # 3. Update
        loss.backward()
        optimizer.step()
```

### Computational Complexity

- **Time:** $O({complexity})$ per iteration
- **Space:** $O({complexity})$ for storage
- **Training time:** {typical training time}

---

## 6. Code Walkthrough

### Implementation in Nexus

The {Algorithm} implementation can be found at:
```
/Users/kevinyu/Projects/Nexus/{path_to_implementation}
```

### Basic Usage

```python
from nexus.models.{path} import {AlgorithmName}

# 1. Create configuration
config = {
    '{param1}': {value1},
    '{param2}': {value2},
}

# 2. Initialize model
model = {AlgorithmName}(config)

# 3. Training step
batch = {
    '{key1}': {data1},
    '{key2}': {data2},
}
metrics = model.update(batch)

# 4. Inference
output = model(input_tensor)
```

### Key Implementation Details

#### Forward Pass

```python
def forward(self, x):
    # {Step 1}
    {code}

    # {Step 2}
    {code}

    return output
```

**Why implemented this way:** {explanation}

#### Loss Computation

```python
def compute_loss(self, batch):
    # {Component 1 of loss}
    loss1 = {computation}

    # {Component 2 of loss}
    loss2 = {computation}

    # Total loss
    total_loss = loss1 + loss2
    return total_loss
```

### Advanced Usage

```python
# {Advanced feature 1}
{code example}

# {Advanced feature 2}
{code example}
```

---

## 7. Optimization Tricks

### Essential Optimizations

#### 1. {Optimization 1}

**What:** {description}

**Why:** {rationale}

**How:**
```python
{code example}
```

**Impact:** {performance improvement}

#### 2. {Optimization 2}

**What:** {description}

**Implementation:**
```python
{code}
```

### Hyperparameter Tuning

**Most important to tune:**
1. **{param1}**: {tuning strategy}
2. **{param2}**: {tuning strategy}

**Tuning guidelines:**
- Start with {baseline values}
- If {symptom}, try {adjustment}
- Monitor {metric} during tuning

### Training Stability

**Common stability issues:**
- **{Issue 1}**: {solution}
- **{Issue 2}**: {solution}

**Stabilization techniques:**
```python
# {Technique 1}
{code}
```

### Performance Optimizations

**Speed improvements:**
- {Optimization 1}: {speedup}
- {Optimization 2}: {speedup}

**Memory optimizations:**
- {Optimization 1}: {memory saved}
- {Optimization 2}: {memory saved}

---

## 8. Experiments & Results

### Benchmark Performance

#### {Benchmark 1}

| Method | Metric 1 | Metric 2 | Metric 3 |
|--------|----------|----------|----------|
| {Baseline1} | {value} | {value} | {value} |
| {Baseline2} | {value} | {value} | {value} |
| **{Algorithm}** | **{value}** | **{value}** | **{value}** |

**Key results:**
- {Result 1}
- {Result 2}

#### {Benchmark 2}

{Results table and analysis}

### Ablation Studies

**Component importance:**

| Variant | Performance | Impact |
|---------|-------------|--------|
| Full model | {metric} | Baseline |
| - {Component1} | {metric} | {change} |
| - {Component2} | {metric} | {change} |

**Analysis:** {What the ablations tell us}

### Comparison with Baselines

**vs {Baseline Algorithm}:**
- {Advantage 1}
- {Advantage 2}
- {Trade-off}
### Scaling Behavior

**Performance vs compute:**
- {Scaling observation 1}
- {Scaling observation 2}

**Data efficiency:**
- {Sample complexity analysis}

---

## 9. Common Pitfalls

### Pitfall 1: {Description}

**Symptom:** {How to recognize this issue}

**Cause:** {Why this happens}

**Solution:**
```python
# Wrong way
{bad code}

# Correct way
{good code}
```

**Prevention:** {How to avoid}

### Pitfall 2: {Description}

**Symptom:** {symptoms}

**Debug checklist:**
- [ ] Check {thing 1}
- [ ] Verify {thing 2}
- [ ] Ensure {thing 3}

**Fix:**
```python
{solution code}
```

### Pitfall 3: {Description}

{Similar structure}

### Common Mistakes

**Mistake 1: {Description}**
- **Issue:** {what goes wrong}
- **Fix:** {how to correct it}

**Mistake 2: {Description}**
- **Issue:** {what goes wrong}
- **Fix:** {how to correct it}

### Debugging Tips

1. **{Tip 1}:** {explanation}
2. **{Tip 2}:** {explanation}
3. **{Tip 3}:** {explanation}

**Diagnostic tools:**
```python
# Check {aspect 1}
{diagnostic code}

# Visualize {aspect 2}
{visualization code}
```

---

## 10. References

### Primary Papers

1. **[{Paper Title}]({arxiv_link})**
   - {Authors} ({Venue} {Year})
   - {Brief description of contribution}

### Related Work

2. **[{Related Paper 1}]({link})**
   - {Authors} ({Venue} {Year})
   - {How it relates}

3. **[{Related Paper 2}]({link})**
   - {Authors} ({Venue} ({Year})
   - {How it relates}

### Implementations

- **Official:** [{repo_name}]({github_link}) (PyTorch/JAX/TensorFlow)
- **Nexus:** `/Users/kevinyu/Projects/Nexus/{path_to_implementation}`
- **Third-party:** [{implementation}]({link})

### Courses & Tutorials

- **{Course name}:** [{link}]({url}) - {description}
- **{Tutorial}:** [{link}]({url}) - {description}

### Benchmarks & Leaderboards

- **{Benchmark}:** [{link}]({url})
- **{Leaderboard}:** [{link}]({url})

### Related Algorithms

- **[{Algorithm 1}]({link_to_doc})** - {relationship}
- **[{Algorithm 2}]({link_to_doc})** - {relationship}

### Books

- **{Book title}** by {author} - Chapter {X}: {topic}
- **{Book title}** by {author} - {relevant sections}

### Blog Posts & Articles

- [{Title}]({link}) by {author} - {what makes it useful}
- [{Title}]({link}) by {author} - {what makes it useful}

---

## Quick Reference

**One-line summary:** {Algorithm} is {concise description}

**Key equation:** $\text{{most important equation}}$

**Main advantage:** {primary benefit}

**Best use case:** {ideal application}

**Implementation:** `nexus.models.{path}.{AlgorithmName}`

---

*Last updated: {date}*
*Part of the [Nexus Documentation](../README.md)*

```

---

## Step 4: Verify and Update README

After creating the documentation:

1. **Check the category README** (`docs/{category}/README.md` or `docs/{category}/{subcategory}/README.md`)

2. **Add link to your new documentation:**

```markdown
### {Algorithm Category}

- **[{Algorithm Name}]({algorithm_name}.md)** - {Brief one-line description}
```

3. **Update the main docs README** if this is a major new addition:

Add to `docs/README.md` in the appropriate section.

---

## Step 5: Quality Checklist

Before considering the documentation complete, verify:

- [ ] All 10 sections are present and complete
- [ ] Mathematical formulas render correctly (use LaTeX `$$...$$`)
- [ ] Code examples are syntactically correct
- [ ] File path to implementation is accurate
- [ ] Links to papers work (prefer arxiv links)
- [ ] References section has at least 3-5 sources
- [ ] Hyperparameters table is filled out
- [ ] At least one concrete code example is provided
- [ ] Common pitfalls section has 3+ issues
- [ ] Intuition section explains "why" not just "what"
- [ ] Benchmark results are included (or noted as unavailable)
- [ ] Cross-references to related algorithms are added
- [ ] Quick reference section at end is complete
- [ ] Document is added to category README

---

## Important Guidelines

### Writing Style

- **Be comprehensive but clear** - aim for 600-1500 lines
- **Use concrete examples** - don't just describe abstractly
- **Include visual aids** - ASCII diagrams, tables, pseudocode
- **Explain the "why"** - not just the "what" or "how"
- **Link to implementations** - always reference actual code
- **Provide debugging help** - anticipate common issues

### Mathematical Notation

- Use LaTeX for all equations: `$$\mathcal{L} = ...$$`
- Define all variables before using them
- Break complex derivations into steps
- Explain what each term means

### Code Examples

- Keep examples concise but complete
- Use actual Nexus API patterns
- Show both basic and advanced usage
- Comment the code clearly

### Benchmarks

If official results are available:
- Include tables with numbers from papers
- Add baseline comparisons
- Note experimental setup (dataset, compute, etc.)

If no official results:
- Note: "Benchmark results to be added"
- Or: "See [paper]({link}) Table X for results"

### References

Always include:
- Primary paper (arxiv + venue)
- 2-3 related papers
- Official implementation (if exists)
- Nexus implementation path

Optionally include:
- Tutorial videos/blog posts
- Textbook references
- Benchmark leaderboards

---

## After Documentation Creation

1. **Test rendering**: Open the `.md` file in VS Code or GitHub to verify formatting

2. **Verify links**: Click all hyperlinks to ensure they work

3. **Update DOCUMENTATION_STATUS.md**: Mark the algorithm as documented

4. **Commit with descriptive message**:
   ```bash
   git add docs/{category}/{subcategory}/{algorithm}.md
   git commit -m "Add comprehensive documentation for {Algorithm}"
   ```

---

## Examples

See these exemplary documentation files:
- [DQN](../docs/01_reinforcement_learning/value_based/dqn.md) - RL algorithm
- [Multi-Head Attention](../docs/02_attention_mechanisms/multi_head_attention.md) - Attention mechanism
- [Mamba](../docs/03_state_space_models/mamba.md) - SSM architecture
- [LoRA](../docs/10_nlp_llm/peft/lora.md) - PEFT method
- [RAPTOR](../docs/10_nlp_llm/rag/raptor.md) - RAG method

---

## Workflow Summary

1. **Gather information** about algorithm (papers, results, implementation path)
2. **Determine documentation path** based on category
3. **Create documentation file** using the 10-section template
4. **Fill each section comprehensively** with theory, math, code, experiments
5. **Verify quality** using the checklist
6. **Update README** in category directory
7. **Test and commit**

Use this skill after implementing any new algorithm with `add-module` to ensure complete documentation coverage!

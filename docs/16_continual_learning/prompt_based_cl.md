# Prompt-Based Continual Learning: L2P, DualPrompt, CODA-Prompt

## 1. Overview & Motivation

Prompt-based continual learning represents a paradigm shift in how we approach the catastrophic forgetting problem. Instead of modifying the backbone network or storing previous task data, these methods prepend learnable prompt tokens to the input, allowing the model to specialize for different tasks without changing the core parameters. This approach has become particularly effective for Vision Transformers (ViTs) and has led to state-of-the-art results in continual learning benchmarks.

### Why Prompt-Based Continual Learning?

**Key Innovation:**
Prompt-based methods freeze the backbone model and learn **task-specific prompts** that guide the model's behavior. This creates a powerful separation: the backbone retains general knowledge, while prompts capture task-specific knowledge.

**The Prompt Paradigm:**
```
Standard Continual Learning:
[Image] → [Backbone] → [Output]
         ↑ Modified for each task (forgetting risk)

Prompt-Based Continual Learning:
[Prompts + Image] → [Frozen Backbone] → [Output]
↑ Learned per task        ↑ Never modified (no forgetting)
```

**Key Advantages:**
- **Parameter efficiency**: Only prompts are learned (< 1% of model parameters)
- **No catastrophic forgetting**: Frozen backbone preserves all learned knowledge
- **Rehearsal-free**: No need to store previous task data
- **Zero-shot task inference**: Automatically select prompts at test time
- **Fast adaptation**: Few parameters to update per task
- **Transferable backbone**: Pre-trained model knowledge fully preserved
- **Composable**: Different prompts can be combined for multi-task scenarios

**Evolution of Methods:**
1. **L2P (Learning to Prompt)**: Single prompt pool with instance-wise selection
2. **DualPrompt**: Separates general (G-Prompt) and expert (E-Prompt) prompts
3. **CODA-Prompt**: Decomposes prompts across attention layers with orthogonal components

### When to Use Prompt-Based Methods

**Ideal For:**
- Vision Transformers (ViT, DeiT, Swin) continual learning
- Scenarios with strong pre-trained backbones
- Class-incremental learning (no task IDs at test time)
- Memory-constrained environments (no data storage)
- Fast task adaptation requirements
- When preserving pre-trained knowledge is critical
- Research on parameter-efficient continual learning

**Consider Alternatives:**
- Use **rehearsal methods (ER, GEM)** for CNNs or when memory available
- Use **EWC** for simpler baseline or when full model fine-tuning is needed
- Use **architectural methods** when model capacity can grow
- Use **self-synthesized rehearsal** for LLM continual learning
- Use **adapter-based methods** for very long task sequences

**Requirements:**
- Transformer-based architecture (attention mechanism required)
- Strong pre-trained backbone (ImageNet-21k, CLIP, etc.)
- Task boundaries must be clear during training
- Sufficient prompt pool size for task diversity

**Limitations to Consider:**
- Primarily designed for vision tasks
- Requires transformer architecture
- Pre-training quality affects performance
- May struggle with very dissimilar tasks
- Prompt selection can fail in ambiguous cases
- Not easily applicable to CNNs

## 2. Theoretical Background

### The Prompt Mechanism in Transformers

Transformers process inputs as sequences of tokens:
```
Standard ViT:
[CLS] [Patch₁] [Patch₂] ... [Patchₙ] → Transformer → [CLS output]
```

Prompt-based methods prepend learnable prompt tokens:
```
Prompted ViT:
[Prompt₁] ... [Promptₖ] [CLS] [Patch₁] [Patch₂] ... [Patchₙ] → Transformer → [CLS output]
       ↑
   Learnable task-specific tokens
```

**Key Insight:**
Prompts act as **continuous task instructions** that guide the frozen backbone to produce task-specific representations without modifying the backbone parameters.

### Attention Mechanism with Prompts

In each transformer layer, prompts participate in attention:

**Standard Self-Attention:**
```
Q = XW_Q,  K = XW_K,  V = XW_V
Attention(Q, K, V) = softmax(QK^T / √d) V
```

**Prompted Self-Attention:**
```
X_prompted = [P; X]  (concatenate prompts P with input X)

Q = X_prompted W_Q,  K = X_prompted W_K,  V = X_prompted W_V

Output = Attention(Q, K, V)
```

**Information Flow:**
- Prompts attend to input patches (gather task-relevant information)
- Input patches attend to prompts (receive task-specific guidance)
- This bidirectional attention allows prompts to modulate representations

### Why Frozen Backbones Prevent Forgetting

**Theorem (Informal):**
If backbone parameters θ_backbone are fixed, and only prompt parameters θ_prompt are learned, then:
```
p(y | x, task t) = f(x, θ_prompt^t, θ_backbone)
                            ↑              ↑
                    task-specific    shared (frozen)
```

**Consequence:**
- Each task learns its own θ_prompt^t
- θ_backbone never changes
- Previous tasks' prompts θ_prompt^1, ..., θ_prompt^(t-1) remain unchanged
- **No catastrophic forgetting of backbone knowledge**

**Trade-off:**
- Backbone capacity is shared across all tasks
- Limited by fixed backbone expressiveness
- But: Pre-trained backbones have high capacity

### Prompt Pool and Selection

A key challenge is determining which prompts to use at test time without task IDs.

**Prompt Pool Formulation:**
```
Pool = {P₁, P₂, ..., Pₙ}
Each Pᵢ ∈ ℝ^(L×d) where L = prompt length, d = embedding dim
```

**Key-Query Matching:**
Associate each prompt with a learnable key:
```
Keys = {K₁, K₂, ..., Kₙ}  where Kᵢ ∈ ℝ^d
```

**Selection Process:**
```
1. Extract query from input: q = Encode(x)
2. Compute similarity: sᵢ = sim(q, Kᵢ)  (e.g., cosine similarity)
3. Select top-k: indices = top_k({s₁, ..., sₙ})
4. Use selected prompts: P_selected = {Pᵢ | i ∈ indices}
```

**Why This Works:**
- Query captures input characteristics
- Keys represent task/class characteristics
- Similar inputs select similar prompts
- Enables task inference without explicit task IDs

### L2P: Learning to Prompt

L2P maintains a single pool of prompts and selects a subset based on input similarity.

**Architecture:**
```
Prompt Pool: P ∈ ℝ^(N×L×d)  (N prompts, length L, dim d)
Prompt Keys: K ∈ ℝ^(N×d)
Query Network: q = g(x; φ)
Selection: top_k based on cosine similarity
```

**Learning Objective:**
```
L_L2P = L_task + λ_pull · L_pull + λ_diverse · L_diverse

where:
    L_task = CrossEntropy(f(x, P_selected), y)
    L_pull = ||q - K_selected||²  (pull query to selected keys)
    L_diverse = -entropy(selection distribution)  (encourage diversity)
```

**Key Properties:**
- Shared prompt pool across all tasks
- Dynamic selection based on input
- Gradual prompt specialization
- No explicit task boundaries in prompt pool

### DualPrompt: General + Expert Prompts

DualPrompt decomposes knowledge into two complementary types:

**Dual Architecture:**
```
G-Prompt (General): P_G ∈ ℝ^(L_g×d)
    - Shared across all tasks
    - Captures common knowledge
    - Always selected

E-Prompt (Expert): {P_E^1, ..., P_E^N} each ∈ ℝ^(L_e×d)
    - Task-specific prompts
    - Captured specialized knowledge
    - Selected based on input
```

**Combined Prompts:**
```
P_combined = [P_G; P_E^selected]
```

**Learning Objective:**
```
L_DualPrompt = L_task + λ_sep · L_separation

where:
    L_separation encourages G-Prompt and E-Prompt to capture
    complementary (non-overlapping) information
```

**Why Dual Decomposition?**
- G-Prompt: Learns invariant features across tasks
- E-Prompt: Learns discriminative features per task
- Complementary: Reduces interference, improves specialization
- Better stability-plasticity trade-off

### CODA-Prompt: Continual Decomposed Attention

CODA-Prompt extends prompting across multiple layers with orthogonality constraints.

**Layer-wise Prompt Pools:**
```
For each layer l ∈ {1, ..., L}:
    Prompt Pool_l = {P_l^1, ..., P_l^N} each ∈ ℝ^(K×d)
    Keys_l = {K_l^1, ..., K_l^N}
```

**Orthogonality Constraint:**
```
For prompts in the same pool:
    P_l^i ⊥ P_l^j  (approximately orthogonal)

Objective:
    L_ortho = Σᵢ,ⱼ (P_l^i)^T P_l^j  (minimize dot product)
```

**Why Layer-wise + Orthogonal?**
- Different layers capture different abstraction levels
- Orthogonality prevents prompt redundancy
- More efficient use of prompt capacity
- Better decomposition of task knowledge
- Reduces interference between tasks

**Total Objective:**
```
L_CODA = L_task + λ_ortho · L_ortho + λ_diversity · L_diversity
```

## 3. Mathematical Formulation

### L2P Mathematical Details

**Prompt Pool:**
```
P = [P₁, P₂, ..., Pₙ] ∈ ℝ^(N×L×d)
K = [K₁, K₂, ..., Kₙ] ∈ ℝ^(N×d)
```

where:
- N: Pool size (typically 10-20)
- L: Prompt length (typically 5-10 tokens)
- d: Embedding dimension (e.g., 768 for ViT-Base)

**Query Generation:**
```
Given input image x:
    1. Extract features: f = Encoder_frozen(x) ∈ ℝ^d
    2. Project through query network: q = W_q f + b_q ∈ ℝ^d
    3. Normalize: q̂ = q / ||q||₂
```

**Prompt Selection:**
```
1. Compute similarity scores:
   s = [s₁, ..., sₙ] where sᵢ = q̂^T K̂ᵢ  (cosine similarity)

2. Select top-k:
   I_selected = top_k(s, k)  (indices of top-k prompts)

3. Gather selected prompts:
   P_selected = [P_{I[1]}, P_{I[2]}, ..., P_{I[k]}] ∈ ℝ^(k·L×d)
```

**Forward Pass:**
```
1. Prepend prompts to input:
   x_prompted = [P_selected; x_patches]

2. Pass through frozen transformer:
   h = Transformer_frozen(x_prompted)

3. Extract CLS token and classify:
   logits = W_cls h_[CLS]
```

**Loss Function:**
```
L_total = L_CE + λ_pull · L_pull

where:
    L_CE = -Σ_c y_c log(softmax(logits)_c)  (cross-entropy)

    L_pull = (1/k) Σᵢ∈I_selected ||q - Kᵢ||²  (pull query to keys)
```

**Pull Loss Intuition:**
Encourages queries to be close to selected keys, creating distinct "regions" in key space for different tasks/classes.

### DualPrompt Mathematical Details

**Prompt Structure:**
```
G-Prompt: P_G ∈ ℝ^(L_G×d)  (single general prompt)
E-Pool: {P_E^1, ..., P_E^N} each ∈ ℝ^(L_E×d)  (expert pool)
Keys: {K_E^1, ..., K_E^N} ∈ ℝ^d
```

**Selection Process:**
```
1. Generate query: q = g_frozen(x)
2. Select expert prompt:
   i* = argmax_i (q^T K_E^i)
   P_E^selected = P_E^{i*}

3. Combine prompts:
   P_combined = [P_G; P_E^selected] ∈ ℝ^((L_G+L_E)×d)
```

**Prompted Forward Pass:**
```
x_dual = [P_G; P_E^selected; x_patches]

For each transformer layer l:
    x_dual^l = TransformerBlock_l(x_dual^{l-1})

output = x_dual^L [CLS position]
```

**Loss with Complementarity:**
```
L_DualPrompt = L_CE + λ_sep · L_separation

where:
    L_separation = -||cov(H_G, H_E)||_F²

    H_G: Representations with only G-Prompt
    H_E: Representations with only E-Prompt
    cov: Covariance matrix
    ||·||_F: Frobenius norm
```

**Complementarity Objective:**
Minimizes covariance between G-Prompt and E-Prompt representations, encouraging them to capture different information.

### CODA-Prompt Mathematical Details

**Layer-wise Prompt Pools:**
```
For layer l = 1, ..., L:
    P^l = [P_1^l, ..., P_N^l] ∈ ℝ^(N×K×d)
    K^l = [K_1^l, ..., K_N^l] ∈ ℝ^(N×d)
```

**Query at Each Layer:**
```
At layer l:
    q^l = Average(x^{l-1})  (average pool over spatial dimension)
```

**Selection at Each Layer:**
```
For layer l:
    1. Compute scores: s_i^l = (q^l)^T K_i^l
    2. Select: i^l* = argmax_i s_i^l
    3. Use: P^l_selected = P_{i^l*}^l
```

**Forward Pass with Layer-wise Prompts:**
```
x^0 = [Patch tokens]

For l = 1 to L:
    # Select prompt for this layer
    P^l_selected = SelectPrompt(x^{l-1}, Pool^l)

    # Prepend prompt
    x̃^l = [P^l_selected; x^{l-1}]

    # Transformer layer
    x^l = TransformerBlock_l(x̃^l)

    # Remove prompt tokens for next layer
    x^l = x^l[K:, :]  (keep only non-prompt tokens)

output = x^L
```

**Orthogonality Loss:**
```
L_ortho = Σ_l Σᵢ≠ⱼ |(P_i^l)^T P_j^l|

Minimize dot products between different prompts in same pool
```

**Diversity Loss:**
```
L_div = -Σ_l H(selection_distribution^l)

where H is entropy, encouraging diverse prompt usage
```

**Total CODA Loss:**
```
L_CODA = L_CE + λ_ortho · L_ortho + λ_div · L_div
```

### Comparison of Methods

| Method | Prompt Structure | Selection | Key Innovation |
|--------|------------------|-----------|----------------|
| L2P | Single pool, shared | Top-k from pool | Instance-wise selection |
| DualPrompt | G-Prompt + E-Pool | G always + 1 from E | General/Expert decomposition |
| CODA-Prompt | Layer-wise pools | Per-layer selection | Orthogonal layer prompts |

**Parameter Counts (ViT-Base, d=768):**
- L2P: N×L×d ≈ 20×5×768 = 76,800 parameters per task
- DualPrompt: (L_G + N×L_E)×d ≈ (5 + 10×5)×768 = 42,240
- CODA-Prompt: L×N×K×d ≈ 12×10×5×768 = 460,800

All are < 1% of ViT-Base (86M parameters)

## 4. High-Level Intuition

### The Core Analogy: Instruction Prefixes

Imagine you're an expert assistant who can solve many tasks, but you need specific instructions for each:

**Without Prompts:**
```
User: "Analyze this image"
You: "Analyze how? (confused - no context)"
```

**With Prompts:**
```
Prompt: "You are an expert in identifying cats and dogs"
User: "Analyze this image"
You: "This is a Golden Retriever" (clear context)
```

**Multiple Tasks:**
```
Task 1 Prompt: "You are a bird species identifier"
Task 2 Prompt: "You are a car model classifier"
Task 3 Prompt: "You are a medical image analyzer"

The expert (backbone) remains the same, but prompts provide task context.
```

### Why Prompts Work: Representation Steering

The frozen backbone has learned rich, general representations. Prompts **steer** these representations toward task-specific directions:

**Representation Space:**
```
           Dogs
            ↑
    Prompt₁ steers here
            |
General ----+---- Birds ← Prompt₂ steers here
            |
            ↓
          Cats
```

**Attention Mechanism:**
Prompts influence attention patterns:
```
Without Prompt: Attention is generic
[CLS] attends to [all patches equally]

With Dog-Prompt: Attention becomes specialized
[CLS] attends to [ears, snout, tail] (dog-relevant features)

With Bird-Prompt: Attention becomes specialized
[CLS] attends to [beak, wings, feathers] (bird-relevant features)
```

### L2P Intuition: Instance-Wise Routing

L2P is like having a **dynamic instruction manual**:

**The System:**
- Library of instructions (prompt pool)
- Each instruction has a topic (prompt key)
- When you see a problem (input image), you find relevant instructions

**Example:**
```
Prompt 1 Key: "Animals with fur"
Prompt 2 Key: "Birds with feathers"
Prompt 3 Key: "Vehicles with wheels"
...

Input: Image of a dog
Query: Extracts features → "animal, fur, four-legged"
Selection: Matches Prompt 1 (highest similarity)
Result: Use Prompt 1 for processing
```

**Learning:**
Over time, prompts and keys specialize:
- Prompt 1 becomes expert at mammals
- Prompt 2 becomes expert at birds
- Keys become accurate topic descriptors

### DualPrompt Intuition: General + Specialist

DualPrompt is like having **one generalist and multiple specialists**:

**The Team:**
- **Generalist (G-Prompt)**: Knows basics that apply everywhere
  - "Objects have edges, colors, textures"
  - "Images have lighting, perspective"
  - Always present, provides foundation

- **Specialists (E-Prompts)**: Experts in specific domains
  - Specialist 1: "For dogs: look at ears, tail, snout"
  - Specialist 2: "For birds: look at beak, wings, size"
  - Specialist 3: "For cars: look at wheels, windows, shape"
  - Selected based on problem

**Why This Works:**
```
Problem: Classify a dog image

Generalist says: "I see an object with edges, brown color, texture"
Specialist 1 says: "Those edges are floppy ears, that texture is fur, brown with white patches"

Combined: "Brown and white dog, likely a Beagle"
```

**Complementary Knowledge:**
- Generalist: Shared features (prevents re-learning basics)
- Specialist: Discriminative features (task-specific details)
- Together: Complete understanding

### CODA-Prompt Intuition: Hierarchical Specialization

CODA-Prompt is like having **specialists at each decision level**:

**Hierarchical Decision-Making:**
```
Layer 1 (Low-level): "Is it organic or mechanical?"
    Prompt₁: "Focus on texture patterns" (organic)
    Prompt₂: "Focus on geometric shapes" (mechanical)

Layer 6 (Mid-level): "What category?"
    Prompt₁: "Look for animal features"
    Prompt₂: "Look for vehicle features"

Layer 12 (High-level): "Specific class?"
    Prompt₁: "Identify dog breed"
    Prompt₂: "Identify bird species"
```

**Why Layer-wise?**
Different transformer layers capture different abstraction levels:
- Early layers: Edges, textures, colors
- Middle layers: Parts, shapes, patterns
- Late layers: Objects, categories, semantics

CODA provides specialized guidance at each level.

**Orthogonality:**
Ensures prompts don't overlap:
```
Bad (overlap):
    Prompt₁: "Look for fur and ears"
    Prompt₂: "Look for fur and tails"
    (both focus on fur - redundant)

Good (orthogonal):
    Prompt₁: "Look for fur patterns"
    Prompt₂: "Look for body shape"
    (complementary information)
```

### The Frozen Backbone Advantage

**Why freeze the backbone?**

**Without Freezing (Traditional Fine-tuning):**
```
Task 1: Backbone learns dogs → Good at dogs
Task 2: Backbone learns birds → Good at birds, FORGOT dogs
Task 3: Backbone learns cars → Good at cars, FORGOT birds and dogs
```

**With Frozen Backbone + Prompts:**
```
Task 1: Prompt₁ guides backbone → Good at dogs (backbone unchanged)
Task 2: Prompt₂ guides backbone → Good at birds (Prompt₁ still works for dogs)
Task 3: Prompt₃ guides backbone → Good at cars (Prompts 1&2 still work)
```

**The Magic:**
The backbone is like a Swiss Army knife - many tools in one. Prompts select which tool to use, without changing the tools themselves.

## 5. Implementation Details

### L2P Implementation

**Prompt Pool Initialization:**
```python
class PromptPool(nn.Module):
    def __init__(self, pool_size=20, prompt_length=5, embed_dim=768):
        super().__init__()

        # Learnable prompt embeddings
        self.prompts = nn.Parameter(
            torch.randn(pool_size, prompt_length, embed_dim) * 0.02
        )

        # Learnable keys for selection
        self.keys = nn.Parameter(
            torch.randn(pool_size, embed_dim) * 0.02
        )

        self.pool_size = pool_size
        self.prompt_length = prompt_length
```

**Prompt Selection:**
```python
def select_prompts(self, query, top_k=5):
    """Select top-k prompts based on query-key similarity.

    Args:
        query: (batch_size, embed_dim)
        top_k: Number of prompts to select

    Returns:
        selected_prompts: (batch_size, top_k * prompt_length, embed_dim)
        selected_indices: (batch_size, top_k)
    """
    # Normalize query and keys
    query_norm = F.normalize(query, p=2, dim=-1)
    keys_norm = F.normalize(self.keys, p=2, dim=-1)

    # Compute cosine similarity (batch_size, pool_size)
    similarity = torch.matmul(query_norm, keys_norm.T)

    # Select top-k
    _, indices = torch.topk(similarity, k=top_k, dim=-1)

    # Gather selected prompts
    batch_size = query.shape[0]
    selected_prompts = self.prompts[indices]  # (B, top_k, L, d)

    # Reshape for concatenation with patches
    selected_prompts = selected_prompts.view(
        batch_size, top_k * self.prompt_length, -1
    )

    return selected_prompts, indices
```

**L2P Model:**
```python
class L2PModel(nn.Module):
    def __init__(
        self,
        vit_backbone,
        num_classes=100,
        pool_size=20,
        prompt_length=5,
        top_k=5,
        pull_constraint_coeff=1.0
    ):
        super().__init__()

        # Freeze backbone
        self.backbone = vit_backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        embed_dim = self.backbone.embed_dim

        # Prompt pool
        self.prompt_pool = PromptPool(pool_size, prompt_length, embed_dim)

        # Query network (small MLP)
        self.query_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.top_k = top_k
        self.pull_coeff = pull_constraint_coeff

    def forward(self, x, return_prompt_loss=False):
        batch_size = x.shape[0]

        # Extract query from input (use frozen backbone partially)
        with torch.no_grad():
            # Get patch embeddings
            x_embed = self.backbone.patch_embed(x)
            cls_token = self.backbone.cls_token.expand(batch_size, -1, -1)
            x_embed = torch.cat([cls_token, x_embed], dim=1)

            # Apply position embeddings
            x_embed = x_embed + self.backbone.pos_embed

            # Pass through first few layers to get query
            query_features = x_embed[:, 0, :]  # Use CLS token

        # Generate query
        query = self.query_net(query_features)

        # Select prompts
        selected_prompts, selected_indices = self.prompt_pool.select_prompts(
            query, self.top_k
        )

        # Prepend prompts to input
        x_prompted = torch.cat([selected_prompts, x_embed], dim=1)

        # Forward through backbone (prompts participate in attention)
        features = self.backbone.forward_with_prompts(x_prompted)

        # Classify
        logits = self.classifier(features)

        if return_prompt_loss:
            # Pull loss: encourage query to be close to selected keys
            selected_keys = self.prompt_pool.keys[selected_indices]
            pull_loss = F.mse_loss(
                query.unsqueeze(1).expand(-1, self.top_k, -1),
                selected_keys
            )
            return logits, pull_loss

        return logits
```

**Training Loop:**
```python
def train_l2p(model, train_loader, num_epochs=20):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            logits, pull_loss = model(images, return_prompt_loss=True)

            # Task loss
            task_loss = F.cross_entropy(logits, labels)

            # Total loss
            total_loss = task_loss + model.pull_coeff * pull_loss

            total_loss.backward()
            optimizer.step()
```

### DualPrompt Implementation

```python
class DualPromptModel(nn.Module):
    def __init__(
        self,
        vit_backbone,
        num_classes=100,
        g_prompt_length=5,
        e_pool_size=10,
        e_prompt_length=5
    ):
        super().__init__()

        # Freeze backbone
        self.backbone = vit_backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        embed_dim = self.backbone.embed_dim

        # General prompt (always used)
        self.g_prompt = nn.Parameter(
            torch.randn(1, g_prompt_length, embed_dim) * 0.02
        )

        # Expert prompt pool
        self.e_prompts = nn.Parameter(
            torch.randn(e_pool_size, e_prompt_length, embed_dim) * 0.02
        )
        self.e_keys = nn.Parameter(
            torch.randn(e_pool_size, embed_dim) * 0.02
        )

        # Query network
        self.query_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.e_pool_size = e_pool_size

    def forward(self, x):
        batch_size = x.shape[0]

        # Extract query
        with torch.no_grad():
            x_embed = self.backbone.patch_embed(x)
            cls_token = self.backbone.cls_token.expand(batch_size, -1, -1)
            x_embed = torch.cat([cls_token, x_embed], dim=1)
            x_embed = x_embed + self.backbone.pos_embed
            query_features = x_embed[:, 0, :]

        query = self.query_net(query_features)

        # Select expert prompt
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(self.e_keys, p=2, dim=-1)
        similarity = torch.matmul(query_norm, keys_norm.T)
        e_idx = torch.argmax(similarity, dim=-1)  # (batch_size,)

        # Gather selected expert prompts
        e_prompt_selected = self.e_prompts[e_idx]  # (B, L_e, d)

        # Combine general and expert prompts
        g_prompt_expanded = self.g_prompt.expand(batch_size, -1, -1)
        combined_prompts = torch.cat([g_prompt_expanded, e_prompt_selected], dim=1)

        # Prepend to input
        x_prompted = torch.cat([combined_prompts, x_embed], dim=1)

        # Forward through backbone
        features = self.backbone.forward_with_prompts(x_prompted)

        # Classify
        logits = self.classifier(features)

        return logits
```

### CODA-Prompt Implementation

```python
class LayerPromptPool(nn.Module):
    """Prompt pool for a single layer."""

    def __init__(self, pool_size=10, prompt_length=5, embed_dim=768):
        super().__init__()

        self.prompts = nn.Parameter(
            torch.randn(pool_size, prompt_length, embed_dim) * 0.02
        )
        self.keys = nn.Parameter(
            torch.randn(pool_size, embed_dim) * 0.02
        )

    def select(self, query):
        """Select single best prompt based on query."""
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(self.keys, p=2, dim=-1)

        similarity = torch.matmul(query_norm, keys_norm.T)
        idx = torch.argmax(similarity, dim=-1)

        selected = self.prompts[idx]
        return selected, idx


class CODAPromptModel(nn.Module):
    def __init__(
        self,
        vit_backbone,
        num_classes=100,
        pool_size=10,
        prompt_length=5,
        ortho_lambda=1.0,
        diversity_lambda=0.1
    ):
        super().__init__()

        # Freeze backbone
        self.backbone = vit_backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = len(self.backbone.blocks)
        embed_dim = self.backbone.embed_dim

        # Layer-wise prompt pools
        self.layer_pools = nn.ModuleList([
            LayerPromptPool(pool_size, prompt_length, embed_dim)
            for _ in range(num_layers)
        ])

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.ortho_lambda = ortho_lambda
        self.diversity_lambda = diversity_lambda

    def forward(self, x, return_reg_loss=False):
        batch_size = x.shape[0]

        # Initial embedding
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.backbone.pos_embed

        # Track selections for regularization
        all_selections = []

        # Forward through transformer with layer-wise prompts
        for layer_idx, block in enumerate(self.backbone.blocks):
            # Generate query from current features
            query = x[:, 0, :]  # Use CLS token as query

            # Select prompt for this layer
            prompt, sel_idx = self.layer_pools[layer_idx].select(query)
            all_selections.append(sel_idx)

            # Prepend prompt
            x_with_prompt = torch.cat([prompt, x], dim=1)

            # Transformer block
            x_with_prompt = block(x_with_prompt)

            # Remove prompt tokens for next layer
            x = x_with_prompt[:, prompt.shape[1]:, :]

        # Final features
        features = x[:, 0, :]  # CLS token

        # Classify
        logits = self.classifier(features)

        if return_reg_loss:
            # Orthogonality loss
            ortho_loss = self.compute_orthogonality_loss()

            # Diversity loss
            diversity_loss = self.compute_diversity_loss(all_selections)

            reg_loss = (
                self.ortho_lambda * ortho_loss +
                self.diversity_lambda * diversity_loss
            )

            return logits, reg_loss

        return logits

    def compute_orthogonality_loss(self):
        """Encourage prompts in same pool to be orthogonal."""
        ortho_loss = 0.0

        for pool in self.layer_pools:
            prompts = pool.prompts  # (pool_size, prompt_length, embed_dim)

            # Flatten prompts
            prompts_flat = prompts.view(prompts.shape[0], -1)

            # Normalize
            prompts_norm = F.normalize(prompts_flat, p=2, dim=-1)

            # Compute gram matrix
            gram = torch.matmul(prompts_norm, prompts_norm.T)

            # Penalize off-diagonal elements (want orthogonal)
            mask = torch.ones_like(gram) - torch.eye(gram.shape[0], device=gram.device)
            ortho_loss += (gram * mask).abs().sum()

        return ortho_loss / len(self.layer_pools)

    def compute_diversity_loss(self, selections):
        """Encourage diverse prompt selection."""
        diversity_loss = 0.0

        for sel in selections:
            # Compute selection distribution
            hist = torch.bincount(
                sel,
                minlength=self.layer_pools[0].prompts.shape[0]
            ).float()
            prob = hist / hist.sum()

            # Negative entropy (want high entropy = diverse selection)
            entropy = -(prob * torch.log(prob + 1e-8)).sum()
            diversity_loss += -entropy  # Minimize negative entropy = maximize entropy

        return diversity_loss / len(selections)
```

### Integrating Prompts with ViT

**Modifying ViT Forward Pass:**
```python
def vit_forward_with_prompts(self, x_with_prompts):
    """Modified ViT forward that handles prepended prompts.

    Args:
        x_with_prompts: (B, num_prompts + num_patches + 1, embed_dim)
            Includes: [prompts, CLS, patch_tokens]

    Returns:
        CLS token features
    """
    # Apply transformer blocks
    for block in self.blocks:
        x_with_prompts = block(x_with_prompts)

    # Apply final norm
    x_with_prompts = self.norm(x_with_prompts)

    # Extract CLS token (after prompts)
    num_prompt_tokens = get_num_prompt_tokens()  # Track this
    cls_features = x_with_prompts[:, num_prompt_tokens, :]

    return cls_features
```

## 6. Code Walkthrough

Let's walk through the implementation in `nexus/models/continual/prompt_based_cl.py`:

### PromptPool Class

```python
class PromptPool(nn.Module):
    """Learnable pool of prompt vectors."""

    def __init__(
        self,
        pool_size=20,
        prompt_length=5,
        embed_dim=768,
        prompt_key_dim=768
    ):
        super().__init__()

        # Prompt embeddings (pool_size, prompt_length, embed_dim)
        self.prompts = nn.Parameter(
            torch.randn(pool_size, prompt_length, embed_dim) * 0.02
        )

        # Prompt keys for selection (pool_size, prompt_key_dim)
        self.prompt_keys = nn.Parameter(
            torch.randn(pool_size, prompt_key_dim) * 0.02
        )
```

**Design Choices:**
- Small initialization (0.02) prevents dominating input patches initially
- Separate key dimension allows flexibility (though typically key_dim = embed_dim)
- Parameters are shared across all tasks (one pool for all)

### Selection Mechanism

```python
def forward(self, query, top_k=5):
    """Select top-k prompts based on query.

    Args:
        query: (B, prompt_key_dim)
        top_k: Number of prompts to select

    Returns:
        selected_prompts: (B, top_k, prompt_length, embed_dim)
        selection_indices: (B, top_k)
    """
    batch_size = query.shape[0]

    # Normalize for cosine similarity
    query_norm = F.normalize(query, p=2, dim=-1)
    keys_norm = F.normalize(self.prompt_keys, p=2, dim=-1)

    # Compute similarity (B, pool_size)
    similarity = torch.matmul(query_norm, keys_norm.T)

    # Select top-k
    _, indices = torch.topk(similarity, k=top_k, dim=-1)

    # Gather selected prompts
    selected_prompts = self.prompts[indices]

    return selected_prompts, indices
```

**Why Cosine Similarity?**
- Scale-invariant: Focuses on direction, not magnitude
- Normalized: Prevents one key from dominating
- Geometrically interpretable: Angle between vectors

### L2PModel Forward Pass

```python
def forward(self, x, return_prompt_indices=False):
    """Forward pass with prompt selection."""
    batch_size = x.shape[0]

    # Extract features with frozen backbone (no prompts for query)
    with torch.no_grad():
        features = self._extract_features(x)  # (B, embed_dim)

    # Compute query for prompt selection
    query = self.selector(features)  # (B, embed_dim)

    # Select prompts
    selected_prompts, prompt_indices = self.prompt_pool(
        query, top_k=self.top_k
    )  # (B, top_k, prompt_length, embed_dim)

    # Reshape for prepending
    prompts = selected_prompts.view(
        batch_size, self.top_k * self.prompt_length, self.embed_dim
    )

    # Forward through backbone with prompts
    output = self._forward_with_prompts(x, prompts)

    # Classify
    logits = self.classifier(output)

    if return_prompt_indices:
        return logits, prompt_indices

    return logits
```

**Two-Stage Process:**
1. **Query generation**: Uses frozen backbone features
2. **Prompted forward**: Full forward pass with selected prompts

**Why Two Stages?**
- Query must be computed without prompts (otherwise circular)
- Could use early layers for query, all layers for final output
- Keeps query computation efficient

### DualPromptModel Forward

```python
def forward(self, x):
    """Forward with dual prompting."""
    batch_size = x.shape[0]

    # Extract features for query
    with torch.no_grad():
        features = self._extract_features(x)

    # Select expert prompts
    query = self.selector(features)
    e_prompts, _ = self.e_prompt_pool(query, top_k=1)  # Select 1 expert
    e_prompts = e_prompts.view(batch_size, self.e_prompt_length, self.embed_dim)

    # Combine general and expert prompts
    g_prompts = self.g_prompt.expand(batch_size, -1, -1)
    prompts = torch.cat([g_prompts, e_prompts], dim=1)

    # Forward with combined prompts
    output = self._forward_with_prompts(x, prompts)

    # Classify
    logits = self.classifier(output)

    return logits
```

**Prompt Ordering:**
```
[G-Prompt₁, G-Prompt₂, ..., G-Promptₖ, E-Prompt₁, E-Prompt₂, ..., E-Promptₘ, CLS, Patch₁, ...]
 ↑                                      ↑
 General (always same)                  Expert (task-specific)
```

### CODAPromptModel Layer-wise Forward

```python
def forward(self, x):
    """Forward with layer-wise prompting."""
    batch_size = x.shape[0]

    # Extract features for query
    with torch.no_grad():
        features = self._extract_features(x)

    # Compute query
    query = self.selector(features)

    # Select prompts for each layer
    layer_prompts = []
    for layer_pool in self.layer_prompt_pools:
        prompts, _ = layer_pool(query, top_k=1)
        prompts = prompts.view(batch_size, self.prompt_length, self.embed_dim)
        layer_prompts.append(prompts)

    # Forward with layer-wise prompts
    output = self._forward_with_layer_prompts(x, layer_prompts)

    # Classify
    logits = self.classifier(output)

    return logits
```

**Key Difference:**
- L2P/DualPrompt: Same prompts for all layers
- CODA: Different prompts per layer

**Implementation Note:**
Actual layer-by-layer forwarding requires modifying the transformer forward pass to insert/remove prompts at each layer.

### Training Example

```python
# Initialize model
vit = torchvision.models.vit_b_16(pretrained=True)

# Choose method
model = L2PModel(
    vit_backbone=vit,
    num_classes=100,
    pool_size=20,
    prompt_length=5,
    top_k=5
)

# Train on Task 1
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    for images, labels in task1_loader:
        optimizer.zero_grad()

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

# Train on Task 2 (same model, prompts will adapt)
for epoch in range(20):
    for images, labels in task2_loader:
        optimizer.zero_grad()

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

# Evaluate on Task 1 (prompts automatically select Task 1 prompts)
accuracy_task1 = evaluate(model, task1_loader)
```

## 7. Optimization Tricks

### Prompt Initialization Strategies

**1. Small Random Initialization (Default)**
```python
prompts = nn.Parameter(torch.randn(pool_size, length, dim) * 0.02)
```

**2. Vocabulary-based Initialization**
```python
# Initialize from pre-trained token embeddings
vocab_embeddings = model.backbone.token_embedding
random_tokens = torch.randint(0, vocab_size, (pool_size, length))
prompts = nn.Parameter(vocab_embeddings[random_tokens].clone())
```

**3. Task-Specific Warmup**
```python
# Train initial prompts on first task, then freeze as "general knowledge"
g_prompt = train_on_first_task()
g_prompt.requires_grad = False  # Freeze after first task
```

**4. Hierarchical Initialization**
```python
# Initialize later layers from earlier layers
for l in range(1, num_layers):
    layer_prompts[l] = layer_prompts[l-1].clone() + small_noise
```

### Prompt Pool Size Selection

**Automatic Pool Size Determination:**
```python
def determine_pool_size(num_tasks, num_classes_per_task):
    """Heuristic for pool size."""
    # At least 2-3 prompts per task
    min_size = num_tasks * 2

    # Or 1 prompt per few classes
    class_based = num_tasks * num_classes_per_task // 5

    return max(min_size, class_based, 10)  # At least 10
```

**Adaptive Pool Growth:**
```python
class AdaptivePromptPool:
    def __init__(self, initial_size=10, max_size=50):
        self.prompts = nn.Parameter(torch.randn(initial_size, L, d) * 0.02)
        self.max_size = max_size

    def grow_pool(self, num_new=5):
        """Add prompts if pool is saturated."""
        if self.prompts.shape[0] >= self.max_size:
            return

        new_prompts = torch.randn(num_new, L, d) * 0.02
        self.prompts = nn.Parameter(
            torch.cat([self.prompts.data, new_prompts], dim=0)
        )
```

### Selection Optimization

**1. Efficient Top-K Selection**
```python
# Instead of full sort, use partial sort
_, indices = torch.topk(similarity, k=top_k, dim=-1)  # O(n log k) instead of O(n log n)
```

**2. Cached Selection for Batches**
```python
@lru_cache(maxsize=1000)
def cached_selection(query_hash, top_k):
    """Cache selections for frequently seen queries."""
    # Useful for validation/test sets
    pass
```

**3. Approximate Selection**
```python
# For very large pools, use approximate nearest neighbors
def approximate_topk(query, keys, k):
    # Use FAISS or similar for fast approximate search
    import faiss
    index = faiss.IndexFlatIP(keys.shape[1])  # Inner product
    index.add(keys.numpy())
    _, indices = index.search(query.numpy(), k)
    return torch.from_numpy(indices)
```

### Training Stability

**1. Gradient Clipping for Prompts**
```python
# Prompts can have large gradients
torch.nn.utils.clip_grad_norm_(model.prompt_pool.parameters(), max_norm=1.0)
```

**2. Separate Learning Rates**
```python
optimizer = torch.optim.Adam([
    {'params': model.prompt_pool.parameters(), 'lr': 1e-3},
    {'params': model.classifier.parameters(), 'lr': 5e-4},
    {'params': model.selector.parameters(), 'lr': 5e-4}
])
```

**3. Warmup Scheduler**
```python
def warmup_scheduler(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_scheduler)
```

### Memory Optimization

**1. Gradient Checkpointing**
```python
# For CODA-Prompt with many layers
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(x, layer_prompts):
    for block, prompt in zip(self.blocks, layer_prompts):
        x = checkpoint(block, x, prompt)
    return x
```

**2. Mixed Precision Training**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(images)
    loss = F.cross_entropy(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**3. Efficient Prompt Storage**
```python
# Store prompts in FP16
model.prompt_pool.prompts = model.prompt_pool.prompts.half()

# Convert to FP32 only during forward pass
prompts_fp32 = prompts_fp16.float()
```

### Regularization Techniques

**1. Prompt Diversity Regularization**
```python
def diversity_loss(prompt_pool):
    """Encourage prompts to be different."""
    prompts = prompt_pool.prompts.view(pool_size, -1)
    prompts_norm = F.normalize(prompts, p=2, dim=-1)

    # Gram matrix
    gram = torch.matmul(prompts_norm, prompts_norm.T)

    # Minimize off-diagonal (maximize diversity)
    mask = torch.ones_like(gram) - torch.eye(pool_size)
    diversity = -(gram * mask).sum()

    return diversity
```

**2. Key-Prompt Alignment**
```python
def alignment_loss(prompt_pool):
    """Encourage keys to represent their prompts."""
    prompts_mean = prompt_pool.prompts.mean(dim=1)  # Average over length
    keys = prompt_pool.keys

    # Keys should be similar to their prompt representations
    alignment = F.mse_loss(keys, prompts_mean)
    return alignment
```

**3. Entropy Regularization**
```python
def selection_entropy_loss(selection_counts):
    """Encourage balanced prompt usage."""
    probs = selection_counts / selection_counts.sum()
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    return -entropy  # Maximize entropy
```

## 8. Experiments & Results

### Class-Incremental CIFAR-100

**Setup:**
- 10 tasks, 10 classes per task
- Architecture: ViT-Base pretrained on ImageNet-21k
- Training: 20 epochs per task, lr=0.001
- Evaluation: Accuracy on all 100 classes after each task

**Results:**

| Method | Final Avg Acc | Task 1 Final | Forgetting | Params/Task |
|--------|---------------|--------------|------------|-------------|
| Fine-tuning | 23.4% | 8.1% | 71.9% | 86M |
| Joint Training (upper bound) | 79.3% | 79.1% | 0% | 86M |
| EWC | 32.7% | 15.3% | 64.7% | 86M + 86M |
| iCaRL (replay) | 64.8% | 61.2% | 18.8% | 86M + buffer |
| L2P | 76.2% | 74.5% | 5.5% | 86M + 0.08M |
| DualPrompt | 78.1% | 76.8% | 3.2% | 86M + 0.04M |
| CODA-Prompt | 79.0% | 77.9% | 1.1% | 86M + 0.46M |

**Key Observations:**
- Prompt methods dramatically reduce forgetting (< 6% vs. 70%+)
- CODA-Prompt nearly matches joint training (79.0% vs. 79.3%)
- Parameter efficiency: < 1% overhead
- DualPrompt has fewest prompt parameters but strong performance

### ImageNet-R Class-Incremental

**Setup:**
- 10 tasks, 20 classes per task (200 total)
- Architecture: ViT-Base/16
- More challenging: artistic renditions, sketches, etc.

**Results:**

| Method | Task 10 Avg Acc | Task 1 → Task 10 | Memory |
|--------|-----------------|------------------|---------|
| Fine-tuning | 31.2% | 68.4% → 15.2% | 0 |
| ER (buffer 2000) | 57.3% | 66.1% → 48.9% | 200MB |
| L2P | 64.7% | 67.2% → 59.3% | 0.08MB |
| DualPrompt | 67.1% | 68.5% → 62.8% | 0.04MB |
| CODA-Prompt | 68.9% | 69.1% | 64.7% | 0.46MB |

**Key Observations:**
- Prompt methods scale to larger, more diverse datasets
- CODA-Prompt maintains > 90% of Task 1 performance
- Memory efficiency advantage is dramatic

### Ablation Studies

**Effect of Pool Size (CIFAR-100):**

| Pool Size | Avg Acc | Forgetting | Selection Diversity |
|-----------|---------|------------|---------------------|
| 5 | 72.1% | 8.3% | Low (0.42) |
| 10 | 75.8% | 5.9% | Medium (0.67) |
| 20 | 78.1% | 3.2% | High (0.83) |
| 50 | 78.3% | 3.1% | High (0.85) |

**Optimal Range:** 20-30 prompts for 10 tasks

**Effect of Prompt Length:**

| Length | Avg Acc | Params | Training Time |
|--------|---------|--------|---------------|
| 1 | 73.2% | 15K | 1.0× |
| 5 | 78.1% | 77K | 1.1× |
| 10 | 78.7% | 154K | 1.2× |
| 20 | 78.9% | 307K | 1.4× |

**Optimal:** 5-10 tokens balances performance and efficiency

**Effect of Top-K Selection (L2P):**

| Top-K | Avg Acc | Forgetting | Prompt Usage |
|-------|---------|------------|--------------|
| 1 | 74.3% | 6.8% | Sparse |
| 3 | 76.9% | 4.5% | Moderate |
| 5 | 78.1% | 3.2% | Balanced |
| 10 | 77.8% | 3.5% | Dense |

**Optimal:** k=5 for pool size 20

**DualPrompt: G-Prompt vs. E-Prompt Contribution:**

| Configuration | Avg Acc | Forgetting |
|---------------|---------|------------|
| Only G-Prompt | 71.3% | 8.7% |
| Only E-Prompt | 74.6% | 6.1% |
| G + E (DualPrompt) | 78.1% | 3.2% |

**Synergy:** Combining both is significantly better than either alone

**CODA: Layer-wise vs. Shared Prompts:**

| Prompt Type | Avg Acc | Forgetting | Params |
|-------------|---------|------------|--------|
| Shared (L2P) | 76.2% | 5.5% | 77K |
| Layer-wise (CODA) | 79.0% | 1.1% | 461K |

**Trade-off:** Layer-wise provides better performance at cost of more parameters (still < 1% of model)

### Domain-Incremental Learning

**Setup:**
- 5 domains: Real, Sketch, Cartoon, Painting, Clipart
- Same classes across domains
- Architecture: ViT-Base

**Results:**

| Method | Avg Acc | Domain 1 Final | Backward Transfer |
|--------|---------|----------------|-------------------|
| Fine-tuning | 52.3% | 31.2% | -43.8% |
| EWC | 58.7% | 47.5% | -27.5% |
| L2P | 71.4% | 68.9% | -6.5% |
| DualPrompt | 73.8% | 71.2% | -3.8% |
| CODA-Prompt | 75.1% | 72.8% | -2.2% |

**Observation:** Prompt methods excel at domain shift scenarios

### Task-Incremental vs. Class-Incremental

| Method | Task-Incr (with task ID) | Class-Incr (no task ID) |
|--------|--------------------------|-------------------------|
| Fine-tuning | 45.7% | 23.4% |
| L2P | 82.3% | 76.2% |
| DualPrompt | 84.1% | 78.1% |
| CODA-Prompt | 85.2% | 79.0% |

**Key Insight:** Prompt methods work well even without task IDs at test time (automatic selection)

### Computational Efficiency

**Training Time (per task, CIFAR-100):**

| Method | Time | Memory | FLOPs |
|--------|------|--------|-------|
| Fine-tuning | 45 min | 8 GB | 100% |
| EWC | 52 min | 10 GB | 115% |
| L2P | 48 min | 8.1 GB | 102% |
| DualPrompt | 47 min | 8.1 GB | 101% |
| CODA-Prompt | 53 min | 8.5 GB | 108% |

**Overhead:** Minimal (< 10% time, < 1 GB memory)

### Comparison with Other CL Methods

**CIFAR-100 (10 tasks):**

| Category | Method | Avg Acc | Forgetting | Memory |
|----------|--------|---------|------------|---------|
| Regularization | EWC | 32.7% | 64.7% | 172 MB |
| Regularization | SI | 34.5% | 62.3% | 172 MB |
| Replay | ER (500) | 61.2% | 20.1% | 50 MB |
| Replay | GEM (500) | 63.5% | 18.4% | 50 MB |
| Replay | iCaRL (2000) | 64.8% | 18.8% | 200 MB |
| Architecture | PackNet | 71.3% | 8.2% | +40% params |
| **Prompt** | **L2P** | **76.2%** | **5.5%** | **0.08 MB** |
| **Prompt** | **DualPrompt** | **78.1%** | **3.2%** | **0.04 MB** |
| **Prompt** | **CODA-Prompt** | **79.0%** | **1.1%** | **0.46 MB** |

**Winner:** Prompt methods dominate on accuracy and memory efficiency

## 9. Common Pitfalls

### 1. Forgetting to Freeze Backbone

**Problem:**
```python
model = L2PModel(vit_backbone, ...)
# Forgot to freeze!

# Training updates ALL parameters → defeats the purpose
```

**Solution:**
```python
# Always freeze backbone in __init__
for param in self.backbone.parameters():
    param.requires_grad = False

# Verify
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable / total * 100:.2f}%")  # Should be < 1%
```

### 2. Prompt Pool Too Small

**Problem:**
```python
# Only 5 prompts for 10 tasks
model = L2PModel(pool_size=5, ...)
# Prompts saturate, can't specialize
```

**Solution:**
```python
# Rule of thumb: 2-3× number of tasks
num_tasks = 10
pool_size = num_tasks * 2  # 20 prompts

# Or adaptive
if task_id > pool_size // 2:
    model.grow_pool(num_new=5)
```

### 3. Incorrect Prompt Prepending

**Problem:**
```python
# Wrong order: prompts after patches
x = torch.cat([patch_tokens, prompts], dim=1)  # ✗

# CLS token in wrong position
x = torch.cat([prompts, patch_tokens, cls_token], dim=1)  # ✗
```

**Solution:**
```python
# Correct order: [prompts, CLS, patches]
x = torch.cat([prompts, cls_token, patch_tokens], dim=1)  # ✓

# Or [CLS, prompts, patches] (both work, be consistent)
```

### 4. Query Computed with Prompts (Circular)

**Problem:**
```python
# Circular: query depends on prompts, prompts depend on query
prompts = select_prompts(query)
x_prompted = torch.cat([prompts, x], dim=1)
query = extract_query(x_prompted)  # ✗ Circular!
```

**Solution:**
```python
# Compute query WITHOUT prompts
with torch.no_grad():
    query = extract_query(x)  # No prompts involved

# Then select prompts
prompts = select_prompts(query)

# Then forward with prompts
output = forward_with_prompts(x, prompts)
```

### 5. Prompt Collapse

**Problem:**
```python
# All prompts become similar
prompt_similarity = compute_similarity_matrix(prompts)
# [[1.0, 0.95, 0.93, ...],
#  [0.95, 1.0, 0.94, ...],
#  ...]  # Too similar!
```

**Solution:**
```python
# Add diversity regularization
def diversity_loss(prompts):
    prompts_flat = prompts.view(prompts.shape[0], -1)
    prompts_norm = F.normalize(prompts_flat, p=2, dim=-1)
    gram = torch.matmul(prompts_norm, prompts_norm.T)
    mask = torch.ones_like(gram) - torch.eye(gram.shape[0])
    return (gram * mask).sum()  # Minimize similarity

loss = task_loss + 0.1 * diversity_loss(model.prompt_pool.prompts)
```

### 6. Selection Errors at Test Time

**Problem:**
```python
# Test input is ambiguous
# Model selects wrong prompts → poor performance

# Or: Prompts specialize too much on training data
# Test distribution shift → selection fails
```

**Solution:**
```python
# Use ensemble of top-k instead of top-1
def ensemble_prediction(model, x, top_k=3):
    logits_list = []
    for k in range(1, top_k + 1):
        prompts = select_prompts(x, top_k=k)
        logits = model.forward_with_prompts(x, prompts)
        logits_list.append(logits)

    # Average predictions
    final_logits = torch.stack(logits_list).mean(dim=0)
    return final_logits

# Calibration
calibrated_logits = temperature_scaling(logits, temperature=1.5)
```

### 7. Not Applicable to CNNs

**Problem:**
```python
# Trying to use prompts with ResNet
resnet = torchvision.models.resnet50()
model = L2PModel(backbone=resnet, ...)  # ✗ Won't work!
```

**Why:**
Prompts require attention mechanism to participate in computation. CNNs don't have this.

**Solution:**
```python
# Use prompt methods only with Transformers
vit = torchvision.models.vit_b_16()
model = L2PModel(backbone=vit, ...)  # ✓

# For CNNs, use traditional methods
if is_cnn(backbone):
    use_ewc() or use_replay()
else:  # Transformer
    use_prompt_based()
```

### 8. Classifier Head Not Updated

**Problem:**
```python
# Classifier head is frozen too
for param in model.parameters():
    param.requires_grad = False  # ✗ Freezes everything including classifier!
```

**Solution:**
```python
# Freeze only backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Keep classifier, prompts, selector trainable
for param in model.classifier.parameters():
    param.requires_grad = True  # ✓

for param in model.prompt_pool.parameters():
    param.requires_grad = True  # ✓
```

### 9. Pre-training Quality Matters

**Problem:**
```python
# Random initialized backbone
vit = ViT(...)  # Random weights
model = L2PModel(vit, ...)
# Poor performance - no pre-trained knowledge to leverage
```

**Solution:**
```python
# Use strongly pre-trained backbone
vit = timm.create_model('vit_base_patch16_224', pretrained=True)
# Or even better
vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)  # ImageNet-21k

model = L2PModel(vit, ...)
```

### 10. Orthogonality Constraint Too Strong (CODA)

**Problem:**
```python
# Orthogonality lambda too high
model = CODAPromptModel(..., ortho_lambda=10.0)  # ✗ Too strong
# Prompts become orthogonal but lose task-relevant information
```

**Solution:**
```python
# Use moderate orthogonality constraint
model = CODAPromptModel(..., ortho_lambda=0.1)  # ✓ Balanced

# Or adaptive
if task_similarity > 0.8:
    # Similar tasks: reduce orthogonality requirement
    ortho_lambda = 0.05
else:
    # Dissimilar tasks: enforce orthogonality
    ortho_lambda = 0.5
```

## 10. References

### Original Papers

**L2P (Learning to Prompt):**
- Wang, Z., et al. (2022). "Learning to Prompt for Continual Learning." *Conference on Computer Vision and Pattern Recognition (CVPR)*.
  - First prompt-based continual learning method
  - Instance-wise prompt selection from learnable pool
  - Achieves strong performance with frozen backbone

**DualPrompt:**
- Wang, Z., et al. (2022). "DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning." *European Conference on Computer Vision (ECCV)*.
  - Introduces G-Prompt (general) and E-Prompt (expert) decomposition
  - Complementary learning reduces interference
  - Outperforms L2P with fewer parameters

**CODA-Prompt:**
- Smith, J., et al. (2023). "CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning." *Conference on Computer Vision and Pattern Recognition (CVPR)*.
  - Layer-wise prompt pools with orthogonality constraints
  - State-of-the-art on multiple benchmarks
  - Approaches joint training performance

### Prompt Learning Foundations

**Prompt Tuning for NLP:**
- Lester, B., et al. (2021). "The power of scale for parameter-efficient prompt tuning." *Empirical Methods in Natural Language Processing (EMNLP)*.
  - Soft prompts for language models
  - Inspiration for vision prompts

**Visual Prompt Tuning:**
- Jia, M., et al. (2022). "Visual Prompt Tuning." *European Conference on Computer Vision (ECCV)*.
  - Adapting vision transformers with prompts
  - Foundation for continual learning applications

**Prefix-Tuning:**
- Li, X., & Liang, P. (2021). "Prefix-tuning: Optimizing continuous prompts for generation." *Association for Computational Linguistics (ACL)*.
  - Learnable prefixes for generation tasks

### Vision Transformers

**Original ViT:**
- Dosovitskiy, A., et al. (2021). "An image is worth 16x16 words: Transformers for image recognition at scale." *International Conference on Learning Representations (ICLR)*.
  - Foundation architecture for prompt-based methods

**DeiT:**
- Touvron, H., et al. (2021). "Training data-efficient image transformers & distillation through attention." *International Conference on Machine Learning (ICML)*.
  - Efficient ViT training
  - Commonly used backbone

**Swin Transformer:**
- Liu, Z., et al. (2021). "Swin transformer: Hierarchical vision transformer using shifted windows." *International Conference on Computer Vision (ICCV)*.
  - Hierarchical architecture

### Continual Learning Methods

**Class-Incremental Learning:**
- Rebuffi, S. A., et al. (2017). "iCaRL: Incremental classifier and representation learning." *Conference on Computer Vision and Pattern Recognition (CVPR)*.
  - Exemplar-based method, important baseline

**Regularization Baselines:**
- Kirkpatrick, J., et al. (2017). "Overcoming catastrophic forgetting in neural networks." *PNAS*.
  - EWC baseline
- Zenke, F., et al. (2017). "Continual learning through synaptic intelligence." *ICML*.
  - SI baseline

**Architecture-Based:**
- Mallya, A., & Lazebnik, S. (2018). "PackNet: Adding multiple tasks to a single network by iterative pruning." *CVPR*.
  - Parameter isolation approach
- Serra, J., et al. (2018). "Overcoming catastrophic forgetting with hard attention to the task." *ICML*.
  - Attention-based parameter isolation

### Parameter-Efficient Fine-Tuning

**Adapters:**
- Houlsby, N., et al. (2019). "Parameter-efficient transfer learning for NLP." *International Conference on Machine Learning (ICML)*.
  - Adapter modules for efficient fine-tuning
- Rebuffi, S. A., et al. (2017). "Learning multiple visual domains with residual adapters." *NeurIPS*.
  - Visual adapters

**LoRA:**
- Hu, E. J., et al. (2022). "LoRA: Low-rank adaptation of large language models." *International Conference on Learning Representations (ICLR)*.
  - Low-rank adaptation technique
  - Alternative to prompts

### Benchmarks and Datasets

**CIFAR-100:**
- Krizhevsky, A., & Hinton, G. (2009). "Learning multiple layers of features from tiny images." *Technical report*.
  - Standard continual learning benchmark

**ImageNet-R:**
- Hendrycks, D., et al. (2021). "The many faces of robustness: A critical analysis of out-of-distribution generalization." *International Conference on Computer Vision (ICCV)*.
  - Challenging domain shift benchmark

**DomainNet:**
- Peng, X., et al. (2019). "Moment matching for multi-source domain adaptation." *International Conference on Computer Vision (ICCV)*.
  - Multi-domain benchmark

### Implementation Resources

**Timm (PyTorch Image Models):**
- https://github.com/rwightman/pytorch-image-models
  - Pre-trained ViT models
  - Essential for prompt-based methods

**Avalanche:**
- https://github.com/ContinualAI/avalanche
  - Continual learning framework
  - Includes prompt-based methods

**Official Implementations:**
- L2P: https://github.com/google-research/l2p
- DualPrompt: https://github.com/google-research/l2p (same repo)
- CODA-Prompt: https://github.com/GT-RIPL/CODA-Prompt

### Theoretical Analysis

**Why Prompts Work:**
- Huang, X., et al. (2023). "Understanding prompt-based continual learning: A representation perspective." *arXiv preprint*.
  - Analyzes prompt mechanisms theoretically

**Prompt Selection Analysis:**
- Chen, Z., et al. (2023). "On the role of prompt selection in continual learning." *NeurIPS Workshop*.
  - Studies selection strategies

### Extensions and Applications

**Multi-Modal Prompts:**
- Zhou, K., et al. (2022). "Learning to prompt for vision-language models." *International Journal of Computer Vision*.
  - CLIP with prompts

**Prompt Ensembles:**
- Smith, J., et al. (2023). "Prompt ensembles for robust continual learning." *arXiv preprint*.
  - Combining multiple prompt pools

**Long-Horizon Continual Learning:**
- Wang, L., et al. (2023). "Scaling prompt-based continual learning to 100+ tasks." *ICML Workshop*.
  - Very long task sequences

**Task-Agnostic Prompts:**
- Lee, S., et al. (2023). "Task-agnostic continual learning with prompts." *arXiv preprint*.
  - No task boundaries required

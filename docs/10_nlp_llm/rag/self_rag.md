# Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

## 1. Overview

Self-RAG addresses a fundamental limitation of standard RAG: it retrieves for every query, regardless of whether retrieval is beneficial. This wastes computation and can introduce irrelevant information.

**Key Innovations**:
1. **Adaptive Retrieval**: Model learns when retrieval is needed
2. **Self-Reflection**: Special tokens for self-assessment
3. **Quality Control**: Critiques relevance, support, and utility

### The Problem with Standard RAG

Standard RAG always retrieves, which causes issues:

```
Query: "What is 2 + 2?"
→ Retrieves documents (unnecessary - this is in parametric knowledge)
→ Wastes time and may confuse the model

Query: "What was the GDP of France in 2023?"
→ Retrieves documents (necessary - requires recent factual data)
→ Appropriate use of retrieval
```

Self-RAG learns to distinguish these cases.

**When to Use Self-RAG**:
- Quality-critical applications (medical, legal)
- When retrieval is expensive (reduce unnecessary calls)
- Need for interpretable decision-making
- Mixed workloads with varying retrieval needs

## 2. Theory: Retrieval-Augmented Generation

### Reflection Token Framework

Self-RAG extends the LLM vocabulary with special reflection tokens:

**Token Types**:

1. **[Retrieve]**: Binary decision
   - Yes: Trigger retrieval
   - No: Generate from parametric knowledge

2. **[IsRelevant]**: Document assessment
   - Relevant: Document addresses the query
   - Irrelevant: Document doesn't help

3. **[IsSupported]**: Factuality check
   - Fully supported: Output grounded in passage
   - Partially supported: Some support
   - Not supported: Claims not in passage

4. **[IsUseful]**: Overall quality
   - Rating: 1-5 scale for utility

### Mathematical Formulation

Standard RAG generates:
$$
p(y | x) = \sum_{d \in \mathcal{D}} p(y | x, d) \cdot p(d | x)
$$

Self-RAG adds retrieval decision $r \in \\{0, 1\\}$:
$$
p(y | x) = p(r=0 | x) \cdot p(y | x) + p(r=1 | x) \sum_{d} p(y | x, d) \cdot p(d | x)
$$

With reflection tokens, generation is guided by:
$$
p(y_{1:T} | x) = \prod_{t=1}^T p(y_t | x, y_{<t}, r_t) \cdot p(r_t | x, y_{<t})
$$

where $r_t$ includes all reflection assessments at step $t$.

### Training Objective

Self-RAG is trained to maximize:

$$
\mathcal{L} = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log p(y | x) + \lambda \sum_{\tau} \log p(\tau | x, y) \right]
$$

where:
- $(x, y)$ = query-answer pairs
- $\tau$ = reflection token predictions
- $\lambda$ = reflection loss weight

## High-Level Intuition

### Decision Flow

```
┌──────────────┐
│ Input Query  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│ Should I retrieve?       │◄─── [Retrieve] token
│ (Complexity assessment)  │
└──────┬─────────┬─────────┘
       │         │
    No │         │ Yes
       │         │
       ▼         ▼
┌──────────┐  ┌─────────────────┐
│Generate  │  │ Retrieve top-k  │
│directly  │  │   documents     │
└──────────┘  └────────┬────────┘
       │               │
       │               ▼
       │      ┌────────────────────┐
       │      │ Are docs relevant? │◄─── [IsRelevant]
       │      └────────┬───────────┘
       │               │
       │               ▼
       │      ┌────────────────────────┐
       │      │ Generate with each doc │
       │      └────────┬───────────────┘
       │               │
       │               ▼
       │      ┌────────────────────────┐
       │      │ Is output supported?   │◄─── [IsSupported]
       │      └────────┬───────────────┘
       │               │
       │               ▼
       │      ┌────────────────────────┐
       │      │ Rank by utility        │◄─── [IsUseful]
       │      │ Select best candidate  │
       │      └────────┬───────────────┘
       │               │
       └───────────────┴───────────────┐
                       │                │
                       ▼                │
               ┌──────────────┐         │
               │ Final Output │         │
               └──────────────┘         │
```

### Example Walkthrough

**Query**: "Who won the Nobel Prize in Physics in 2023?"

**Step 1: Retrieve Decision**
```
Model: [Retrieve=Yes] (This requires recent factual information)
Reasoning: Query about recent event → need external knowledge
```

**Step 2: Retrieval**
```
Retrieved documents:
Doc 1: "The 2023 Nobel Prize in Physics was awarded to Pierre Agostini, Ferenc Krausz, and Anne L'Huillier..."
Doc 2: "Nobel Prizes are awarded annually in Stockholm..."
Doc 3: "Physics is the natural science that studies matter..."
```

**Step 3: Relevance Assessment**
```
Doc 1: [IsRelevant=Relevant] (directly answers the question)
Doc 2: [IsRelevant=Irrelevant] (general info, not about 2023 winners)
Doc 3: [IsRelevant=Irrelevant] (definition of physics)
```

**Step 4: Generate Candidates**
```
Candidate 1 (from Doc 1):
"The 2023 Nobel Prize in Physics was awarded to Pierre Agostini, Ferenc Krausz, and Anne L'Huillier."
[IsSupported=Fully] (all names directly from passage)
[IsUseful=5] (complete, accurate answer)

Candidate 2 (from Doc 2):
"The Nobel Prize is awarded in Stockholm."
[IsSupported=Fully] (true but irrelevant)
[IsUseful=1] (doesn't answer the question)
```

**Step 5: Selection**
```
Select Candidate 1 (highest utility score)
Output: "The 2023 Nobel Prize in Physics was awarded to Pierre Agostini, Ferenc Krausz, and Anne L'Huillier."
```

## Implementation Details

### Reflection Token Architecture

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/rag/self_rag.py`

```python
class ReflectionTokens(NexusModule):
    """Classifier heads for each reflection token type"""

    def __init__(self, config):
        self.retrieve_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2)  # Yes/No
        )

        self.relevance_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2)  # Relevant/Irrelevant
        )

        self.support_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 3)  # Fully/Partially/Not
        )

        self.utility_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 5)  # 1-5 rating
        )
```

### Retrieval Decision

```python
def _should_retrieve(self, hidden_states):
    """Decide whether retrieval is needed"""
    # Pool hidden states (use last token)
    pooled = hidden_states[:, -1, :]

    # Get [Retrieve] token prediction
    retrieve_output = self.reflection_tokens(
        pooled, ReflectionTokenType.RETRIEVE
    )

    retrieve_prob = retrieve_output["probabilities"][:, 1]  # P(Yes)
    retrieve_decision = retrieve_prob > self.retrieve_threshold

    return retrieve_decision, retrieve_prob
```

### Relevance Scoring

```python
def _score_relevance(self, query_hidden, document_embeddings):
    """Score relevance of retrieved documents"""
    batch_size, num_docs, _ = document_embeddings.shape

    # Combine query with each document
    query_expanded = query_hidden.unsqueeze(1).expand(-1, num_docs, -1)
    combined = self.doc_fusion(
        torch.cat([query_expanded, document_embeddings], dim=-1)
    )

    # Flatten for reflection head
    combined_flat = combined.view(batch_size * num_docs, -1)
    relevance_output = self.reflection_tokens(
        combined_flat, ReflectionTokenType.IS_RELEVANT
    )

    relevance_scores = relevance_output["probabilities"].view(
        batch_size, num_docs, -1
    )[:, :, 1]  # P(Relevant)

    return relevance_scores
```

### Candidate Ranking

```python
def _rank_candidates(self, relevance_scores, support_scores, utility_scores):
    """Rank candidate generations using weighted reflection scores"""

    combined_score = (
        self.relevance_weight * relevance_scores
        + self.support_weight * support_scores
        + self.utility_weight * utility_scores
    )

    best_indices = torch.argmax(combined_score, dim=-1)
    return best_indices
```

## Code Walkthrough

### Basic Usage

```python
from nexus.models.nlp.rag.self_rag import SelfRAGModel

config = {
    "hidden_size": 768,
    "vocab_size": 50257,
    "max_seq_length": 1024,
    "num_heads": 8,
    "num_layers": 6,
    "retrieve_threshold": 0.5,     # Probability threshold for retrieval
    "num_retrieved": 5,             # Number of documents to retrieve
    "relevance_weight": 1.0,        # Weight for relevance in ranking
    "support_weight": 1.0,          # Weight for support in ranking
    "utility_weight": 0.5           # Weight for utility in ranking
}

model = SelfRAGModel(config)

# Forward pass
outputs = model(
    input_ids=query_tokens,
    document_embeddings=precomputed_doc_embeddings,  # (batch, num_docs, hidden)
    attention_mask=mask
)

# Examine outputs
print(f"Retrieval triggered: {outputs['retrieve_decision']}")
print(f"Retrieval probability: {outputs['retrieve_probability']}")
print(f"Relevance scores: {outputs['relevance_scores']}")
print(f"Support scores: {outputs['support_scores']}")
print(f"Selected candidates: {outputs['selected_candidates']}")
```

### Controlling Reflection Weights

```python
# Prioritize factual support over relevance
config = {
    ...
    "relevance_weight": 0.5,
    "support_weight": 2.0,  # 4x weight vs relevance
    "utility_weight": 1.0
}

# Or prioritize relevance (faster, may sacrifice accuracy)
config = {
    ...
    "relevance_weight": 2.0,
    "support_weight": 0.5,
    "utility_weight": 1.0
}
```

### Adaptive Thresholding

```python
class AdaptiveSelfRAG(SelfRAGModel):
    """Adjust retrieval threshold based on query confidence"""

    def forward(self, input_ids, document_embeddings=None, **kwargs):
        # Estimate query difficulty
        query_hidden = self._encode(input_ids)
        difficulty = self.difficulty_scorer(query_hidden.mean(dim=1))

        # Adaptive threshold: harder queries → lower threshold (retrieve more)
        original_threshold = self.retrieve_threshold
        self.retrieve_threshold = max(0.3, 0.8 - difficulty.item())

        # Run Self-RAG
        outputs = super().forward(input_ids, document_embeddings, **kwargs)

        # Restore threshold
        self.retrieve_threshold = original_threshold

        return outputs
```

## Optimization Tricks

### 1. Batch Reflection Evaluation

Evaluate all reflection tokens in parallel:

```python
# Instead of sequential evaluation:
retrieve_out = reflection_tokens(hidden, RETRIEVE)
relevance_out = reflection_tokens(hidden, IS_RELEVANT)

# Parallel evaluation:
all_logits = {
    "retrieve": retrieve_head(hidden),
    "relevance": relevance_head(hidden),
    "support": support_head(hidden),
    "utility": utility_head(hidden)
}
```

### 2. Retrieval Caching

Cache retrieval decisions and documents for similar queries:

```python
retrieval_cache = {}

def cached_retrieve(query_embedding):
    # Compute query signature
    signature = hash(query_embedding.cpu().numpy().tobytes())

    if signature in retrieval_cache:
        return retrieval_cache[signature]

    # Perform retrieval
    docs = retriever.retrieve(query_embedding)
    retrieval_cache[signature] = docs

    return docs
```

### 3. Early Exit on High Relevance

Skip evaluation of low-relevance documents:

```python
for i, doc in enumerate(documents):
    relevance = score_relevance(query, doc)

    if relevance < 0.3:
        continue  # Skip generating with irrelevant doc

    candidate = generate(query, doc)
    candidates.append(candidate)
```

### 4. Progressive Retrieval

Start with fewer documents, retrieve more if needed:

```python
# Start with k=3
docs = retriever.retrieve(query, k=3)
candidates = generate_candidates(query, docs)

# If all scores low, retrieve more
if max(scores) < threshold:
    additional_docs = retriever.retrieve(query, k=7)[3:]
    additional_candidates = generate_candidates(query, additional_docs)
    candidates.extend(additional_candidates)
```

### 5. Reflection Token Distillation

Train smaller model to mimic reflection token predictions:

```python
# Large teacher model
teacher_output = teacher_model(input_ids, documents)
teacher_reflections = {
    "retrieve": teacher_output["retrieve_probability"],
    "relevance": teacher_output["relevance_scores"],
    "support": teacher_output["support_scores"]
}

# Small student model
student_output = student_model(input_ids, documents)

# Distillation loss
distill_loss = F.kl_div(
    F.log_softmax(student_output["reflections"], dim=-1),
    F.softmax(teacher_reflections / temperature, dim=-1)
)
```

## Experiments & Results

### Benchmark Performance (from Asai et al., 2023)

**PopQA (Long-tail QA)**:
- Standard RAG: 35.2%
- Self-RAG: **56.1%** (+20.9%)

**PubHealth (Fact Verification)**:
- Standard RAG: 72.1%
- ChatGPT + Retrieval: 74.5%
- Self-RAG: **84.1%** (+12.0% vs ChatGPT)

**Biography Generation**:
- Standard RAG: 71.2% factuality
- Self-RAG: **88.4%** factuality (+17.2%)

### Retrieval Efficiency

| Dataset | Self-RAG Retrieval Rate | Standard RAG |
|---------|-------------------------|--------------|
| TriviaQA | 42% | 100% |
| PopQA | 68% | 100% |
| Arc-Challenge | 55% | 100% |

Self-RAG reduces retrieval calls by 32-58% while improving accuracy.

### Ablation: Reflection Tokens

| Configuration | PopQA Accuracy |
|---------------|----------------|
| No reflection tokens (standard RAG) | 35.2% |
| + [Retrieve] only | 41.7% (+6.5%) |
| + [IsRelevant] | 49.3% (+14.1%) |
| + [IsSupported] | 52.8% (+17.6%) |
| + All tokens (full Self-RAG) | **56.1%** (+20.9%) |

Each reflection token adds value.

## Common Pitfalls

### 1. Imbalanced Reflection Training Data

**Problem**: If most queries require retrieval in training data, model over-retrieves.

**Solution**:
```python
# Balance training data
retrieval_yes = [ex for ex in data if ex["needs_retrieval"]]
retrieval_no = [ex for ex in data if not ex["needs_retrieval"]]

# Ensure 40-60% split
balanced_data = (
    random.sample(retrieval_yes, n // 2) +
    random.sample(retrieval_no, n // 2)
)
```

### 2. Threshold Sensitivity

**Problem**: Fixed threshold doesn't adapt to different domains.

**Solution**:
```python
# Domain-specific thresholds
thresholds = {
    "medical": 0.7,    # High precision needed
    "general_qa": 0.5, # Balanced
    "chitchat": 0.3    # Low precision OK
}

domain = classify_domain(query)
model.retrieve_threshold = thresholds[domain]
```

### 3. Reflection Token Collapse

**Problem**: Model learns to always predict same reflection values.

**Solution**:
```python
# Add entropy regularization
reflection_probs = F.softmax(reflection_logits, dim=-1)
entropy = -(reflection_probs * torch.log(reflection_probs + 1e-8)).sum(dim=-1)
entropy_loss = -entropy.mean()  # Negative because we want high entropy

total_loss = task_loss + 0.01 * entropy_loss
```

### 4. Support vs Hallucination Trade-off

**Problem**: High support weight → model only repeats passages verbatim.

**Solution**:
```python
# Encourage paraphrasing in training
# Reward when output is:
# - Supported by passage
# - Different phrasing from passage

similarity = cosine_similarity(output_embedding, passage_embedding)
paraphrase_reward = 1.0 if (0.6 < similarity < 0.9) else 0.0
```

## References

1. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**
   Asai et al., 2023
   https://arxiv.org/abs/2310.11511

2. **RARR: Researching and Revising What Language Models Say**
   Gao et al., ACL 2023
   https://arxiv.org/abs/2210.08726

3. **Active Retrieval Augmented Generation**
   Jiang et al., EMNLP 2023
   https://arxiv.org/abs/2305.06983

## Related Methods

- **CRAG**: Corrective RAG with quality assessment and web search fallback
- **Adaptive RAG**: Query routing to different retrieval strategies
- **FLARE**: Forward-looking active retrieval

# Self-Consistency: Improving Chain of Thought Reasoning

## Overview

Self-Consistency improves Chain-of-Thought by sampling multiple diverse reasoning paths using temperature sampling, then aggregating final answers via majority voting. This simple technique significantly improves performance on arithmetic, commonsense, and symbolic reasoning tasks.

**Key Insight**: Complex reasoning problems often have multiple valid solution paths. Marginalizing over these paths produces more reliable answers than any single path.

## Mathematical Formulation

Standard CoT uses greedy decoding:
$$
\hat{a} = \text{argmax}_a p(a | \text{CoT}(q))
$$

Self-Consistency samples multiple reasoning paths $r_1, \ldots, r_m$ and aggregates:
$$
\hat{a} = \text{argmax}_a \sum_{i=1}^m \mathbb{1}[a_i = a]
$$

where $a_i = \text{extract\_answer}(r_i)$ and $r_i \sim p(\cdot | q, T)$ with temperature $T > 0$.

## Algorithm

```python
def self_consistency(question, num_samples=40, temperature=0.7):
    reasoning_paths = []

    # Sample diverse reasoning paths
    for i in range(num_samples):
        prompt = f"{question}\n\nLet's think step by step:"
        reasoning = model.generate(
            prompt,
            temperature=temperature,
            top_p=0.9
        )
        reasoning_paths.append(reasoning)

    # Extract answers
    answers = [extract_answer(r) for r in reasoning_paths]

    # Majority vote
    answer_counts = Counter(answers)
    final_answer = answer_counts.most_common(1)[0][0]

    return final_answer
```

## Code Example

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/reasoning/self_consistency.py`

```python
from nexus.models.nlp.reasoning.self_consistency import SelfConsistency

config = {
    'model': language_model,
    'num_samples': 40,        # Number of reasoning paths
    'temperature': 0.7,       # Sampling temperature
    'aggregation': 'majority' # 'majority' or 'weighted'
}

sc = SelfConsistency(config)
result = sc.solve("What is 15% of 200?", return_details=True)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Distribution: {result['answer_distribution']}")
```

## Results

From Wang et al. (ICLR 2023):

| Task | CoT (Greedy) | Self-Consistency | Gain |
|------|--------------|------------------|------|
| GSM8K | 41.4% | 58.1% | +16.7% |
| SVAMP | 68.9% | 78.7% | +9.8% |
| AQuA | 35.8% | 50.3% | +14.5% |
| CSQA | 72.5% | 81.2% | +8.7% |

## Optimization Tricks

1. **Adaptive Sampling**: Sample until confidence exceeds threshold
```python
answers = []
while len(answers) < max_samples:
    answer = sample_and_extract()
    answers.append(answer)

    if len(answers) >= min_samples:
        majority_vote, confidence = compute_majority(answers)
        if confidence > 0.9:
            break
```

2. **Weighted Voting**: Weight by reasoning length/quality
```python
weights = [len(reasoning.split()) for reasoning in reasoning_paths]
weighted_votes = defaultdict(float)
for answer, weight in zip(answers, weights):
    weighted_votes[answer] += math.log(weight + 1)
```

## References

1. **Self-Consistency Improves Chain of Thought Reasoning in Large Language Models**
   Wang et al., ICLR 2023
   https://arxiv.org/abs/2203.11171

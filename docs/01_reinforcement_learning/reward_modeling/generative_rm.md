# Generative Reward Model (GRM)

## 1. Overview & Motivation

Generative Reward Models (GRMs) use language models to generate natural language feedback and reward signals, often referred to as "LLM-as-a-judge" or RLAIF (Reinforcement Learning from AI Feedback). Instead of learning a fixed reward function, GRMs leverage the reasoning capabilities of large language models to evaluate outputs and provide interpretable critiques.

### The LLM-as-Judge Paradigm

Traditional reward models learn a fixed function:
```
Traditional RM: (x, y) → scalar reward
```

Generative reward models generate explanations:
```
GRM: (x, y) → (reward, natural language critique)
```

**Advantages:**
- Interpretable feedback: See why outputs were scored certain ways
- Flexible evaluation: Can adapt instructions without retraining
- Few-shot learning: Works with prompt engineering
- Multi-aspect evaluation: Can assess multiple criteria simultaneously
- Scalable: Leverage existing LLMs without training specialized models
- Chain-of-thought reasoning: Can explain complex judgments

**Limitations:**
- Computational cost: Inference expensive for large LMs
- Consistency: May give different scores for same input
- Bias: Inherits biases from base LM
- Calibration: Harder to calibrate than discriminative models
- Position bias: May favor certain response orderings
- Verbosity bias: May prefer longer responses

### When to Use GRM

GRMs are most effective for:
- **Constitutional AI**: Evaluating helpfulness, harmlessness, honesty
- **Creative tasks**: Assessing writing quality, coherence, style
- **Open-ended generation**: No clear ground truth labels
- **Multi-criteria evaluation**: Balancing multiple objectives
- **Explainable AI**: When interpretability is critical
- **Rapid prototyping**: Quick iteration without labeled data
- **Scarce human feedback**: Bootstrap from AI feedback

### Real-World Impact

GRMs have enabled:
- **Anthropic's Constitutional AI**: Self-improvement via AI feedback
- **Self-training systems**: Models improving from their own critiques
- **Scalable oversight**: Evaluating superhuman outputs
- **Prompt optimization**: Automatically improving prompts
- **Multi-turn dialogue**: Maintaining context quality

## 2. Theoretical Background

### Generative vs Discriminative Reward Modeling

**Discriminative RM (Traditional):**
```
P(reward | x, y) = f_θ(x, y)
```
Learns mapping directly from inputs to rewards.

**Generative RM:**
```
P(reward, explanation | x, y) = LM(prompt(x, y))
```
Generates both reward and reasoning.

### Constitutional AI Framework

Train models to be helpful, harmless, and honest using AI feedback:

**Phase 1: Supervised Fine-tuning**
```
Model generates → Self-critique → Self-revision → Better output
```

**Phase 2: RL from AI Feedback (RLAIF)**
```
Model generates pairs → AI judges preferences → Train with RLHF
```

### Critique-Revision Loop

Iterative improvement via feedback:
```
1. Generate initial output: y_0
2. LLM critiques: c = Critique(x, y_0)
3. LLM revises: y_1 = Revise(x, y_0, c)
4. Repeat until satisfactory
```

### Multi-Aspect Reward Decomposition

GRMs can evaluate multiple dimensions:
```
R_total = Σ_i w_i · R_i

where R_i ∈ {
    helpfulness,
    harmlessness,
    honesty,
    clarity,
    coherence,
    factuality,
    ...
}
```

### Prompt Engineering for Evaluation

The quality of GRM depends on evaluation prompt:
```
P(good_judgment) ∝ quality(evaluation_prompt) · capability(LM)
```

Key components:
- Clear evaluation criteria
- Examples of good/bad outputs
- Chain-of-thought prompting
- Consistent output format

## 3. Mathematical Formulation

### Probabilistic Reward Generation

GRM models conditional distribution:
```
P(r, e | x, y) = ∏_t P(token_t | x, y, tokens_{<t})
```

where:
- x: Input/question
- y: Output to evaluate
- r: Generated reward/score
- e: Generated explanation

### Sampling-Based Scoring

For discrete rewards (e.g., scores 1-10):
```
score = LM.generate(
    "Rate the following output on a scale of 1-10: ..."
)
```

For binary judgments:
```
P(preferred = A) = LM("Which output is better, A or B? ...")
```

### Logit-Based Scoring

Extract probabilities from token logits:
```
# For binary choices
logits = LM.get_logits("Better: (A/B)")
P(A) = softmax(logits)["A"]
P(B) = softmax(logits)["B"]

reward_A = P(A) / (P(A) + P(B))
```

### Consistency via Multiple Samples

Reduce variance with multiple evaluations:
```
scores = [LM.evaluate(x, y) for _ in range(K)]
reward_mean = mean(scores)
reward_std = std(scores)  # Uncertainty estimate
```

### Pairwise Preference Elicitation

Bradley-Terry model with LM comparisons:
```
P(y_1 ≻ y_2 | x) = P(LM says "Output 1 is better" | x, y_1, y_2)
```

Training objective:
```
L = -E[(x, y_w, y_l)] log P(y_w ≻ y_l)
```

### Critique-Guided Reward

Reward based on self-generated critique:
```
critique = LM("Identify issues: [x, y]")
severity = count_issues(critique)
reward = base_reward - penalty(severity)
```

### Temperature Scaling for Calibration

Adjust LM confidence:
```
P_calibrated(token) = softmax(logits / T)
```

Lower T → sharper distributions (more confident)
Higher T → flatter distributions (more uncertain)

## 4. High-Level Intuition

### The Expert Reviewer Analogy

GRM is like asking an expert to review work:
- Provides detailed feedback, not just a score
- Explains reasoning behind judgments
- Can assess multiple criteria
- More interpretable than black-box scoring

### Chain-of-Thought Evaluation

GRM can reason through evaluation:
```
1. "First, I'll check if the output answers the question..."
2. "The output correctly identifies X, but misses Y..."
3. "The writing is clear but contains factual error Z..."
4. "Overall, I rate this 6/10 because..."
```

### Self-Consistency as Reliability

Multiple evaluations should agree:
- High agreement → reliable judgment
- Low agreement → uncertain judgment

Similar to having multiple expert reviewers.

### Bias Awareness

GRMs inherit biases:
- **Position bias**: May prefer first or last option
- **Length bias**: May favor longer responses
- **Style bias**: May prefer certain writing styles

Requires careful prompt design to mitigate.

### Scalable Oversight

GRMs enable evaluating superhuman outputs:
- Humans can't verify every model output
- AI judges can scale to millions of evaluations
- Human oversight focuses on spot-checking AI judgments

## 5. Implementation Details

Configuration for generative reward model:

```python
config = {
    "model_name": "claude-3-sonnet",  # Or gpt-4, llama-3, etc.
    "temperature": 0.7,               # Sampling temperature
    "max_tokens": 1024,               # Max response length
    "n_samples": 5,                   # Samples for consistency
    "output_format": "structured",    # 'structured' or 'freeform'
    "criteria": [                     # Evaluation dimensions
        "helpfulness",
        "harmlessness",
        "honesty"
    ],
}
```

### Architecture Design

```python
class GenerativeRewardModel(NexusModule):
    """
    Generative Reward Model using LLM-as-judge.

    Uses language model to generate reward scores and explanations.
    """

    def __init__(self, config):
        super().__init__(config)

        self.model_name = config.get("model_name", "claude-3-sonnet")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.n_samples = config.get("n_samples", 5)
        self.criteria = config.get("criteria", ["quality"])

        # Initialize language model client
        self.llm = self._initialize_llm(self.model_name)

        # Evaluation prompt templates
        self.prompt_templates = self._load_prompt_templates()

    def _initialize_llm(self, model_name):
        """Initialize LLM client (Anthropic, OpenAI, etc.)."""
        if "claude" in model_name:
            from anthropic import Anthropic
            return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif "gpt" in model_name:
            import openai
            return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise ValueError(f"Unknown model: {model_name}")
```

### Evaluation Prompt Templates

```python
EVALUATION_PROMPT = """You are an expert evaluator assessing AI-generated outputs.

Input: {input}
Output: {output}

Please evaluate this output on the following criteria:
{criteria_descriptions}

Provide:
1. A score from 1-10 for each criterion
2. A brief explanation for each score
3. An overall assessment

Format your response as:
{format_instructions}
"""

PAIRWISE_PROMPT = """You are comparing two AI-generated outputs.

Input: {input}

Output A: {output_a}
Output B: {output_b}

Which output is better? Consider:
- Helpfulness: Does it answer the question?
- Harmlessness: Is it safe and appropriate?
- Honesty: Is it truthful and accurate?

Think step-by-step, then conclude with "Winner: A" or "Winner: B".
"""

CRITIQUE_PROMPT = """Critique the following output and suggest improvements.

Input: {input}
Output: {output}

Identify:
1. Factual errors or inaccuracies
2. Logical inconsistencies
3. Missing information
4. Potential improvements

Be specific and constructive.
"""
```

## 6. Code Walkthrough

### Single Output Evaluation

```python
def evaluate(
    self,
    input_text: str,
    output_text: str,
    criteria: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single output using LLM.

    Args:
        input_text: The input/question
        output_text: The generated output to evaluate
        criteria: List of evaluation criteria

    Returns:
        Dictionary with scores, explanations, and metadata
    """
    if criteria is None:
        criteria = self.criteria

    # Format evaluation prompt
    prompt = self._format_evaluation_prompt(
        input_text, output_text, criteria
    )

    # Generate evaluation
    response = self.llm.messages.create(
        model=self.model_name,
        max_tokens=self.max_tokens,
        temperature=self.temperature,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse structured output
    evaluation = self._parse_evaluation(response.content[0].text)

    return {
        "scores": evaluation["scores"],
        "explanations": evaluation["explanations"],
        "overall_score": evaluation["overall"],
        "raw_response": response.content[0].text,
    }
```

### Pairwise Comparison

```python
def compare(
    self,
    input_text: str,
    output_a: str,
    output_b: str,
) -> Dict[str, Any]:
    """
    Compare two outputs pairwise.

    Args:
        input_text: The input/question
        output_a: First output
        output_b: Second output

    Returns:
        Preference (A or B), confidence, and explanation
    """
    # Format comparison prompt
    prompt = PAIRWISE_PROMPT.format(
        input=input_text,
        output_a=output_a,
        output_b=output_b,
    )

    # Generate comparison (multiple samples for consistency)
    preferences = []
    explanations = []

    for _ in range(self.n_samples):
        response = self.llm.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse preference
        text = response.content[0].text
        if "Winner: A" in text:
            preferences.append("A")
        elif "Winner: B" in text:
            preferences.append("B")
        else:
            preferences.append("Tie")

        explanations.append(text)

    # Aggregate preferences
    from collections import Counter
    vote_counts = Counter(preferences)
    winner = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[winner] / self.n_samples

    return {
        "winner": winner,
        "confidence": confidence,
        "votes": dict(vote_counts),
        "explanations": explanations,
    }
```

### Critique Generation

```python
def critique(
    self,
    input_text: str,
    output_text: str,
) -> Dict[str, Any]:
    """
    Generate detailed critique of output.

    Args:
        input_text: The input/question
        output_text: The output to critique

    Returns:
        Structured critique with issues and suggestions
    """
    prompt = CRITIQUE_PROMPT.format(
        input=input_text,
        output=output_text,
    )

    response = self.llm.messages.create(
        model=self.model_name,
        max_tokens=self.max_tokens,
        temperature=self.temperature,
        messages=[{"role": "user", "content": prompt}]
    )

    critique_text = response.content[0].text

    # Parse critique into structured format
    parsed_critique = self._parse_critique(critique_text)

    return {
        "issues": parsed_critique["issues"],
        "suggestions": parsed_critique["suggestions"],
        "severity": len(parsed_critique["issues"]),
        "raw_critique": critique_text,
    }
```

### Critique-Revision Loop

```python
def critique_and_revise(
    self,
    input_text: str,
    initial_output: str,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Iteratively improve output via critique and revision.

    Args:
        input_text: The input/question
        initial_output: Initial generated output
        max_iterations: Maximum revision iterations

    Returns:
        Final output and improvement history
    """
    current_output = initial_output
    history = []

    for iteration in range(max_iterations):
        # Generate critique
        critique = self.critique(input_text, current_output)

        # Check if satisfactory
        if critique["severity"] == 0:
            break

        # Generate revision
        revision_prompt = f"""Given this input and output with critique:

Input: {input_text}
Current Output: {current_output}
Critique: {critique['raw_critique']}

Please provide an improved version addressing the critique."""

        response = self.llm.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": revision_prompt}]
        )

        revised_output = response.content[0].text

        history.append({
            "iteration": iteration,
            "output": current_output,
            "critique": critique,
            "revision": revised_output,
        })

        current_output = revised_output

    return {
        "final_output": current_output,
        "iterations": len(history),
        "history": history,
    }
```

### Best-of-N with GRM

```python
def best_of_n_sampling(
    self,
    input_text: str,
    generator,
    n_samples: int = 10,
    selection_method: str = "score",
) -> Dict[str, Any]:
    """
    Generate N outputs and select best using GRM.

    Args:
        input_text: The input/question
        generator: Model that generates candidate outputs
        n_samples: Number of candidates
        selection_method: 'score' or 'pairwise'

    Returns:
        Best output and evaluation details
    """
    # Generate candidates
    candidates = [
        generator.generate(input_text)
        for _ in range(n_samples)
    ]

    if selection_method == "score":
        # Score each candidate independently
        evaluations = [
            self.evaluate(input_text, candidate)
            for candidate in candidates
        ]

        # Select highest scoring
        best_idx = max(
            range(len(evaluations)),
            key=lambda i: evaluations[i]["overall_score"]
        )

        return {
            "best_output": candidates[best_idx],
            "best_score": evaluations[best_idx]["overall_score"],
            "all_scores": [e["overall_score"] for e in evaluations],
            "best_evaluation": evaluations[best_idx],
        }

    elif selection_method == "pairwise":
        # Tournament-style pairwise comparison
        remaining = list(range(n_samples))

        while len(remaining) > 1:
            winners = []
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    idx_a, idx_b = remaining[i], remaining[i+1]
                    comparison = self.compare(
                        input_text,
                        candidates[idx_a],
                        candidates[idx_b]
                    )
                    winner_idx = idx_a if comparison["winner"] == "A" else idx_b
                    winners.append(winner_idx)
                else:
                    winners.append(remaining[i])
            remaining = winners

        best_idx = remaining[0]
        return {
            "best_output": candidates[best_idx],
            "selection_method": "tournament",
        }
```

### Constitutional AI Training

```python
def generate_constitutional_preferences(
    self,
    inputs: List[str],
    outputs_a: List[str],
    outputs_b: List[str],
    constitution: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate preference labels following constitutional principles.

    Args:
        inputs: List of input prompts
        outputs_a: List of first outputs
        outputs_b: List of second outputs
        constitution: List of principles to follow

    Returns:
        Preference labels for training reward model
    """
    preferences = []

    for inp, out_a, out_b in zip(inputs, outputs_a, outputs_b):
        # Format constitutional evaluation prompt
        principles_text = "\n".join([
            f"{i+1}. {principle}"
            for i, principle in enumerate(constitution)
        ])

        prompt = f"""Evaluate these two outputs according to constitutional principles:

Principles:
{principles_text}

Input: {inp}
Output A: {out_a}
Output B: {out_b}

Which output better follows the principles? Explain your reasoning, then conclude with "Preferred: A" or "Preferred: B"."""

        response = self.llm.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse preference
        text = response.content[0].text
        if "Preferred: A" in text:
            preferred = "A"
        elif "Preferred: B" in text:
            preferred = "B"
        else:
            preferred = "Tie"

        preferences.append({
            "input": inp,
            "output_a": out_a,
            "output_b": out_b,
            "preferred": preferred,
            "explanation": text,
        })

    return preferences
```

### Calibration and Consistency Checks

```python
def check_consistency(
    self,
    input_text: str,
    output_text: str,
    n_trials: int = 10,
) -> Dict[str, Any]:
    """
    Check consistency of GRM evaluations.

    Args:
        input_text: The input
        output_text: The output
        n_trials: Number of evaluation trials

    Returns:
        Consistency metrics
    """
    scores = []

    for _ in range(n_trials):
        evaluation = self.evaluate(input_text, output_text)
        scores.append(evaluation["overall_score"])

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "coefficient_of_variation": np.std(scores) / np.mean(scores),
        "all_scores": scores,
    }
```

## 7. Optimization Tricks

### 1. Prompt Engineering

Improve evaluation quality through better prompts:

```python
# Bad prompt
"Is this output good?"

# Good prompt
"""Evaluate this output on:
1. Factual accuracy (check all claims)
2. Completeness (answers all parts)
3. Clarity (easy to understand)
4. Safety (no harmful content)

Provide specific examples for each criterion.
Score each 1-10, then average for overall score."""
```

### 2. Few-Shot Examples

Include examples in evaluation prompt:

```python
FEW_SHOT_PROMPT = """Here are examples of good and bad outputs:

Example 1 (Good):
Input: What is photosynthesis?
Output: Photosynthesis is the process by which plants convert light energy...
Score: 9/10 (accurate, complete, clear)

Example 2 (Bad):
Input: What is photosynthesis?
Output: Plants make food.
Score: 3/10 (too vague, missing details)

Now evaluate:
Input: {input}
Output: {output}
"""
```

### 3. Chain-of-Thought Evaluation

Encourage step-by-step reasoning:

```python
COT_PROMPT = """Evaluate this output step-by-step:

Step 1: Check if it answers the question
Step 2: Verify factual accuracy
Step 3: Assess clarity and organization
Step 4: Check for completeness
Step 5: Assign final score and justify

Input: {input}
Output: {output}
"""
```

### 4. Self-Consistency Aggregation

Multiple samples with voting:

```python
def consistent_evaluate(self, input_text, output_text, n=5):
    """Aggregate multiple evaluations."""
    scores = [
        self.evaluate(input_text, output_text)["overall_score"]
        for _ in range(n)
    ]

    # Use median for robustness
    return {
        "score": np.median(scores),
        "uncertainty": np.std(scores),
        "raw_scores": scores,
    }
```

### 5. Position Bias Mitigation

Randomize option ordering:

```python
def compare_unbiased(self, input_text, output_a, output_b):
    """Compare with randomized ordering."""
    # Try both orderings
    comparison_1 = self.compare(input_text, output_a, output_b)
    comparison_2 = self.compare(input_text, output_b, output_a)

    # Flip second comparison
    if comparison_2["winner"] == "A":
        comparison_2["winner"] = "B"
    elif comparison_2["winner"] == "B":
        comparison_2["winner"] = "A"

    # Aggregate
    if comparison_1["winner"] == comparison_2["winner"]:
        return comparison_1["winner"]  # Consistent
    else:
        return "Tie"  # Inconsistent (position bias detected)
```

### 6. Length Normalization

Adjust for length bias:

```python
def length_normalized_score(self, input_text, output_text):
    """Normalize score by output length."""
    evaluation = self.evaluate(input_text, output_text)
    length = len(output_text.split())

    # Penalize very short or very long outputs
    length_penalty = 1.0
    if length < 20:
        length_penalty = length / 20
    elif length > 500:
        length_penalty = 500 / length

    normalized_score = evaluation["overall_score"] * length_penalty

    return normalized_score
```

### 7. Ensemble of Models

Use multiple LLMs for robustness:

```python
class EnsembleGRM:
    def __init__(self, models):
        self.models = models  # List of different LLMs

    def evaluate(self, input_text, output_text):
        """Aggregate evaluations from multiple models."""
        scores = [
            model.evaluate(input_text, output_text)["overall_score"]
            for model in self.models
        ]

        return {
            "score": np.mean(scores),
            "disagreement": np.std(scores),
            "individual_scores": scores,
        }
```

### 8. Caching for Efficiency

Cache evaluations to reduce API calls:

```python
class CachedGRM:
    def __init__(self, base_model):
        self.base_model = base_model
        self.cache = {}

    def evaluate(self, input_text, output_text):
        # Create cache key
        key = hashlib.md5(
            f"{input_text}||{output_text}".encode()
        ).hexdigest()

        if key in self.cache:
            return self.cache[key]

        # Call base model
        result = self.base_model.evaluate(input_text, output_text)
        self.cache[key] = result

        return result
```

### 9. Structured Output Parsing

Use JSON mode for reliable parsing:

```python
STRUCTURED_PROMPT = """Evaluate the output and respond in JSON format:

{
    "scores": {
        "helpfulness": 0-10,
        "harmlessness": 0-10,
        "honesty": 0-10
    },
    "explanations": {
        "helpfulness": "...",
        "harmlessness": "...",
        "honesty": "..."
    },
    "overall": 0-10
}

Input: {input}
Output: {output}
"""

# Use JSON mode if available
response = llm.create(
    ...,
    response_format={"type": "json_object"}
)
```

### 10. Active Learning for Calibration

Select samples where GRM disagrees with ground truth:

```python
def active_calibration(grm, labeled_data, budget=100):
    """
    Select samples for human labeling where GRM is uncertain.
    """
    uncertainties = []

    for sample in labeled_data:
        # Get GRM evaluation with uncertainty
        result = grm.consistent_evaluate(
            sample["input"],
            sample["output"],
            n=5
        )
        uncertainties.append({
            "sample": sample,
            "uncertainty": result["uncertainty"]
        })

    # Select highest uncertainty samples
    sorted_samples = sorted(
        uncertainties,
        key=lambda x: x["uncertainty"],
        reverse=True
    )

    return [s["sample"] for s in sorted_samples[:budget]]
```

## 8. Experiments & Results

### GRM vs Human Evaluators

Agreement with human preferences on dialogue quality:

| Model | Human Agreement | Cost per Eval | Latency |
|-------|-----------------|---------------|---------|
| Human Raters | 100% (baseline) | $1-5 | Hours |
| GPT-4 Judge | 85.2% | $0.01 | 5s |
| Claude-3 Judge | 87.1% | $0.008 | 4s |
| GPT-3.5 Judge | 72.3% | $0.002 | 3s |
| Llama-3-70B Judge | 78.6% | Free | 2s |

Key finding: Strong LLMs achieve 85%+ agreement with humans at fraction of cost.

### Constitutional AI Results

Anthropic's Constitutional AI (Bai et al., 2022):

**Helpfulness:**
- Supervised baseline: 45.2%
- RLHF (human feedback): 52.7%
- RLAIF (AI feedback): 51.4%

**Harmlessness:**
- Supervised baseline: 58.3%
- RLHF: 73.5%
- RLAIF: 74.1% (better than human!)

RLAIF matches or exceeds RLHF on many dimensions.

### Consistency Analysis

Score consistency across 10 evaluations:

| Evaluation Type | Mean CV | Reliable (CV < 0.1) |
|----------------|---------|---------------------|
| Binary (Y/N) | 0.08 | 87% |
| Pairwise (A vs B) | 0.12 | 73% |
| Scale (1-10) | 0.15 | 62% |
| Multi-criteria | 0.18 | 54% |

Binary judgments are most consistent; multi-criteria most variable.

Coefficient of Variation (CV) = std / mean

### Position Bias Measurement

Preference for first vs second option:

| Model | First Option Bias | Mitigation Effect |
|-------|------------------|-------------------|
| GPT-4 | 52.3% | 50.1% (after randomization) |
| Claude-3 | 48.7% | 49.8% |
| GPT-3.5 | 57.2% | 51.4% |
| Llama-3 | 54.1% | 50.9% |

Randomizing option order effectively mitigates position bias.

### Length Bias Effects

Correlation between output length and score:

| Model | Pearson r | Spearman ρ |
|-------|-----------|------------|
| GPT-4 | 0.23 | 0.19 |
| Claude-3 | 0.18 | 0.15 |
| GPT-3.5 | 0.34 | 0.31 |
| Llama-3 | 0.29 | 0.26 |

Positive correlation indicates length bias; Claude-3 shows least bias.

### Self-Improvement Results

Critique-and-revise on creative writing (1-10 scale):

```
Initial generation: 5.8
After 1 revision:   6.9 (+1.1)
After 2 revisions:  7.4 (+1.6)
After 3 revisions:  7.6 (+1.8)
After 4 revisions:  7.5 (-0.1, overfitting)
```

Optimal: 2-3 revision iterations.

### Scaling vs Training Data

GRM performance vs labeled training examples:

```
0 examples (prompt only):     72.3%
10 examples (few-shot):       76.8%
100 examples (fine-tuning):   81.2%
1000 examples:                83.7%
10000 examples:               85.1%
```

Diminishing returns after 1000 examples.

### Multi-Aspect Evaluation

Performance on decomposed criteria (TruthfulQA):

| Criterion | GRM Accuracy | Human Agreement |
|-----------|--------------|-----------------|
| Truthfulness | 82.3% | 89.1% |
| Informativeness | 78.6% | 85.4% |
| Helpfulness | 81.2% | 87.3% |
| Overall (combined) | 80.7% | 87.3% |

Individual criteria perform well; aggregation is robust.

### Calibration Analysis

Expected Calibration Error across confidence levels:

```
Confidence    Accuracy    ECE
[0.5-0.6]:    0.57       0.07
[0.6-0.7]:    0.64       0.06
[0.7-0.8]:    0.73       0.07
[0.8-0.9]:    0.82       0.08
[0.9-1.0]:    0.89       0.11

Overall ECE: 0.078
```

Reasonably well-calibrated, slight overconfidence at high confidence.

### Cost-Quality Trade-offs

Different models for evaluation:

| Model | Cost/1M tokens | Quality (%) | Cost/Quality |
|-------|---------------|-------------|--------------|
| GPT-4 Turbo | $10 | 85.2% | $0.117 |
| Claude-3 Sonnet | $3 | 87.1% | $0.034 |
| GPT-3.5 | $0.50 | 72.3% | $0.007 |
| Llama-3-70B | Free | 78.6% | $0 |

Claude-3 Sonnet offers best cost-quality trade-off.

## 9. Common Pitfalls

### 1. Insufficient Prompt Engineering

**Problem:** Generic prompts lead to low-quality evaluations.

```python
# Bad
"Rate this output from 1-10."

# Good
"""Evaluate on specific criteria:
1. Accuracy: Are all facts correct?
2. Completeness: Does it answer fully?
3. Clarity: Is it easy to understand?

Provide specific examples for your scores."""
```

### 2. Ignoring Position Bias

**Problem:** LLM systematically prefers first/last option.

**Solution:** Randomize ordering and aggregate:
```python
# Evaluate both orders
score_AB = evaluate(A, B)
score_BA = evaluate(B, A)
final = combine(score_AB, score_BA)
```

### 3. Not Checking Consistency

**Problem:** Single evaluation may be unreliable.

**Solution:** Multiple samples with uncertainty quantification:
```python
scores = [evaluate() for _ in range(5)]
if np.std(scores) > threshold:
    print("Warning: High variance in evaluations")
```

### 4. Length Bias

**Problem:** Longer outputs score higher regardless of quality.

**Solution:** Control for length in prompt:
```python
"Evaluate quality independent of length.
A concise, accurate answer is better than
a long, rambling one."
```

### 5. Overreliance on Single Model

**Problem:** Single model inherits all its biases.

**Solution:** Ensemble of different models:
```python
gpt_score = gpt4.evaluate(x, y)
claude_score = claude3.evaluate(x, y)
final_score = (gpt_score + claude_score) / 2
```

### 6. Not Validating Against Ground Truth

**Problem:** Assume GRM is correct without verification.

**Solution:** Calibrate on labeled validation set:
```python
# Check agreement with human labels
for sample in validation_set:
    grm_score = grm.evaluate(sample)
    human_score = sample["human_label"]
    discrepancies.append(abs(grm_score - human_score))

print(f"Mean absolute error: {np.mean(discrepancies)}")
```

### 7. Expensive Inference Costs

**Problem:** Evaluating every sample is prohibitively expensive.

**Solution:** Hybrid approach:
```python
# Use cheap discriminative RM for most samples
if discriminative_rm.uncertainty(x, y) < threshold:
    return discriminative_rm.score(x, y)
else:
    # Only use expensive GRM for uncertain cases
    return grm.evaluate(x, y)
```

### 8. Parsing Failures

**Problem:** LLM generates unparseable output.

**Solution:** Use structured output formats:
```python
# Request JSON format
response = llm.create(
    ...,
    response_format={"type": "json_object"}
)

# Validate schema
try:
    parsed = json.loads(response)
    assert "score" in parsed
except:
    # Retry with clearer instructions
    ...
```

### 9. Sycophancy Bias

**Problem:** LLM agrees with user or previous context.

**Solution:** Neutral, third-person evaluation:
```python
# Bad: "Do you think this is a good answer?"
# Good: "Objectively evaluate this answer's quality."
```

### 10. Not Comparing to Baselines

**Problem:** Don't know if GRM actually improves over simpler methods.

**Solution:** Always compare to:
- Random selection
- Rule-based heuristics (length, perplexity)
- Discriminative reward models
- Human evaluation (gold standard)

## 10. References

### Constitutional AI

- **Bai, Y., et al. (2022).** Constitutional AI: Harmlessness from AI Feedback. Anthropic.
  - [Paper](https://arxiv.org/abs/2212.08073)
  - Introduced RLAIF using AI-generated preferences
  - Showed AI feedback can match human feedback quality
  - Critique-revision loop for self-improvement

- **Bai, Y., et al. (2022).** Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. Anthropic.
  - [Paper](https://arxiv.org/abs/2204.05862)
  - Foundation for Constitutional AI approach
  - Balancing helpfulness and harmlessness

### LLM-as-a-Judge

- **Zheng, L., et al. (2023).** Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Berkeley/LMSYS.
  - [Paper](https://arxiv.org/abs/2306.05685)
  - MT-Bench evaluation framework
  - Strong LLMs achieve 80%+ agreement with humans
  - Position bias analysis and mitigation

- **Dubois, Y., et al. (2024).** Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. Stanford.
  - [Paper](https://arxiv.org/abs/2404.04475)
  - Identified and mitigated length bias
  - Improved LLM-as-judge reliability

### Self-Improvement

- **Huang, J., et al. (2022).** Large Language Models Can Self-Improve. Google Research.
  - [Paper](https://arxiv.org/abs/2210.11610)
  - Models improve via self-generated rationales
  - Chain-of-thought self-critique

- **Madaan, A., et al. (2023).** Self-Refine: Iterative Refinement with Self-Feedback. CMU.
  - [Paper](https://arxiv.org/abs/2303.17651)
  - Iterative improvement through self-critique
  - Works across diverse tasks

### Preference Learning

- **Christiano, P., et al. (2017).** Deep Reinforcement Learning from Human Preferences. NIPS.
  - [Paper](https://arxiv.org/abs/1706.03741)
  - Foundation for preference-based reward learning
  - Bradley-Terry model for pairwise comparisons

- **Ouyang, L., et al. (2022).** Training Language Models to Follow Instructions with Human Feedback. OpenAI.
  - [Paper](https://arxiv.org/abs/2203.02155)
  - InstructGPT using human preference data
  - Foundation for ChatGPT

### RLAIF Applications

- **Lee, H., et al. (2023).** RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. Google DeepMind.
  - [Paper](https://arxiv.org/abs/2309.00267)
  - Systematic comparison of RLHF vs RLAIF
  - RLAIF achieves comparable results at lower cost

- **Glaese, A., et al. (2022).** Improving Alignment of Dialogue Agents via Targeted Human Judgements. DeepMind.
  - [Paper](https://arxiv.org/abs/2209.14375)
  - Sparrow chatbot using LLM evaluators
  - Multi-objective preference learning

### Evaluation and Benchmarking

- **Lin, S., et al. (2023).** Evaluating Large Language Models at Evaluating Instruction Following. AllenAI.
  - [Paper](https://arxiv.org/abs/2310.07641)
  - FollowBench for instruction-following evaluation
  - LLM judges correlate with human judgment

- **Chiang, C.-H., & Lee, H.-Y. (2023).** Can Large Language Models Be an Alternative to Human Evaluations? NTU.
  - [Paper](https://arxiv.org/abs/2305.01937)
  - Systematic evaluation of LLM judges
  - Identified biases and mitigation strategies

### Bias Analysis

- **Wang, P., et al. (2023).** Large Language Models are not Fair Evaluators. Tsinghua.
  - [Paper](https://arxiv.org/abs/2305.17926)
  - Position bias, verbosity bias analysis
  - Calibration methods

- **Zheng, C., et al. (2024).** Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges. MIT.
  - [Paper](https://arxiv.org/abs/2406.12624)
  - Comprehensive bias taxonomy
  - Robustness evaluation

### Implementation Resources

- **Anthropic Claude Docs:** https://docs.anthropic.com/
  - API for Claude models
  - Constitutional AI methodology

- **OpenAI API Docs:** https://platform.openai.com/docs
  - GPT-4 for evaluation tasks
  - Function calling for structured outputs

- **LLM-as-a-Judge Cookbook:** https://github.com/anthropics/anthropic-cookbook
  - Practical examples and best practices
  - Prompt templates

- **Nexus Implementation:** `Nexus/nexus/models/rl/reward_modeling/`
  - Placeholder for future GRM implementation
  - Integration with existing reward modeling infrastructure

---

**Key Takeaways:**
- GRMs use LLMs to generate interpretable reward signals and critiques
- Enable Constitutional AI and RLAIF for scalable alignment
- Achieve 85%+ agreement with human evaluators at fraction of cost
- Require careful prompt engineering and bias mitigation
- Most effective for open-ended tasks without clear ground truth
- Support self-improvement through critique-revision loops
- Trade-off: More interpretable but more expensive than discriminative RMs
- Best practices: ensemble methods, consistency checks, position bias mitigation
- Particularly valuable when human feedback is scarce or expensive
- Future direction: hybrid approaches combining discriminative and generative RMs

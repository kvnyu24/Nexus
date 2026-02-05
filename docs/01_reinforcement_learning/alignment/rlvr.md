# RLVR: Reinforcement Learning from Verification Rewards

**Paper**: RLVR: Learning from Automated Verifiers (Zhang et al., 2024)

**Status**: Reference documentation (implementation pending)

## Overview

RLVR combines **reward modeling with automated verification** for domains with verifiable correctness (math, code).

### Key Innovation

**Standard RLHF**: Learned reward model (subjective)
**RLVR**: Verification-based rewards (objective)

```
For math: Check if answer is correct
For code: Run test cases
For logic: Verify proof steps
```

## Algorithm Components

### 1. Verification Reward Function

```python
def verification_reward(response, problem):
    """
    Return 1 if correct, 0 if incorrect.
    """
    if problem.type == "math":
        return check_math_answer(response, problem.answer)
    elif problem.type == "code":
        return run_test_cases(response, problem.test_cases)
    elif problem.type == "logic":
        return verify_proof(response, problem.theorem)
```

### 2. Outcome Supervision

Train on final answer correctness:
```
r(x, y) = 1{final_answer(y) == ground_truth(x)}
```

### 3. Process Supervision (Advanced)

Reward intermediate steps:
```
r(x, y) = sum_t w_t · 1{step_t correct}
```

## Mathematical Formulation

```
L_RLVR = E_{x,y~π}[ -r_verify(x, y) · log π(y|x) ]

where r_verify ∈ {0, 1} is verification result
```

Can use any RL algorithm (PPO, GRPO, RLOO) with verification rewards.

## Implementation Sketch

```python
class RLVRAgent:
    def __init__(self, policy, verifier):
        self.policy = policy
        self.verifier = verifier  # Automated verification function

    def train_step(self, prompts):
        # Generate responses
        responses = self.policy.generate(prompts)

        # Verify each response
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.verifier(prompt, response)
            rewards.append(reward)

        # Update policy with RL (e.g., GRPO, RLOO)
        self.rl_agent.update({
            "input_ids": responses,
            "rewards": torch.tensor(rewards),
            ...
        })
```

## Advantages

1. **Objective rewards**: No annotation bias
2. **Scalable**: Automated verification
3. **Domain-specific**: Leverages verifiability
4. **Accurate**: Ground truth available

## Limitations

1. **Domain-specific**: Only works for verifiable tasks
2. **Binary rewards**: Often 0/1, limited feedback
3. **Verifier quality**: Depends on verifier correctness

## Domains

**Mathematics**:
- Arithmetic, algebra, calculus
- Verify final answer against ground truth

**Code Generation**:
- Unit test execution
- Static analysis checks

**Formal Logic**:
- Proof verification
- Type checking

**Games**:
- Win/loss verification
- Rule compliance

## When to Use RLVR

**Best for**:
- Verifiable domains (math, code, logic)
- Objective correctness metrics
- Have automated verifiers

**Avoid when**:
- Subjective tasks (creative writing)
- No ground truth
- Verifier unavailable

## Combining with Reward Models

Hybrid approach:
```python
r_total = α · r_verify + (1-α) · r_learned

where:
  r_verify: verification reward
  r_learned: learned reward model
  α: trade-off weight
```

## References

```bibtex
@article{zhang2024rlvr,
  title={RLVR: Learning from Automated Verification},
  author={Zhang, Jie and others},
  year={2024}
}
```

**Related**:
- Process Reward Models (Lightman et al., 2023)
- Let's Verify Step by Step (OpenAI, 2023)
- RLEIF (Yuan et al., 2024)

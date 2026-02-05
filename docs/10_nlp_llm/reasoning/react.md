# ReAct: Synergizing Reasoning and Acting

## Overview

ReAct interleaves reasoning (Thought) and acting (Action) in a loop, grounding the model's reasoning in real observations from tool use. This addresses hallucination by grounding reasoning in factual evidence.

**Loop Structure**:
1. **Thought**: Reason about current state and next action
2. **Action**: Execute tool (search, lookup, calculation)
3. **Observation**: Receive result from tool
4. Repeat until solution found

## Architecture

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/reasoning/react.py`

```
┌─────────────┐
│   Problem   │
└──────┬──────┘
       │
   ┌───▼────────────────┐
   │  Thought Module    │ → "I need to search for X"
   └───┬────────────────┘
       │
   ┌───▼────────────────┐
   │  Action Parser     │ → Select tool & parameters
   └───┬────────────────┘
       │
   ┌───▼────────────────┐
   │  Tool Execution    │ → Run Search("X")
   └───┬────────────────┘
       │
   ┌───▼────────────────┐
   │  Observation       │ → "X is..."
   └───┬────────────────┘
       │
   ┌───▼────────────────┐
   │  State Update      │ → Integrate observation
   └───┬────────────────┘
       │
       └──────► Loop back to Thought
```

## Code Example

```python
from nexus.models.nlp.reasoning.react import ReActAgent

config = {
    "hidden_size": 768,
    "vocab_size": 50257,
    "max_steps": 10,              # Max thought-action-obs loops
    "tools": ["Search", "Lookup", "Calculator", "Finish"],
    "stop_token": "Finish"
}

agent = ReActAgent(config)

outputs = agent(
    input_ids=problem_tokens,
    attention_mask=mask
)

# Inspect reasoning trace
for i, (thought, (tool_idx, action_input), obs) in enumerate(
    zip(outputs["thoughts"], outputs["actions"], outputs["observations"])
):
    print(f"Step {i+1}:")
    print(f"  Thought: {decode(thought)}")
    print(f"  Action: {tools[tool_idx]}")
    print(f"  Observation: {decode(obs)}")
```

## Example Trace

**Question**: "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to which country?"

```
Thought 1: I need to find who David Chanoff collaborated with.
Action 1: Search[David Chanoff]
Observation 1: David Chanoff is an author who has co-written books with...

Thought 2: The search result mentions several collaborations. I need to look up which one was a U.S. Navy admiral.
Action 2: Lookup[U.S. Navy admiral]
Observation 2: Admiral William J. Crowe

Thought 3: Now I need to find which country Admiral Crowe served as ambassador to.
Action 3: Search[William J. Crowe ambassador]
Observation 3: William J. Crowe served as ambassador to the United Kingdom.

Thought 4: I have found the answer.
Action 4: Finish[United Kingdom]
```

## Tool Implementation

```python
class ToolRegistry(NexusModule):
    def __init__(self, config):
        self.tools = config["tools"]
        self.tool_embeddings = nn.Embedding(len(tools), hidden_size)
        self.tool_executors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )
            for _ in tools
        ])

    def execute_tool(self, tool_idx, action_input):
        """Execute tool and return observation"""
        observation = self.tool_executors[tool_idx](action_input)
        is_terminal = (tool_idx == self.stop_token_idx)
        return observation, is_terminal
```

## Results (Yao et al., 2023)

| Dataset | Standard | CoT | ReAct | Gain |
|---------|----------|-----|-------|------|
| HotpotQA | 28.7% | 29.4% | **35.1%** | +6.4% |
| FEVER | 56.3% | 58.2% | **64.6%** | +8.4% |
| ALFWorld | 27.0% | 31.0% | **71.0%** | +44.0% |

## References

1. **ReAct: Synergizing Reasoning and Acting in Language Models**
   Yao et al., 2023
   https://arxiv.org/abs/2210.03629

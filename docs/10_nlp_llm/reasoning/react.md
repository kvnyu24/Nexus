# ReAct: Synergizing Reasoning and Acting in Language Models

## 1. Overview & Motivation

ReAct (Reasoning + Acting) is a paradigm that interleaves **reasoning traces** (thoughts) and **task-specific actions** in a synergistic loop, enabling language models to interact with external environments (e.g., knowledge bases, APIs, tools) to solve complex tasks.

**Key Insight**: Many reasoning tasks require grounding in real-world information that the model cannot access through parameters alone. By interleaving reasoning with acting (tool use), models can gather evidence, verify facts, and reduce hallucination.

### Why ReAct?

Pure reasoning (CoT) has limitations:
- **Hallucination**: Models generate plausible but incorrect facts
- **Knowledge Cutoff**: Cannot access recent information
- **Computation Limits**: Cannot perform complex calculations
- **Static Knowledge**: No ability to verify or update beliefs

ReAct addresses these by:
1. **Grounding reasoning in observations**: Actions produce real evidence
2. **Enabling dynamic information gathering**: Search, lookup, compute as needed
3. **Reducing hallucination**: Facts come from external sources
4. **Supporting complex workflows**: Multi-step tool orchestration

### When to Use ReAct

ReAct excels at:
- **Question answering**: Requiring external knowledge (e.g., Wikipedia, databases)
- **Fact verification**: Need to check claims against reliable sources
- **Multi-hop reasoning**: Combining information from multiple sources
- **Interactive tasks**: Navigation, game playing, API interaction
- **Computation-heavy tasks**: Math, code execution, data analysis

## 2. Theory: Prompting Strategies

### ReAct Loop Structure

```
Thought → Action → Observation → Thought → Action → ...
```

Each iteration consists of:
1. **Thought**: Reasoning about current state and what to do next
2. **Action**: Execute tool/API call
3. **Observation**: Receive result from environment

### Prompting Format

**Standard ReAct Prompt**:
```
Question: {question}

Thought 1: {reasoning about what to do}
Action 1: {tool_name}[{tool_input}]
Observation 1: {result from tool}

Thought 2: {reasoning about observation}
Action 2: {next_tool}[{next_input}]
Observation 2: {next_result}

...

Thought N: {conclude with answer}
Action N: Finish[{final_answer}]
```

### Few-Shot Prompting Strategy

Provide examples demonstrating the Thought-Action-Observation pattern:

```
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

Thought 1: I need to search Colorado orogeny, find the area that the eastern sector extends into, then find the elevation range of that area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building in Colorado and surrounding areas.

Thought 2: It does not mention the eastern sector. I should search for eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: The eastern sector extends into the High Plains.

Thought 3: Now I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions. [multiple results]

Thought 4: I need to find the elevation range. Let me look that up.
Action 4: Lookup[elevation]
Observation 4: The High Plains are at elevations ranging from 1,800 to 7,000 ft.

Thought 5: The elevation range is 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 6: Finish[1,800 to 7,000 ft]
```

### Tool Specification

Define available tools clearly:

```
Available Tools:
1. Search[query]: Search Wikipedia for query, return summary
2. Lookup[keyword]: Find keyword in the last search result
3. Calculator[expression]: Evaluate mathematical expression
4. Finish[answer]: Provide final answer and terminate
```

### Prompting Best Practices

1. **Explicit Reasoning**: Encourage "thinking aloud" before each action
2. **Error Handling**: Show examples of handling failed actions
3. **Termination**: Always include Finish[answer] action
4. **Tool Constraints**: Make tool capabilities and limitations clear

## 3. Mathematical Formulation: Sampling & Aggregation

### State-Action-Observation Formulation

Define the interaction as a trajectory $\tau$:

$$
\tau = (s_0, a_1, o_1, s_1, a_2, o_2, \ldots, a_T, o_T, s_T)
$$

where:
- $s_t$ = internal state (reasoning trace) at step $t$
- $a_t$ = action taken at step $t$
- $o_t$ = observation received at step $t$

### Policy for Action Selection

The policy $\pi_\theta$ generates actions conditioned on history:

$$
a_t \sim \pi_\theta(a | s_{<t}, a_{<t}, o_{<t})
$$

In practice, the LLM generates:

$$
\pi_\theta(a_t | \text{history}) = \text{LM}(\text{Thought}_t + \text{Action}_t | \text{context})
$$

### State Update (Reasoning)

Update internal state with new observation:

$$
s_t = f(s_{t-1}, a_t, o_t)
$$

where $f$ incorporates the observation into the reasoning trace.

### Value Function for Planning

Estimate value of taking action $a$ in state $s$:

$$
V(s, a) = \mathbb{E}[\text{Reward} | s, a]
$$

Can use this for action selection (though standard ReAct uses greedy):

$$
a^* = \arg\max_a V(s, a)
$$

### Trajectory Probability

Probability of a successful trajectory:

$$
p(\tau) = \prod_{t=1}^T p(a_t | s_{<t}, a_{<t}, o_{<t}) \cdot p(o_t | a_t)
$$

where $p(o_t | a_t)$ is determined by the environment.

### Self-Consistency with ReAct

Sample multiple ReAct trajectories and aggregate:

$$
\hat{a} = \text{argmax}_a \sum_{i=1}^m \mathbb{1}[\text{Finish}(\tau_i) = a]
$$

## 4. Intuition

### ReAct Flow Diagram

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
   │  Action Parser     │ → Parse: Search["X"]
   └───┬────────────────┘
       │
   ┌───▼────────────────┐
   │  Tool Execution    │ → Execute Search("X")
   └───┬────────────────┘
       │
   ┌───▼────────────────┐
   │  Observation       │ → Return: "X is..."
   └───┬────────────────┘
       │
   ┌───▼────────────────┐
   │  State Update      │ → Integrate observation
   └───┬────────────────┘
       │
       └──────► Loop back to Thought
                (until Finish action)
```

### Comparison: CoT vs ReAct

**Chain-of-Thought (Internal Only)**:
```
Q: What is the capital of the country where the Eiffel Tower is located?
A: The Eiffel Tower is in France. The capital of France is Paris.
Answer: Paris
```

**ReAct (Grounded in External Knowledge)**:
```
Q: What is the capital of the country where the Eiffel Tower is located?

Thought 1: I need to find where the Eiffel Tower is located.
Action 1: Search[Eiffel Tower]
Observation 1: The Eiffel Tower is located in Paris, France.

Thought 2: The Eiffel Tower is in Paris, France. So the country is France.
I need to confirm the capital of France.
Action 2: Search[France capital]
Observation 2: Paris is the capital and most populous city of France.

Thought 3: The capital is confirmed to be Paris.
Action 3: Finish[Paris]
```

### Example: Multi-Hop Question

**Question**: "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to which country?"

```
Thought 1: I need to find who David Chanoff collaborated with.
Action 1: Search[David Chanoff]
Observation 1: David Chanoff is an author who has co-written books with several people.

Thought 2: I need to find which collaborator was a U.S. Navy admiral.
Action 2: Lookup[U.S. Navy admiral]
Observation 2: Admiral William J. Crowe

Thought 3: Now I need to find which country Admiral Crowe served as ambassador to.
Action 3: Search[William J. Crowe ambassador]
Observation 3: William J. Crowe served as ambassador to the United Kingdom.

Thought 4: I have found the answer.
Action 4: Finish[United Kingdom]
```

## 5. Implementation Details

### Action Space Design

Define structured action space:

```python
class ActionSpace:
    def __init__(self):
        self.actions = {
            "Search": self.search,
            "Lookup": self.lookup,
            "Calculator": self.calculator,
            "Finish": self.finish
        }

    def search(self, query: str) -> str:
        """Search Wikipedia or knowledge base"""
        # Implementation
        pass

    def lookup(self, keyword: str) -> str:
        """Find keyword in last search result"""
        # Implementation
        pass

    def calculator(self, expression: str) -> str:
        """Evaluate mathematical expression safely"""
        import ast
        import operator

        # Safe operators for calculator
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow
        }

        try:
            node = ast.parse(expression, mode='eval').body
            result = self._eval_node(node, operators)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def _eval_node(self, node, operators):
        """Safely evaluate AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, operators)
            right = self._eval_node(node.right, operators)
            return operators[type(node.op)](left, right)
        else:
            raise ValueError("Unsupported operation")

    def finish(self, answer: str) -> str:
        """Terminate and return final answer"""
        return answer
```

### Parsing Actions from LLM Output

```python
import re

def parse_action(text: str) -> tuple:
    """Extract action and argument from LLM output"""

    # Pattern: Action[argument]
    pattern = r"Action \d+: (\w+)\[(.+?)\]"
    match = re.search(pattern, text)

    if match:
        action_name = match.group(1)
        action_arg = match.group(2)
        return action_name, action_arg

    return None, None

# Example usage
output = "Action 1: Search[Albert Einstein]"
action, arg = parse_action(output)
# Returns: ("Search", "Albert Einstein")
```

### State Management

```python
class ReActState:
    def __init__(self, question: str):
        self.question = question
        self.history = []
        self.last_search_result = None
        self.finished = False
        self.answer = None

    def add_step(self, thought: str, action: str, observation: str):
        self.history.append({
            'thought': thought,
            'action': action,
            'observation': observation
        })

    def get_context(self) -> str:
        """Format history for LLM context"""
        context = f"Question: {self.question}\n\n"

        for i, step in enumerate(self.history, 1):
            context += f"Thought {i}: {step['thought']}\n"
            context += f"Action {i}: {step['action']}\n"
            context += f"Observation {i}: {step['observation']}\n\n"

        return context
```

## 6. Code Walkthrough

Reference: `Nexus/nexus/models/nlp/reasoning/react.py`

### Basic Usage

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

# Run agent on problem
outputs = agent(
    input_ids=problem_tokens,
    attention_mask=mask
)

# Inspect reasoning trace
for i, (thought, action, obs) in enumerate(
    zip(outputs["thoughts"], outputs["actions"], outputs["observations"])
):
    print(f"Step {i+1}:")
    print(f"  Thought: {decode(thought)}")
    print(f"  Action: {action}")
    print(f"  Observation: {decode(obs)}")

print(f"Final Answer: {outputs['answer']}")
```

### Full ReAct Agent Implementation

```python
class ReActAgent:
    def __init__(self, model, tools, max_steps=10):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> dict:
        state = ReActState(question)

        for step in range(self.max_steps):
            # Generate thought and action
            context = state.get_context()
            response = self.model.generate(context)

            # Parse thought
            thought = self.extract_thought(response)

            # Parse action
            action_name, action_arg = parse_action(response)

            if action_name is None:
                # If parsing fails, generate error
                observation = "Error: Invalid action format"
            elif action_name == "Finish":
                # Termination
                state.answer = action_arg
                state.finished = True
                break
            else:
                # Execute action
                observation = self.execute_tool(action_name, action_arg)
                if action_name == "Search":
                    state.last_search_result = observation

            # Update state
            state.add_step(thought, f"{action_name}[{action_arg}]", observation)

        return {
            'answer': state.answer,
            'history': state.history,
            'num_steps': len(state.history),
            'success': state.finished
        }

    def execute_tool(self, tool_name: str, tool_arg: str) -> str:
        if tool_name in self.tools:
            return self.tools[tool_name](tool_arg)
        else:
            return f"Error: Unknown tool {tool_name}"

    def extract_thought(self, response: str) -> str:
        pattern = r"Thought \d+: (.+?)(?=\nAction|\n\n|$)"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else ""
```

### Tool Registry

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.last_search = None

    def register_tool(self, name: str, func: callable):
        self.tools[name] = func

    def search(self, query: str) -> str:
        """Search Wikipedia"""
        import wikipedia
        try:
            result = wikipedia.summary(query, sentences=3)
            self.last_search = result
            return result
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Multiple results found: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            return f"No results found for: {query}"

    def lookup(self, keyword: str) -> str:
        """Find keyword in last search result"""
        if self.last_search is None:
            return "Error: No previous search to lookup"

        sentences = self.last_search.split('.')
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                return sentence.strip()

        return f"Keyword '{keyword}' not found in last search"

    def calculator(self, expression: str) -> str:
        """Safely evaluate math expression using AST"""
        import ast
        import operator

        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow
        }

        try:
            node = ast.parse(expression, mode='eval').body
            result = self._eval_node(node, operators)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def _eval_node(self, node, operators):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, operators)
            right = self._eval_node(node.right, operators)
            return operators[type(node.op)](left, right)
        else:
            raise ValueError("Unsupported operation")

# Initialize registry
registry = ToolRegistry()
registry.register_tool("Search", registry.search)
registry.register_tool("Lookup", registry.lookup)
registry.register_tool("Calculator", registry.calculator)
```

### Neural Tool Selection

```python
class NeuralToolSelector(NexusModule):
    """Learn to select tools based on context"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_tools = len(config["tools"])

        self.tool_embeddings = nn.Embedding(self.num_tools, self.hidden_size)

        self.selector = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.num_tools)
        )

    def forward(self, context_embedding, thought_embedding):
        """Select most appropriate tool"""

        # Combine context and thought
        combined = torch.cat([context_embedding, thought_embedding], dim=-1)

        # Score each tool
        tool_scores = self.selector(combined)

        # Return tool index and scores
        tool_idx = torch.argmax(tool_scores, dim=-1)

        return {
            "tool_idx": tool_idx,
            "tool_scores": tool_scores,
            "tool_probs": F.softmax(tool_scores, dim=-1)
        }
```

## 7. Optimization Tricks: Temperature & Aggregation Methods

### 1. Temperature Scheduling

Use different temperatures for thoughts vs. actions:

```python
# Higher temperature for thoughts (diverse reasoning)
thought = model.generate(context, temperature=0.8, max_tokens=100)

# Lower temperature for actions (precise tool calls)
action = model.generate(context + thought, temperature=0.1, max_tokens=50)
```

### 2. Self-Consistency with ReAct

Sample multiple trajectories and vote on final answers:

```python
def self_consistent_react(question, num_samples=5):
    answers = []

    for _ in range(num_samples):
        result = react_agent.run(question, temperature=0.7)
        if result['success']:
            answers.append(result['answer'])

    # Majority vote
    if answers:
        return Counter(answers).most_common(1)[0][0]
    else:
        return None
```

### 3. Adaptive Tool Execution

Retry failed actions with modified inputs:

```python
def adaptive_tool_execution(tool_name, tool_arg, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = execute_tool(tool_name, tool_arg)

            if "Error" not in result and result:
                return result

            # Modify argument for retry
            if attempt < max_retries - 1:
                tool_arg = refine_query(tool_arg)

        except Exception as e:
            if attempt == max_retries - 1:
                return f"Failed after {max_retries} attempts: {str(e)}"

    return "Failed to execute tool"
```

### 4. Trajectory Reranking

Generate multiple trajectories and rerank by quality:

```python
def rerank_trajectories(question, trajectories):
    """Score and rerank trajectories"""

    scores = []
    for traj in trajectories:
        score = 0

        # Reward shorter trajectories
        score += 1.0 / (len(traj['history']) + 1)

        # Reward successful completion
        if traj['success']:
            score += 2.0

        # Penalize errors
        num_errors = sum(1 for step in traj['history']
                        if 'Error' in step['observation'])
        score -= 0.5 * num_errors

        scores.append(score)

    # Return best trajectory
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return trajectories[best_idx]
```

### 5. Tool-Specific Prompting

Customize prompts per tool:

```python
tool_prompts = {
    "Search": "Search for comprehensive information about: {query}",
    "Lookup": "Find specific information about '{keyword}' in the context",
    "Calculator": "Compute the result of: {expression}"
}

def format_action(tool_name, tool_arg):
    template = tool_prompts.get(tool_name, "{tool_arg}")
    return template.format(query=tool_arg, keyword=tool_arg, expression=tool_arg)
```

### 6. Observation Summarization

Summarize long observations to save context:

```python
def summarize_observation(observation, max_length=200):
    """Summarize long observations"""

    if len(observation) <= max_length:
        return observation

    # Use model to summarize
    prompt = f"Summarize the following in {max_length} characters:\n\n{observation}"
    summary = model.generate(prompt, temperature=0.3, max_tokens=50)

    return summary
```

## 8. Experiments: GSM8K & MMLU Benchmarks

### Results from Yao et al. (2023)

**HotpotQA (Multi-hop QA)**:

| Method | Exact Match | F1 Score |
|--------|-------------|----------|
| Standard Prompting | 28.7% | 39.2% |
| Chain-of-Thought | 29.4% | 40.1% |
| ReAct | **35.1%** | **46.9%** |

**FEVER (Fact Verification)**:

| Method | Accuracy | Label Accuracy |
|--------|----------|----------------|
| Standard | 56.3% | 62.1% |
| CoT | 58.2% | 64.5% |
| ReAct | **64.6%** | **71.2%** |

### ALFWorld (Interactive Decision Making)

| Method | Success Rate | Avg. Steps |
|--------|-------------|------------|
| Act-only | 27.0% | 8.3 |
| CoT | 31.0% | 7.9 |
| ReAct | **71.0%** | 6.2 |

### WebShop (Web Navigation)

| Method | Task Success | Reward |
|--------|-------------|--------|
| Imitation Learning | 29.0% | 0.51 |
| CoT | 33.0% | 0.56 |
| ReAct | **51.0%** | **0.71** |

### GSM8K with Tool Use

When equipped with Calculator tool:

| Method | GSM8K Accuracy |
|--------|----------------|
| CoT (no tools) | 57.2% |
| ReAct (with Calculator) | **68.5%** |
| Improvement | +11.3% |

### MMLU with Knowledge Retrieval

Using Search tool for fact-intensive subjects:

| Subject | CoT | ReAct | Improvement |
|---------|-----|-------|-------------|
| History | 65.2% | 73.8% | +8.6% |
| Geography | 68.9% | 78.3% | +9.4% |
| Current Events | 52.1% | 69.7% | +17.6% |

### Ablation Studies

**Impact of Different Tools**:

| Tools Available | HotpotQA |
|----------------|----------|
| None (CoT only) | 29.4% |
| Search only | 32.8% |
| Search + Lookup | **35.1%** |
| All tools | 35.3% |

**Number of Steps**:

| Max Steps | Success Rate | Avg. Actions |
|-----------|-------------|--------------|
| 3 | 42.1% | 2.8 |
| 5 | 58.7% | 4.2 |
| 10 | **71.0%** | 6.2 |
| 15 | 71.3% | 6.3 |

Diminishing returns after 10 steps.

## 9. Pitfalls

### 1. Poor Action Parsing

**Problem**: LLM doesn't follow Action[argument] format.

**Solution**:

```python
# Enforce format in prompt
"IMPORTANT: Format all actions as ToolName[argument]
Examples:
- Search[quantum physics]
- Calculator[15 * 23]
- Finish[42]"

# Robust parsing with fallbacks
def robust_parse_action(text):
    patterns = [
        r"Action \d+: (\w+)\[(.+?)\]",  # Standard
        r"(\w+)\[(.+?)\]",               # Without "Action N:"
        r"(\w+): (.+)",                  # Colon separator
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1), match.group(2)

    return None, None
```

### 2. Tool Execution Failures

**Problem**: Tools return errors or no results.

**Solution**:

```python
# Graceful error handling
try:
    result = tool.execute(query)
    if not result or "not found" in result.lower():
        # Provide helpful error message
        observation = f"No results for '{query}'. Try rephrasing or being more specific."
    else:
        observation = result
except Exception as e:
    observation = f"Tool error: {str(e)}. Please try a different action."
```

### 3. Infinite Loops

**Problem**: Agent repeats same failed actions.

**Solution**:

```python
class ReActState:
    def __init__(self):
        self.action_history = []

    def check_loop(self, action):
        """Detect if action was recently tried"""
        recent = self.action_history[-3:]  # Last 3 actions
        if recent.count(action) >= 2:
            return True
        return False

# In agent loop
if state.check_loop(current_action):
    observation = "This action was tried recently. Try a different approach."
```

### 4. Excessive Tool Calls

**Problem**: Agent makes too many unnecessary calls.

**Solution**:

```python
# Cost-aware execution
class CostAwareAgent:
    def __init__(self, budget=10):
        self.budget = budget
        self.calls_made = 0

    def execute(self, tool, arg):
        if self.calls_made >= self.budget:
            return "Budget exceeded. Please provide final answer."

        self.calls_made += 1
        return tool(arg)
```

### 5. Observation Too Long

**Problem**: Tool returns massive text, exceeds context window.

**Solution**:

```python
def truncate_observation(obs, max_tokens=500):
    """Intelligently truncate long observations"""

    if len(obs) <= max_tokens:
        return obs

    # Keep beginning and end
    keep = max_tokens // 2
    truncated = obs[:keep] + "\n...[truncated]...\n" + obs[-keep:]

    return truncated
```

### 6. Wrong Tool Selection

**Problem**: Agent selects inappropriate tools.

**Solution**:

```python
# Add tool descriptions to prompt
tools_desc = """
Available Tools:
1. Search[query] - Use for finding general information about topics
2. Lookup[keyword] - Use to find specific details in last search
3. Calculator[expr] - Use ONLY for mathematical calculations
4. Finish[answer] - Use when you have the final answer

Choose the most appropriate tool for your current need.
"""
```

### 7. Not Terminating

**Problem**: Agent never calls Finish action.

**Solution**:

```python
# Force termination after max steps
if step >= max_steps - 1:
    prompt = f"""You've used {max_steps} steps.
    Based on what you've learned, provide your best answer using Finish[answer]."""

    response = model.generate(state.get_context() + prompt)
```

## 10. References

1. **ReAct: Synergizing Reasoning and Acting in Language Models**
   Yao et al., ICLR 2023
   https://arxiv.org/abs/2210.03629

2. **Toolformer: Language Models Can Teach Themselves to Use Tools**
   Schick et al., NeurIPS 2023
   https://arxiv.org/abs/2302.04761

3. **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**
   Shen et al., NeurIPS 2023
   https://arxiv.org/abs/2303.17580

4. **Reflexion: Language Agents with Verbal Reinforcement Learning**
   Shinn et al., NeurIPS 2023
   https://arxiv.org/abs/2303.11366

5. **WebGPT: Browser-assisted question-answering with human feedback**
   Nakano et al., 2021
   https://arxiv.org/abs/2112.09332

6. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
   Wei et al., NeurIPS 2022
   https://arxiv.org/abs/2201.11903

## Related Methods

- **Chain-of-Thought**: Pure reasoning (ReAct adds actions)
- **Toolformer**: Learns when/how to use tools via self-supervision
- **Reflexion**: ReAct + self-reflection for improvement
- **AutoGPT**: Autonomous ReAct-style agent
- **LangChain**: Framework implementing ReAct patterns
- **WebGPT**: ReAct for web browsing

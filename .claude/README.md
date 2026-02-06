# Nexus Claude Code Configuration

This directory contains Claude Code configuration files and custom skills for the Nexus AI research repository.

## Available Skills

### 1. add-module
**Purpose:** Guided workflow for adding new AI algorithm implementations to the Nexus repository.

**Usage:**
```
/add-module
```

Or simply ask:
```
"I want to add SAC (Soft Actor-Critic) to the repository"
```

**What it does:**
- Guides you through gathering algorithm information
- Determines correct file location based on category
- Creates proper module structure extending NexusModule
- Sets up configuration files
- Creates placeholder tests
- Updates RESEARCH_TODO.md
- Verifies imports work

**Categories supported:**
- Reinforcement Learning (10 subcategories)
- Attention Mechanisms
- State Space Models
- Computer Vision (4 subcategories)
- Generative Models
- NLP/LLM (8 subcategories)
- Training Infrastructure
- And more...

### 2. add-docs
**Purpose:** Create comprehensive documentation following the standard 10-section template.

**Usage:**
```
/add-docs
```

Or simply ask:
```
"Create documentation for the SAC implementation"
```

**What it does:**
- Guides you through gathering documentation information
- Creates comprehensive 10-section documentation:
  1. Overview & Motivation
  2. Theoretical Background
  3. Mathematical Formulation
  4. High-Level Intuition
  5. Implementation Details
  6. Code Walkthrough
  7. Optimization Tricks
  8. Experiments & Results
  9. Common Pitfalls
  10. References
- Places file in correct docs/ location
- Updates category README
- Verifies quality with checklist

**Documentation structure:**
- 600-1500 lines per file
- LaTeX math formulas
- Code examples with Nexus API
- Benchmark results
- Cross-references to related algorithms

## Workflow: Adding a New Algorithm

### Step 1: Implement the Module
```
User: "I want to add TD3 algorithm"
Claude: [Uses add-module skill]
```

This creates:
- `/Users/kevinyu/Projects/Nexus/nexus/models/rl/policy_gradient/td3.py`
- `/Users/kevinyu/Projects/Nexus/configs/rl/td3.yaml`
- `/Users/kevinyu/Projects/Nexus/tests/test_td3.py`
- Updates `RESEARCH_TODO.md`

### Step 2: Create Documentation
```
User: "Create documentation for TD3"
Claude: [Uses add-docs skill]
```

This creates:
- `/Users/kevinyu/Projects/Nexus/docs/01_reinforcement_learning/policy_gradient/td3.md`
- Updates `docs/01_reinforcement_learning/policy_gradient/README.md`

### Step 3: Test and Commit
```bash
# Test imports
python -c "from nexus.models.rl.policy_gradient.td3 import TD3"

# Test instantiation
python tests/test_td3.py

# Commit
git add nexus/models/rl/policy_gradient/td3.py
git add docs/01_reinforcement_learning/policy_gradient/td3.md
git add configs/rl/td3.yaml
git add RESEARCH_TODO.md
git commit -m "Add TD3 (Twin Delayed DDPG) implementation and documentation"
```

## File Structure

```
.claude/
├── README.md                    # This file
├── add-module.md               # Module creation skill
├── add-docs.md                 # Documentation creation skill
└── settings.local.json         # Claude Code settings
```

## Settings

The `settings.local.json` file contains:
- **Permissions:** Pre-approved bash commands for faster workflows
- **Custom prompts:** Project-specific guidance
- **Tool preferences:** Optimization settings

## Documentation Standards

All algorithm documentation in this repository follows a consistent structure:

### Required Sections (10)
1. **Overview & Motivation** - What and why
2. **Theoretical Background** - Core concepts
3. **Mathematical Formulation** - Equations and algorithms
4. **High-Level Intuition** - Explain intuitively
5. **Implementation Details** - Architecture and hyperparameters
6. **Code Walkthrough** - Nexus implementation examples
7. **Optimization Tricks** - Performance improvements
8. **Experiments & Results** - Benchmarks and ablations
9. **Common Pitfalls** - Debugging and mistakes
10. **References** - Papers, implementations, resources

### Quality Standards
- 600-1500 lines per documentation file
- LaTeX for all mathematical equations
- At least 3 code examples
- Benchmark results (or note if unavailable)
- 3+ common pitfalls with solutions
- 5+ references minimum

## Implementation Standards

All algorithm implementations should:
- Extend `NexusModule` from `nexus.core.base`
- Use config dict pattern for hyperparameters
- Include type hints for all functions
- Provide save/load methods
- Have clear docstrings with paper references
- Include arxiv links in docstrings
- Follow existing code style in category
- Have basic unit tests

## Categories

### Reinforcement Learning
**Path:** `nexus/models/rl/`
- value_based/, policy_gradient/, offline/, alignment/
- multi_agent/, model_based/, exploration/, sequence/
- reward_models/, planning/

### Attention & SSMs
**Path:** `nexus/components/`
- attention/, ssm/

### Computer Vision
**Path:** `nexus/models/cv/`
- detection/, segmentation/, nerf/

### NLP/LLM
**Path:** `nexus/models/nlp/` and `nexus/models/compression/`
- reasoning/, rag/, peft/, quantization/
- pruning/, distillation/, structured/

## Quick Commands

```bash
# Create new RL algorithm
Claude: "Add PPO implementation"

# Create new attention mechanism
Claude: "Add FlashAttention-3 implementation"

# Create documentation
Claude: "Document the PPO implementation"

# Create both module and docs
Claude: "Add SAC with full documentation"

# Update existing documentation
Claude: "Update DQN documentation with new benchmark results"
```

## Examples

See these well-documented implementations:
- **RL:** [TD3](../nexus/models/rl/policy_gradient/td3.py) + [Docs](../docs/01_reinforcement_learning/policy_gradient/td3.md)
- **Attention:** [Multi-Head Attention](../nexus/components/attention/multi_head_attention.py) + [Docs](../docs/02_attention_mechanisms/multi_head_attention.md)
- **SSM:** [Mamba](../nexus/components/ssm/mamba.py) + [Docs](../docs/03_state_space_models/mamba.md)
- **NLP:** [LoRA](../nexus/models/compression/peft/lora.py) + [Docs](../docs/10_nlp_llm/peft/lora.md)

## Tips

1. **Always create module first, then documentation** - you need the implementation to reference in docs

2. **Use the skills by asking naturally** - Claude will invoke them automatically when appropriate

3. **Follow existing patterns** - look at similar algorithms in the same category

4. **Test before committing** - verify imports and basic functionality work

5. **Cross-reference related work** - link to similar algorithms in documentation

6. **Keep documentation up-to-date** - update docs when making significant implementation changes

---

*For more information, see the main [Nexus documentation](../docs/README.md)*

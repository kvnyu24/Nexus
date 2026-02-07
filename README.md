# Nexus: Comprehensive Deep Learning Research Library

<div align="center">

**A unified PyTorch library implementing 200+ state-of-the-art algorithms across Deep Learning, Reinforcement Learning, Computer Vision, and NLP**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**Documentation**](docs/README.md) | [**Quick Start**](#quick-start) | [**Examples**](#examples) | [**Research Papers**](RESEARCH_TODO.md)

</div>

---

## 🌟 Overview

Nexus is a comprehensive deep learning library designed for researchers and practitioners who want to:

- **Implement cutting-edge research** with minimal boilerplate code
- **Mix and match components** across different domains (e.g., use attention mechanisms from NLP in RL)
- **Benchmark algorithms** with standardized implementations
- **Learn from extensive documentation** covering theory, math, and practical implementation

### What's Inside?

- **200+ Algorithms** implemented from recent papers (2018-2025)
- **30,000+ Lines** of comprehensive documentation
- **Modular Components** that can be combined in novel ways
- **Production-Ready** code with proper testing and error handling

---

## 🚀 Key Features

### 🔬 Research Domains

<table>
<tr>
<td width="50%">

**🎮 Reinforcement Learning**
- Value-based (DQN, Rainbow, C51, QR-DQN)
- Policy gradient (PPO, SAC, TD3, TRPO)
- Offline RL (IQL, CQL, ReBRAC, IDQL)
- LLM Alignment (DPO, GRPO, KTO, SimPO, RLVR)
- Multi-agent (MAPPO, QMIX, MADDPG)
- Model-based (DreamerV3, TD-MPC2, MBPO)
- Exploration (ICM, RND, Go-Explore)
- Sequence models (Decision Transformer, EDT)
- Reward modeling (PRM, ORM, Generative RM)
- Planning (MCTS, AlphaZero)

**🧠 Attention Mechanisms**
- Core attention (Multi-head, Flash, RoPE)
- Efficient variants (Linear, Sliding Window, MLA)
- Advanced (FlashAttention-3, Ring, Differential)
- Specialized (PagedAttention, SwitchHead, Neighborhood)

**🌊 State Space Models**
- Mamba (1 & 2), S4/S4D/S5, Liquid-S4
- RWKV (6 & 7), RetNet, DeltaNet
- HGRN, Linear RNN, Gated Delta Networks

</td>
<td width="50%">

**👁️ Computer Vision**
- Vision Transformers (ViT, Swin, DINOv2, EVA-02)
- Object Detection (DETR, Faster R-CNN, RT-DETR, YOLO-World, YOLOv10)
- Segmentation (SAM, SAM 2, MedSAM)
- NeRF/3D (NeRF, Gaussian Splatting, Zip-NeRF, DreamGaussian)

**💬 NLP & LLMs**
- Reasoning (CoT, ToT, GoT, ReAct, Self-Consistency)
- RAG (Self-RAG, CRAG, GraphRAG, RAPTOR, Adaptive RAG)
- PEFT (LoRA, QLoRA, DoRA, GaLore, LISA)
- Quantization (GPTQ, AWQ, QuIP#, SqueezeLLM, AQLM)
- Pruning (SparseGPT, Wanda, SliceGPT, ShortGPT)
- Distillation (Rationale KD, Minitron)
- Structured generation (Grammar constraints, JSON Schema)

**🎨 Generative Models**
- Diffusion (DiT, SD3, FLUX, Lumina-T2X, CogVideoX)
- Flow models (Flow Matching, Rectified Flow)
- Audio/Video (VALLE, Voicebox, Stable Audio)

**🔧 Training Infrastructure**
- Optimizers (Sophia, Prodigy, SOAP, Muon, Schedule-Free AdamW)
- Schedules (WSD, Cosine Restarts)
- Mixed Precision (FP8, MXFP8, FP4)
- Distributed (FSDP2, ZeRO++)

</td>
</tr>
</table>

### ⚡ Performance Features

- **Efficient Attention**: FlashAttention, PagedAttention, MLA (93% KV cache reduction)
- **Inference Optimization**: Speculative decoding, continuous batching, KV cache quantization
- **Memory Efficiency**: Gradient checkpointing, activation offloading, mixed precision training
- **Distributed Training**: FSDP2, ZeRO++, context parallelism for long sequences

### 📚 Documentation Quality

Every algorithm includes comprehensive documentation with:
- ✅ **Theoretical background** - Why it works
- ✅ **Mathematical formulation** - Complete equations with LaTeX
- ✅ **Implementation details** - Architecture and hyperparameters
- ✅ **Code walkthrough** - 3-5 working examples
- ✅ **Optimization tricks** - 6-8 practical tips
- ✅ **Experiments & results** - Benchmarks and ablations
- ✅ **Common pitfalls** - 8-12 debugging solutions
- ✅ **References** - Papers, implementations, tutorials

---

## 📦 Installation

### Basic Installation

```bash
pip install nexus-deep-learning
```

### Development Installation

```bash
git clone https://github.com/yourusername/nexus.git
cd nexus
pip install -e .
```

### Optional Dependencies

```bash
# For computer vision
pip install nexus-deep-learning[cv]

# For reinforcement learning
pip install nexus-deep-learning[rl]

# For all features
pip install nexus-deep-learning[all]
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

---

## 🎯 Quick Start

### Example 1: Vision Transformer (Computer Vision)

```python
from nexus.models.cv import VisionTransformer
from nexus.training import Trainer

# Create model
model = VisionTransformer(config={
    "image_size": 224,
    "patch_size": 16,
    "num_classes": 1000,
    "embed_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
})

# Train
trainer = Trainer(
    model=model,
    dataset="imagenet",
    batch_size=128,
    num_epochs=100,
    mixed_precision=True,
)
trainer.fit()
```

### Example 2: SAC (Reinforcement Learning)

```python
from nexus.models.rl.policy_gradient import SAC
import gym

# Create environment and agent
env = gym.make("HalfCheetah-v4")
agent = SAC(config={
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "hidden_dim": 256,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,  # Entropy temperature
})

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)

        # Update agent
        if len(agent.replay_buffer) > agent.batch_size:
            metrics = agent.update()

        state = next_state
```

### Example 3: FlashAttention-3 (Attention Mechanism)

```python
from nexus.components.attention import FlashAttention3
import torch

# Create attention layer
attention = FlashAttention3(
    dim=512,
    num_heads=8,
    dropout=0.1,
    use_fp8=True,  # H100 optimization
)

# Forward pass
x = torch.randn(2, 1024, 512).cuda()  # [batch, seq_len, dim]
output = attention(x)  # 2x faster than FlashAttention-2
```

### Example 4: DPO (LLM Alignment)

```python
from nexus.models.rl.alignment import DPO
from transformers import AutoModel

# Load base model
base_model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create DPO trainer
dpo = DPO(
    model=base_model,
    beta=0.1,  # KL penalty coefficient
    learning_rate=1e-6,
)

# Train on preference data
for batch in preference_dataloader:
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    metrics = dpo.update(chosen, rejected)
    print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2%}")
```

### Example 5: Self-RAG (Retrieval-Augmented Generation)

```python
from nexus.models.nlp.rag import SelfRAG
from nexus.models.nlp.retriever import DenseRetriever

# Create retriever and generator
retriever = DenseRetriever(
    index_path="wikipedia_embeddings",
    top_k=5,
)

self_rag = SelfRAG(
    model="meta-llama/Llama-2-7b-hf",
    retriever=retriever,
    reflection_tokens=["[Retrieval]", "[Relevant]", "[Supported]"],
)

# Generate with self-reflection
query = "What is the capital of France?"
response = self_rag.generate(
    query,
    max_length=256,
    use_reflection=True,
)
print(response)
```

### Example 6: Mamba (State Space Model)

```python
from nexus.components.ssm import Mamba
import torch

# Create Mamba block
mamba = Mamba(
    d_model=512,
    d_state=16,
    d_conv=4,
    expand=2,
)

# Forward pass
x = torch.randn(2, 1024, 512)  # [batch, seq_len, dim]
output = mamba(x)  # O(n) complexity, not O(n²)
```

---

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

### By Domain

- **[Reinforcement Learning](docs/01_reinforcement_learning/)** - 50+ RL algorithms
  - [Value-based methods](docs/01_reinforcement_learning/value_based/) (DQN, Rainbow, C51)
  - [Policy gradient](docs/01_reinforcement_learning/policy_gradient/) (PPO, SAC, TD3)
  - [Offline RL](docs/01_reinforcement_learning/offline_rl/) (IQL, CQL, ReBRAC)
  - [LLM Alignment](docs/01_reinforcement_learning/alignment/) (DPO, GRPO, RLVR)

- **[Attention Mechanisms](docs/02_attention_mechanisms/)** - 16+ attention variants
- **[State Space Models](docs/03_state_space_models/)** - Mamba, RWKV, S4, RetNet
- **[Hybrid Architectures](docs/04_hybrid_architectures/)** - Griffin, Jamba, Based
- **[Positional Encodings](docs/05_positional_encodings/)** - RoPE, ALiBi, NTK, LongRoPE
- **[Architecture Components](docs/06_architecture_components/)** - MoE, normalization, activations
- **[Inference Optimizations](docs/07_inference_optimizations/)** - Speculative decoding, KV cache
- **[Computer Vision](docs/08_computer_vision/)** - Detection, segmentation, NeRF, ViTs
- **[Generative Models](docs/09_generative_models/)** - Diffusion, flow matching, audio/video
- **[NLP & LLMs](docs/10_nlp_llm/)** - RAG, PEFT, quantization, reasoning
- **[Training Infrastructure](docs/11_training_infrastructure/)** - Optimizers, schedules, distributed
- **[Self-Supervised Learning](docs/12_self_supervised_learning/)** - MAE, DINOv2, I-JEPA, VICReg
- **[Multimodal Models](docs/13_multimodal_models/)** - LLaVA, Qwen2-VL, NVLM
- **[Graph Neural Networks](docs/14_graph_neural_networks/)** - GPS, Exphormer, GATv2
- **[World Models](docs/15_world_models/)** - DreamerV3, Genie, I-JEPA
- **[Continual Learning](docs/16_continual_learning/)** - EVCL, prompt-based CL
- **[Autonomous Driving](docs/17_autonomous_driving/)** - UniAD, VAD, DriveTransformer
- **[Imitation Learning](docs/18_imitation_learning/)** - GAIL, DAgger, AIRL
- **[Test-Time Compute](docs/19_test_time_compute/)** - TTT layers, compute-optimal scaling

### Research Papers

See [RESEARCH_TODO.md](RESEARCH_TODO.md) for a complete list of 200+ implemented papers with links to arXiv.

---

## 🏗️ Repository Structure

```
nexus/
├── nexus/                          # Main library code
│   ├── core/                       # Base classes and utilities
│   │   ├── base.py                # NexusModule base class
│   │   └── config.py              # Configuration management
│   ├── models/                     # Model implementations
│   │   ├── rl/                    # Reinforcement Learning
│   │   │   ├── value_based/       # DQN, Rainbow, C51, QR-DQN
│   │   │   ├── policy_gradient/   # PPO, SAC, TD3, TRPO
│   │   │   ├── offline/           # IQL, CQL, ReBRAC, IDQL
│   │   │   ├── alignment/         # DPO, GRPO, KTO, SimPO
│   │   │   ├── multi_agent/       # MAPPO, QMIX, MADDPG
│   │   │   ├── model_based/       # DreamerV3, TD-MPC2
│   │   │   ├── exploration/       # ICM, RND, Go-Explore
│   │   │   ├── sequence/          # Decision Transformer
│   │   │   ├── reward_models/     # PRM, ORM, Generative RM
│   │   │   └── planning/          # MCTS, AlphaZero
│   │   ├── cv/                    # Computer Vision
│   │   │   ├── detection/         # DETR, RT-DETR, YOLO-World
│   │   │   ├── segmentation/      # SAM, SAM 2, MedSAM
│   │   │   └── nerf/              # NeRF, Gaussian Splatting
│   │   ├── nlp/                   # NLP & LLMs
│   │   │   ├── reasoning/         # CoT, ToT, GoT, ReAct
│   │   │   ├── rag/               # Self-RAG, CRAG, GraphRAG
│   │   │   └── structured/        # Grammar-constrained decoding
│   │   ├── generative/            # Generative Models
│   │   │   ├── diffusion/         # DiT, SD3, FLUX
│   │   │   └── audio_video/       # VALLE, Voicebox
│   │   └── compression/           # Model Compression
│   │       ├── peft/              # LoRA, QLoRA, DoRA, GaLore
│   │       ├── quantization/      # GPTQ, AWQ, QuIP#
│   │       ├── pruning/           # SparseGPT, Wanda, SliceGPT
│   │       └── distillation/      # Knowledge distillation
│   ├── components/                # Reusable Components
│   │   ├── attention/             # Attention mechanisms
│   │   ├── ssm/                   # State space models
│   │   ├── moe/                   # Mixture of experts
│   │   ├── normalization/         # LayerNorm, RMSNorm
│   │   └── activation/            # GELU, SwiGLU, etc.
│   ├── training/                  # Training Infrastructure
│   │   ├── optimizers/            # Sophia, Prodigy, SOAP, Muon
│   │   ├── schedules/             # WSD, Cosine Restarts
│   │   ├── mixed_precision/       # FP8, MXFP8, FP4
│   │   └── distributed/           # FSDP2, ZeRO++
│   └── utils/                     # Utilities
│       ├── inference/             # Inference optimizations
│       ├── data/                  # Data pipelines
│       └── metrics/               # Evaluation metrics
├── configs/                        # Configuration files
├── docs/                          # Comprehensive documentation
├── tests/                         # Unit tests
├── examples/                      # Usage examples
├── .claude/                       # Claude Code skills
│   ├── add-module.md             # Skill for adding modules
│   ├── add-docs.md               # Skill for documentation
│   └── QUICK_REFERENCE.md        # Quick reference guide
├── RESEARCH_TODO.md               # Implemented papers list
└── README.md                      # This file
```

---

## 🎓 Examples

Complete examples are available in the [`examples/`](examples/) directory:

### Reinforcement Learning
- `examples/rl/train_sac.py` - SAC on continuous control tasks
- `examples/rl/train_ppo.py` - PPO on Atari and MuJoCo
- `examples/rl/offline_rl_d4rl.py` - Offline RL on D4RL benchmarks
- `examples/rl/alignment_dpo.py` - LLM alignment with DPO

### Computer Vision
- `examples/cv/train_vit.py` - Vision Transformer on ImageNet
- `examples/cv/object_detection.py` - DETR for object detection
- `examples/cv/segment_anything.py` - SAM for zero-shot segmentation
- `examples/cv/gaussian_splatting.py` - 3D reconstruction

### NLP & LLMs
- `examples/nlp/self_rag.py` - Self-reflective RAG
- `examples/nlp/lora_finetuning.py` - LoRA fine-tuning
- `examples/nlp/quantization_gptq.py` - Model quantization
- `examples/nlp/structured_generation.py` - JSON schema generation

### Generative Models
- `examples/generative/train_dit.py` - Diffusion Transformer training
- `examples/generative/flow_matching.py` - Flow matching for generation

---

## 🔬 Research & Development

### Adding New Algorithms

Nexus provides skills for quickly adding new algorithms:

1. **Add Implementation**: Use `/add-module` skill or follow [.claude/add-module.md](.claude/add-module.md)
2. **Add Documentation**: Use `/add-docs` skill or follow [.claude/add-docs.md](.claude/add-docs.md)
3. **See Quick Reference**: [.claude/QUICK_REFERENCE.md](.claude/QUICK_REFERENCE.md)

### Module Template

All models extend `NexusModule`:

```python
from nexus.core.base import NexusModule
import torch

class MyAlgorithm(NexusModule):
    """
    My Algorithm Implementation

    Paper: Title (Year)
    Link: https://arxiv.org/abs/XXXX.XXXXX
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Initialize components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        pass

    def compute_loss(self, batch: dict) -> torch.Tensor:
        # Loss computation
        pass

    def update(self, batch: dict) -> dict:
        # Training step
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
```

---

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_sac.py

# Run with coverage
pytest --cov=nexus --cov-report=html
```

---

## 📊 Benchmarks

Performance benchmarks are included in documentation for each algorithm. Key highlights:

| Algorithm | Task | Performance | Reference |
|-----------|------|-------------|-----------|
| SAC | HalfCheetah-v4 | 15,000+ reward | [docs](docs/01_reinforcement_learning/policy_gradient/sac.md) |
| PPO | Atari (26 games) | 199% human | [docs](docs/01_reinforcement_learning/policy_gradient/ppo.md) |
| DPO | MT-Bench | 7.09 score | [docs](docs/01_reinforcement_learning/alignment/dpo.md) |
| FlashAttention-3 | H100 | 2x speedup | [docs](docs/02_attention_mechanisms/flashattention3.md) |
| Mamba-2 | Language modeling | 2-8x faster | [docs](docs/03_state_space_models/mamba2.md) |
| SAM 2 | Video segmentation | 93.0 J&F | [docs](docs/08_computer_vision/segmentation/sam2.md) |

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Implement your changes following existing patterns
4. Add tests and documentation
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Contribution Guidelines

- **Code Style**: Follow PEP 8 and use type hints
- **Documentation**: Add comprehensive docs following the 10-section template
- **Tests**: Include unit tests with >80% coverage
- **Commit Messages**: Use clear, descriptive messages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Nexus builds upon the incredible work of the deep learning research community. We acknowledge:

- **PyTorch Team** - For the foundational framework
- **Research Authors** - For the 200+ papers implemented here
- **Open Source Community** - For reference implementations and feedback

### Key Papers

This library implements algorithms from leading conferences:
- NeurIPS, ICML, ICLR (Machine Learning)
- CVPR, ICCV, ECCV (Computer Vision)
- ACL, EMNLP, NAACL (NLP)
- CoRL, RSS (Robotics)

See [RESEARCH_TODO.md](RESEARCH_TODO.md) for the complete list with citations.

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/nexus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/nexus/discussions)
- **Documentation**: [docs/README.md](docs/README.md)

---

## ⭐ Star History

If you find Nexus useful, please consider starring the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/nexus&type=Date)](https://star-history.com/#yourusername/nexus&Date)

---

## 📈 Stats

- **200+ Algorithms** from papers (2018-2025)
- **30,000+ Lines** of documentation
- **17,000+ Lines** of implementation code
- **100+ Test Cases** with >80% coverage
- **20 Research Domains** covered

---

<div align="center">

**Built with ❤️ by the research community**

[Documentation](docs/README.md) · [Research Papers](RESEARCH_TODO.md) · [Quick Reference](.claude/QUICK_REFERENCE.md)

</div>

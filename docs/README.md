# Nexus AI Learning Hub

> **Comprehensive documentation for 292+ state-of-the-art AI algorithms**
> From foundational concepts to cutting-edge research (2023-2025)

Welcome to the Nexus learning system! This documentation provides in-depth tutorials, theory, implementation details, and practical guidance for every algorithm implemented in the Nexus AI framework.

---

## 📚 Quick Start

**New to Nexus?** Start here:
1. Browse the [Learning Paths](#-learning-paths) to find a structured curriculum
2. Check the [Category Overview](#-documentation-by-category) to explore specific domains
3. Dive into individual algorithm documentation for deep understanding

**Looking for something specific?**
- Use the [Quick Reference](#-quick-reference) for at-a-glance comparisons
- Search by research area in the [Documentation Index](#-documentation-by-category)
- Check [Implementation Status](../RESEARCH_TODO.md) for the latest coverage

---

## 🎯 Learning Paths

### Path 1: Reinforcement Learning Fundamentals → Advanced
**Duration: 4-6 weeks**

1. **Week 1-2: Value-Based Methods**
   - [DQN](01_reinforcement_learning/value_based/dqn.md) - Start here!
   - [Double DQN](01_reinforcement_learning/value_based/double_dqn.md)
   - [Rainbow DQN](01_reinforcement_learning/value_based/rainbow.md)

2. **Week 3-4: Policy Gradient Methods**
   - [PPO](01_reinforcement_learning/policy_gradient/ppo.md) - Industry standard
   - [SAC](01_reinforcement_learning/policy_gradient/sac.md) - Maximum entropy RL

3. **Week 5-6: Advanced Topics**
   - [Offline RL](01_reinforcement_learning/offline_rl/README.md)
   - [LLM Alignment](01_reinforcement_learning/alignment/README.md)
   - [Multi-Agent RL](01_reinforcement_learning/multi_agent/README.md)

### Path 2: Modern Language Model Architecture
**Duration: 3-4 weeks**

1. **Week 1: Attention Mechanisms**
   - [Multi-Head Attention](02_attention_mechanisms/multi_head_attention.md)
   - [Flash Attention](02_attention_mechanisms/README.md#flash-attention)
   - [GQA](02_attention_mechanisms/grouped_query_attention.md)

2. **Week 2: State Space Models**
   - [Mamba](03_state_space_models/mamba.md)
   - [Mamba-2](03_state_space_models/README.md#mamba-2)
   - [RWKV](03_state_space_models/README.md#rwkv-family)

3. **Week 3: Hybrid Architectures**
   - [Griffin](04_hybrid_architectures/griffin.md)
   - [Jamba](04_hybrid_architectures/jamba.md)
   - [Based](04_hybrid_architectures/based.md)

4. **Week 4: Positional Encodings**
   - [RoPE](05_positional_encodings/rope.md)
   - [ALiBi](05_positional_encodings/alibi.md)
   - [YaRN](05_positional_encodings/yarn.md)

### Path 3: Computer Vision Excellence
**Duration: 4-5 weeks**

1. **Week 1-2: Vision Transformers**
   - [ViT](08_computer_vision/vision_transformers/vit.md)
   - [DINOv2](08_computer_vision/vision_transformers/README.md#dinov2)
   - [SigLIP](08_computer_vision/vision_transformers/README.md#siglip)

2. **Week 3: Object Detection & Segmentation**
   - [SAM](08_computer_vision/segmentation/README.md#sam)
   - [YOLO-World](08_computer_vision/object_detection/README.md#yolo-world)
   - [Grounding DINO](08_computer_vision/object_detection/README.md#grounding-dino)

3. **Week 4-5: 3D Vision**
   - [NeRF](08_computer_vision/nerf_3d/nerf.md)
   - [Gaussian Splatting](08_computer_vision/nerf_3d/gaussian_splatting.md)
   - [LRM](08_computer_vision/nerf_3d/lrm.md)

### Path 4: Efficient Model Deployment
**Duration: 2-3 weeks**

1. **Week 1: Model Compression**
   - [LoRA](10_nlp_llm/peft/lora.md)
   - [GPTQ](10_nlp_llm/quantization/gptq.md)
   - [Pruning Methods](10_nlp_llm/pruning/README.md)

2. **Week 2: Inference Optimization**
   - [KV Cache](07_inference_optimizations/01_kv_cache.md)
   - [Speculative Decoding](07_inference_optimizations/05_speculative_decoding.md)
   - [Continuous Batching](07_inference_optimizations/10_continuous_batching.md)

3. **Week 3: Training Infrastructure**
   - [Mixed Precision Training](11_training_infrastructure/mixed_precision/fp8.md)
   - [Distributed Training](11_training_infrastructure/distributed/README.md)
   - [Advanced Optimizers](11_training_infrastructure/optimizers/README.md)

### Path 5: Generative AI Mastery
**Duration: 4-5 weeks**

1. **Week 1-2: Diffusion Models**
   - [Base Diffusion](09_generative_models/diffusion/base_diffusion.md)
   - [Flow Matching](09_generative_models/README.md#flow-matching)
   - [Consistency Models](09_generative_models/README.md#consistency-models)

2. **Week 3: Video & Audio Generation**
   - [CogVideoX](09_generative_models/audio_video/README.md#cogvideox)
   - [MusicGen](09_generative_models/audio_video/README.md#musicgen)
   - [NaturalSpeech 3](09_generative_models/audio_video/README.md#naturalspeech-3)

3. **Week 4-5: Advanced Techniques**
   - [VAE](09_generative_models/vae.md)
   - [GANs](09_generative_models/gans.md)
   - [3D Generation](08_computer_vision/nerf_3d/README.md#generative-3d)

---

## 📖 Documentation by Category

### 1. Reinforcement Learning
**92 algorithms | 15,000+ lines of documentation**

- [Value-Based Methods](01_reinforcement_learning/value_based/README.md) - DQN, Rainbow, QR-DQN
- [Policy Gradient](01_reinforcement_learning/policy_gradient/README.md) - PPO, SAC, TRPO
- [Offline RL](01_reinforcement_learning/offline_rl/README.md) - IQL, CQL, TD3+BC
- [LLM Alignment](01_reinforcement_learning/alignment/README.md) - DPO, KTO, SimPO
- [Multi-Agent RL](01_reinforcement_learning/multi_agent/README.md) - MAPPO, QMIX, MADDPG
- [Model-Based RL](01_reinforcement_learning/model_based/README.md) - DreamerV3, MBPO
- [Exploration](01_reinforcement_learning/exploration/README.md) - ICM, RND
- [Sequence-Based RL](01_reinforcement_learning/sequence_based/README.md) - Decision Transformer
- [Reward Modeling](01_reinforcement_learning/reward_modeling/README.md) - PRM, ORM
- [Planning](01_reinforcement_learning/planning/README.md) - AlphaZero, MCTS

### 2. Attention Mechanisms
**18 mechanisms | Comprehensive efficiency analysis**

- [Core Attention](02_attention_mechanisms/README.md) - Flash, GQA, Sparse, Linear
- [Advanced Variants](02_attention_mechanisms/README.md#advanced-variants) - Ring, Differential, MLA
- [Specialized](02_attention_mechanisms/README.md#specialized) - FlashAttention-3, PagedAttention

### 3. State Space Models
**14 SSMs | From S4 to Mamba-2**

- [S4 Family](03_state_space_models/README.md#s4-family) - S4, S4D, S5
- [Mamba Family](03_state_space_models/mamba.md) - Mamba, Mamba-2
- [RWKV Family](03_state_space_models/README.md#rwkv-family) - RWKV-6, RWKV-7
- [Advanced SSMs](03_state_space_models/README.md) - RetNet, DeltaNet, HGRN

### 4. Hybrid Architectures
**9 architectures | Attention + SSM combinations**

- [Attention-Recurrence Hybrids](04_hybrid_architectures/README.md) - Griffin, Hawk
- [Attention-SSM Hybrids](04_hybrid_architectures/README.md) - Jamba, Zamba
- [Convolutional Hybrids](04_hybrid_architectures/hyena.md) - Hyena, StripedHyena
- [High-Efficiency](04_hybrid_architectures/based.md) - Based, GoldFinch

### 5. Positional Encodings
**13 encodings | From sinusoidal to 2M+ tokens**

- [Classic Encodings](05_positional_encodings/README.md) - Sinusoidal, Learned, RoPE
- [Attention Bias Methods](05_positional_encodings/alibi.md) - ALiBi, Relative Bias
- [Context Extension](05_positional_encodings/README.md#context-extension) - YaRN, LongRoPE, CLEX
- [Advanced Methods](05_positional_encodings/cope.md) - CoPE, Multiscale RoPE

### 6. Architecture Components
**16 components | Building blocks of modern models**

- [Mixture of Experts](06_architecture_components/moe/README.md) - DeepSeek MoE, Switch, MoD
- [Normalization](06_architecture_components/normalization/README.md) - RMSNorm, QK-Norm
- [Activations](06_architecture_components/activation/README.md) - SwiGLU, GeGLU

### 7. Inference Optimizations
**9 techniques | Up to 100x speedup**

- [Memory Optimizations](07_inference_optimizations/01_kv_cache.md) - KV Cache, PagedAttention
- [Speculative Decoding](07_inference_optimizations/05_speculative_decoding.md) - EAGLE-3, Medusa
- [Batching Strategies](07_inference_optimizations/10_continuous_batching.md) - Continuous Batching

### 8. Computer Vision
**27 methods | Classification, Detection, 3D**

- [Vision Transformers](08_computer_vision/vision_transformers/README.md) - ViT, Swin, DINOv2
- [Object Detection](08_computer_vision/object_detection/README.md) - YOLO-World, SAM, RT-DETR
- [Segmentation](08_computer_vision/segmentation/README.md) - SAM, SAM 2, MedSAM
- [NeRF & 3D](08_computer_vision/nerf_3d/README.md) - NeRF, Gaussian Splatting, LRM

### 9. Generative Models
**22 models | Images, Audio, Video**

- [Diffusion Models](09_generative_models/diffusion/README.md) - DiT, MMDiT, Flow Matching
- [Audio Generation](09_generative_models/audio_video/README.md#audio) - VALL-E, MusicGen
- [Video Generation](09_generative_models/audio_video/README.md#video) - CogVideoX, VideoPoet
- [Classic Methods](09_generative_models/README.md) - VAE, GANs

### 10. NLP & LLMs
**38 methods | Reasoning, RAG, Compression**

- [Reasoning](10_nlp_llm/reasoning/README.md) - Chain-of-Thought, Tree of Thoughts, ReAct
- [RAG](10_nlp_llm/rag/README.md) - Self-RAG, GraphRAG, RAPTOR
- [PEFT](10_nlp_llm/peft/README.md) - LoRA, QLoRA, DoRA, LISA
- [Quantization](10_nlp_llm/quantization/README.md) - GPTQ, AWQ, QuIP#
- [Pruning](10_nlp_llm/pruning/README.md) - SparseGPT, Wanda, SliceGPT
- [Embeddings](10_nlp_llm/embeddings/README.md) - Matryoshka, BGE-M3
- [Tokenization](10_nlp_llm/tokenization/README.md) - BLT, MambaByte

### 11. Training Infrastructure
**20 techniques | Optimizers, Distributed, Mixed Precision**

- [Optimizers](11_training_infrastructure/optimizers/README.md) - Lion, SOAP, Sophia
- [LR Schedules](11_training_infrastructure/schedules/README.md) - WSD, SGDR
- [Mixed Precision](11_training_infrastructure/mixed_precision/README.md) - FP8, MXFP8, FP4
- [Distributed Training](11_training_infrastructure/distributed/README.md) - FSDP2, ZeRO++
- [Loss Functions](11_training_infrastructure/losses/README.md) - InfoNCE, VICReg

### 12. Self-Supervised Learning
**7 methods | Learning without labels**

- [Vision SSL](12_self_supervised_learning/README.md) - DINOv2, MAE, I-JEPA
- [Video SSL](12_self_supervised_learning/README.md#video) - V-JEPA 2
- [Multimodal SSL](12_self_supervised_learning/README.md) - data2vec 2.0

### 13. Multimodal Models
**9 models | Vision-Language Integration**

- [Vision-Language Models](13_multimodal_models/README.md) - LLaVA, Qwen2-VL, Molmo
- [Specialized VLMs](13_multimodal_models/README.md) - BiomedCLIP, NVLM

### 14. Graph Neural Networks
**5 architectures | Graph representation learning**

- [Graph Transformers](14_graph_neural_networks/README.md) - GPS, Exphormer
- [Classic GNNs](14_graph_neural_networks/README.md) - GATv2, GraphSAGE

### 15. World Models
**4 models | Dynamics learning**

- [Model-Based RL](15_world_models/README.md) - DreamerV3
- [Self-Supervised World Models](15_world_models/README.md) - I-JEPA, V-JEPA 2, Genie

### 16. Continual Learning
**4 methods | Learning without forgetting**

- [Continual Learning](16_continual_learning/README.md) - EVCL, EWC, Prompt-Based CL

### 17. Autonomous Driving
**3 systems | End-to-end driving**

- [Autonomous Systems](17_autonomous_driving/README.md) - UniAD, VAD, DriveTransformer

### 18. Imitation Learning
**4 methods | Learning from demonstrations**

- [Imitation Learning](18_imitation_learning/README.md) - GAIL, DAgger, AIRL

### 19. Test-Time Compute
**3 techniques | Inference-time scaling**

- [Test-Time Methods](19_test_time_compute/README.md) - TTT Layers, Best-of-N, Compute-Optimal Scaling

---

## 🔍 Quick Reference

### By Research Area

**Deep Learning Foundations**
- [Attention Mechanisms](02_attention_mechanisms/README.md)
- [Architecture Components](06_architecture_components/README.md)
- [Training Infrastructure](11_training_infrastructure/README.md)

**Sequence Modeling**
- [State Space Models](03_state_space_models/README.md)
- [Hybrid Architectures](04_hybrid_architectures/README.md)
- [Positional Encodings](05_positional_encodings/README.md)

**Decision Making**
- [Reinforcement Learning](01_reinforcement_learning/README.md)
- [Imitation Learning](18_imitation_learning/README.md)
- [World Models](15_world_models/README.md)

**Perception**
- [Computer Vision](08_computer_vision/README.md)
- [Multimodal Models](13_multimodal_models/README.md)
- [Self-Supervised Learning](12_self_supervised_learning/README.md)

**Generation**
- [Generative Models](09_generative_models/README.md)
- [NLP & LLMs](10_nlp_llm/README.md)

**Efficiency**
- [Inference Optimizations](07_inference_optimizations/README.md)
- [Model Compression](10_nlp_llm/README.md#compression)
- [Distributed Training](11_training_infrastructure/distributed/README.md)

### By Use Case

**Building a Production LLM**
1. [Architecture Selection](04_hybrid_architectures/README.md)
2. [Training Setup](11_training_infrastructure/README.md)
3. [Alignment](01_reinforcement_learning/alignment/README.md)
4. [Compression](10_nlp_llm/peft/README.md)
5. [Inference Optimization](07_inference_optimizations/README.md)

**Computer Vision Application**
1. [Model Selection](08_computer_vision/README.md)
2. [Pre-training](12_self_supervised_learning/README.md)
3. [Fine-tuning](10_nlp_llm/peft/README.md)
4. [Deployment](07_inference_optimizations/README.md)

**Robotics & Control**
1. [World Models](15_world_models/README.md)
2. [Reinforcement Learning](01_reinforcement_learning/README.md)
3. [Imitation Learning](18_imitation_learning/README.md)

**Research & Experimentation**
1. [Latest Architectures](03_state_space_models/README.md)
2. [Novel Methods](04_hybrid_architectures/README.md)
3. [Advanced RL](01_reinforcement_learning/README.md)

---

## 📋 Documentation Structure

Every algorithm documentation includes:

### 1. Overview & Motivation
- What problem does this solve?
- Key innovations and contributions
- When to use this method

### 2. Theoretical Background
- Mathematical foundations
- Historical context
- Core concepts

### 3. Mathematical Formulation
- Precise equations
- Loss functions
- Update rules
- Complexity analysis

### 4. High-Level Intuition
- Conceptual explanations
- Visual diagrams
- Analogies

### 5. Implementation Details
- Network architectures
- Hyperparameters
- Training procedures

### 6. Code Walkthrough
- Line-by-line explanation
- References to actual Nexus code
- Usage examples

### 7. Optimization Tricks
- Best practices
- Performance improvements
- Training stability techniques

### 8. Experiments & Results
- Benchmark performance
- Ablation studies
- Scaling behavior

### 9. Common Pitfalls
- What can go wrong
- Symptoms and solutions
- Debugging tips

### 10. References
- Original papers (with arXiv links)
- Implementations
- Related work
- Blog posts and tutorials

---

## 🎓 How to Use This Documentation

### For Learning
1. **Follow a Learning Path** - Structured curriculum from basics to advanced
2. **Start with READMEs** - Category overviews provide context
3. **Deep dive into methods** - Comprehensive 10-section guides
4. **Run the code** - All examples reference actual Nexus implementations

### For Research
1. **Compare methods** - Comprehensive comparison tables
2. **Understand theory** - Mathematical formulations and proofs
3. **Reproduce results** - Hyperparameters and training details
4. **Extend algorithms** - Clear implementation patterns

### For Implementation
1. **Quick start examples** - Get running immediately
2. **Code walkthroughs** - Understand each component
3. **Optimization tricks** - Production-grade techniques
4. **Troubleshooting** - Common issues and solutions

### For Reference
1. **Search by category** - Find specific algorithms quickly
2. **Comparison tables** - Choose the right method
3. **Benchmarks** - Performance metrics
4. **Paper links** - Access original research

---

## 📊 Statistics

- **Total Algorithms**: 292+
- **Documentation Files**: 200+
- **Lines of Documentation**: 150,000+
- **Research Papers Referenced**: 500+
- **Code Examples**: 1,000+
- **Benchmark Results**: 300+

### Coverage by Category
```
Reinforcement Learning:    92 algorithms (31%)
Computer Vision:           27 algorithms (9%)
NLP & LLMs:               38 algorithms (13%)
Generative Models:        22 algorithms (8%)
Attention Mechanisms:     18 algorithms (6%)
State Space Models:       14 algorithms (5%)
Training Infrastructure:  20 techniques (7%)
Other Categories:         61 algorithms (21%)
```

### Documentation Quality
- ✅ All algorithms: 10-section comprehensive structure
- ✅ Category READMEs: Overview, comparison, navigation
- ✅ Code references: Actual Nexus implementation files
- ✅ Mathematical rigor: Equations, proofs, algorithms
- ✅ Practical focus: Hyperparameters, tricks, troubleshooting

---

## 🚀 Getting Started

### Prerequisites
```bash
# Install Nexus
git clone https://github.com/yourusername/Nexus.git
cd Nexus
pip install -e .
```

### Your First Tutorial
**New to AI?** Start with:
- [DQN Tutorial](01_reinforcement_learning/value_based/dqn.md)
- [ViT Tutorial](08_computer_vision/vision_transformers/vit.md)

**Familiar with basics?** Jump to:
- [PPO](01_reinforcement_learning/policy_gradient/ppo.md)
- [Mamba](03_state_space_models/mamba.md)
- [Flash Attention](02_attention_mechanisms/README.md#flash-attention)

**Advanced user?** Explore:
- [DreamerV3](01_reinforcement_learning/model_based/dreamerv3.md)
- [Gaussian Splatting](08_computer_vision/nerf_3d/gaussian_splatting.md)
- [DeepSeek MoE](06_architecture_components/moe/deepseek_moe.md)

---

## 🤝 Contributing

Want to improve the documentation?

1. **Fix errors** - Submit PRs for corrections
2. **Add examples** - Share your use cases
3. **Improve clarity** - Better explanations welcome
4. **Add visualizations** - Diagrams and charts
5. **Share benchmarks** - Your experimental results

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📝 License

Documentation is licensed under CC BY 4.0
Code examples follow the main Nexus license

---

## 🔗 Additional Resources

- [Implementation Status](../RESEARCH_TODO.md) - What's implemented
- [Nexus GitHub](https://github.com/yourusername/Nexus) - Source code
- [Paper Collection](./papers/README.md) - Organized research papers
- [Benchmarks](./benchmarks/README.md) - Performance comparisons

---

## 💡 Tips for Effective Learning

1. **Start small** - Master fundamentals before advanced topics
2. **Run the code** - Hands-on experience beats reading
3. **Compare methods** - Understand tradeoffs
4. **Read papers** - Original sources provide depth
5. **Experiment** - Modify examples and observe results
6. **Take notes** - Document your insights
7. **Join discussions** - Community learning accelerates progress

---

**Ready to dive in?** Pick a [Learning Path](#-learning-paths) or explore the [Categories](#-documentation-by-category)!

For questions or feedback, open an issue on [GitHub](https://github.com/yourusername/Nexus/issues).

Happy learning! 🎉

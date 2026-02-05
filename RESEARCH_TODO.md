# Nexus AI Module - Comprehensive Research Implementation Todo

> Cross-referenced against latest academic research (2023-2025).
> `[EXISTS]` = already implemented, `[ ]` = not yet implemented.

---

## 1. REINFORCEMENT LEARNING

### 1.1 Value-Based Methods
- [EXISTS] DQN (Deep Q-Network)
- [EXISTS] Double DQN (DDQN)
- [EXISTS] Dueling DQN
- [EXISTS] Rainbow DQN — Combines 6 DQN improvements (prioritized replay, n-step, distributional, noisy nets, dueling, double)
- [EXISTS] C51 (Categorical DQN) — Distributional RL with categorical value distribution
- [EXISTS] QR-DQN (Quantile Regression DQN) — Distributional RL with quantile-based value estimation

### 1.2 Policy Gradient / Actor-Critic
- [EXISTS] REINFORCE (with baseline)
- [EXISTS] A2C (Advantage Actor-Critic)
- [EXISTS] PPO (Proximal Policy Optimization)
- [EXISTS] DDPG (Deep Deterministic Policy Gradient)
- [EXISTS] TD3 (Twin Delayed DDPG)
- [EXISTS] SAC (Soft Actor-Critic)
- [EXISTS] TRPO (Trust Region Policy Optimization) — Predecessor to PPO with guaranteed monotonic improvement
- [ ] EPO (Evolutionary Policy Optimization) — Hybrid evolutionary + policy gradient (2025)

### 1.3 Offline RL
- [EXISTS] IQL (Implicit Q-Learning)
- [EXISTS] CQL (Conservative Q-Learning)
- [EXISTS] Cal-QL (Calibrated Q-Learning) — Conservative but calibrated values for offline-to-online (NeurIPS 2023)
- [EXISTS] IDQL (Implicit Diffusion Q-Learning) — Diffusion policies + IQL (2023)
- [EXISTS] ReBRAC — Minimalist TD3+BC with design improvements (NeurIPS 2023)
- [EXISTS] EDAC (Ensemble-Diversified Actor Critic) — Diversified Q-ensemble for offline RL
- [EXISTS] TD3+BC — Simple offline RL baseline (TD3 + behavior cloning regularization)
- [EXISTS] AWR (Advantage Weighted Regression) — Simple weighted behavior cloning for offline RL

### 1.4 LLM Alignment RL
- [EXISTS] DPO (Direct Preference Optimization)
- [EXISTS] GRPO (Group Relative Policy Optimization)
- [EXISTS] Reward Model (Ensemble-based)
- [EXISTS] KTO (Kahneman-Tversky Optimization) — Binary labels only, prospect theory-based (ICML 2024)
- [EXISTS] SimPO (Simple Preference Optimization) — Reference-free, length-normalized reward (NeurIPS 2024)
- [EXISTS] ORPO (Odds Ratio Preference Optimization) — Monolithic SFT+alignment in one stage (EMNLP 2024)
- [EXISTS] IPO (Identity Preference Optimization) — Bounded DPO with explicit regularization
- [EXISTS] SPIN (Self-Play Fine-Tuning) — Self-play without additional human data (ICML 2024)
- [EXISTS] RLOO (REINFORCE Leave-One-Out) — Simple REINFORCE with leave-one-out baseline (ACL 2024)
- [EXISTS] ReMax — Greedy-decoding baseline for RLHF, 46% memory savings (ICML 2024)
- [EXISTS] RLVR (RL with Verifiable Rewards) — Binary correctness from symbolic verifiers (DeepSeek-R1, 2025)

### 1.5 Multi-Agent RL
- [EXISTS] MAPPO (Multi-Agent PPO) — PPO with centralized critic for cooperative MARL
- [EXISTS] QMIX — Monotonic value factorization for cooperative MARL
- [EXISTS] WQMIX (Weighted QMIX) — Relaxed monotonicity via importance weighting
- [EXISTS] MADDPG (Multi-Agent DDPG) — Centralized training, decentralized execution for mixed cooperative-competitive
- [EXISTS] QPLEX — Duplex dueling for complete IGM decomposition

### 1.6 Model-Based RL
- [EXISTS] DreamerV3 — Universal world model, learns latent dynamics, 150+ tasks (Nature 2025)
- [EXISTS] TD-MPC2 — Scalable MPC with learned latent dynamics (ICLR 2024)
- [EXISTS] MBPO (Model-Based Policy Optimization) — Short branched rollouts for sample efficiency

### 1.7 Exploration
- [EXISTS] ICM (Intrinsic Curiosity Module)
- [EXISTS] RND (Random Network Distillation) — Exploration via prediction error on random network
- [EXISTS] Go-Explore — Archive-based exploration for hard-exploration tasks

### 1.8 Sequence-Based RL
- [EXISTS] Decision Transformer
- [EXISTS] Elastic Decision Transformer (EDT) — Adaptive history length for trajectory stitching (NeurIPS 2023)
- [EXISTS] Online Decision Transformer — DT fine-tuned with online interaction

### 1.9 Reward Modeling
- [EXISTS] Enhanced Reward Model (ensemble)
- [EXISTS] PRM (Probabilistic Roadmap)
- [EXISTS] Process Reward Model (PRM) — Step-level verification for reasoning (OpenAI, 2023)
- [EXISTS] Outcome Reward Model (ORM) — Final-answer-only reward evaluation
- [EXISTS] Generative Reward Model — LLM-as-judge reward signal (RLAIF)

### 1.10 Planning
- [EXISTS] MCTS (Monte Carlo Tree Search)
- [EXISTS] PRM Agent (A* planning)
- [EXISTS] AlphaZero-style Self-Play — Combines MCTS with neural evaluation

---

## 2. ATTENTION MECHANISMS

### 2.1 Core Attention
- [EXISTS] Multi-Head Attention (with RoPE)
- [EXISTS] Flash Attention
- [EXISTS] Cross Attention
- [EXISTS] Self Attention (with caching)
- [EXISTS] Grouped Query Attention (GQA)
- [EXISTS] Sparse Attention
- [EXISTS] Linear Attention
- [EXISTS] Efficient Attention
- [EXISTS] Sliding Window Attention
- [EXISTS] Ring Attention
- [EXISTS] Differential Attention
- [EXISTS] Latent Attention
- [EXISTS] Context Compression
- [EXISTS] Path Attention
- [EXISTS] Spatial Attention
- [EXISTS] Temporal Attention
- [EXISTS] Unified Attention
- [EXISTS] FlashAttention-3 — Hopper GPU asynchrony + FP8 (2024)
- [EXISTS] PagedAttention — OS-inspired virtual memory for KV cache (vLLM, 2023)
- [EXISTS] Multi-Head Latent Attention (MLA) — Low-rank KV compression, 93% cache reduction (DeepSeek-V2, 2024)
- [EXISTS] Neighborhood Attention (NATTEN) — Sliding-window self-attention for vision (CVPR 2023)
- [EXISTS] SwitchHead (MoE Attention) — Expert routing for attention heads (NeurIPS 2024)

### 2.2 Chunked / Prefill
- [EXISTS] Chunked Prefill

---

## 3. STATE SPACE MODELS (SSMs)

- [EXISTS] Mamba (Selective SSM / S6)
- [EXISTS] HGRN (Hierarchical GRN)
- [EXISTS] DeltaNet
- [EXISTS] RetNet
- [EXISTS] Linear RNN
- [EXISTS] Mamba-2 (SSD) — State Space Duality, 2-8x faster than Mamba-1 (ICML 2024)
- [EXISTS] S4 (Structured State Space) — Foundational SSM with HiPPO initialization
- [EXISTS] S4D (Diagonal State Spaces) — Simplified S4 with diagonal matrices
- [EXISTS] S5 (Simplified State Space) — MIMO SSM with parallel scans (ICLR 2023 Oral)
- [EXISTS] Liquid-S4 — Input-dependent state transitions (ICLR 2023)
- [EXISTS] Gated Delta Networks (GDN) — Combines Mamba2 gating + DeltaNet delta rule (ICLR 2025)
- [EXISTS] RWKV-6 (Finch) — RNN with matrix-valued states and dynamic recurrence (COLM 2024)
- [EXISTS] RWKV-7 (Goose) — Generalized delta rule with vector-valued gating (COLM 2025)
- [EXISTS] RWKV (basic WKV component)

---

## 4. EFFICIENT / HYBRID ARCHITECTURES

- [EXISTS] Griffin — Gated linear recurrences + local attention hybrid (Google DeepMind, 2024)
- [EXISTS] Hyena — Subquadratic attention replacement via long convolutions (ICML 2023)
- [EXISTS] Based — Linear attention + sliding window, 24x throughput vs FlashAttention-2 (ICML 2024)
- [EXISTS] Jamba — Transformer-Mamba-MoE hybrid, 256K context (AI21, ICLR 2025)
- [EXISTS] StripedHyena — Attention-Hyena hybrid, 128K context (Together AI, 2023)
- [EXISTS] Zamba — Mamba backbone + shared attention block (Zyphra, 2024)
- [EXISTS] GoldFinch — RWKV-Transformer hybrid, 756-2550x KV cache compression (2024)
- [EXISTS] RecurrentGemma — Griffin-based open model (Google, 2024)
- [EXISTS] Hawk — Pure gated linear recurrence model (Google, 2024)

---

## 5. POSITIONAL ENCODINGS

- [EXISTS] Sinusoidal PE
- [EXISTS] Learned PE
- [EXISTS] Rotary Embeddings (RoPE)
- [EXISTS] Relative Bias
- [EXISTS] ALiBi
- [EXISTS] Multiscale RoPE
- [EXISTS] YaRN
- [EXISTS] CoPE (Contextual Position Encoding)
- [EXISTS] NTK-Aware RoPE Scaling — Non-uniform frequency scaling for context extension
- [EXISTS] LongRoPE — Evolutionary search for 2M+ token context (Microsoft, 2024)
- [EXISTS] FIRE — Functional interpolation for relative positions (CMU/Google, 2023)
- [EXISTS] Resonance RoPE — Integer-wavelength snapping for better extrapolation (ACL 2024)
- [EXISTS] CLEX — Continuous length extrapolation (EMNLP 2024)

---

## 6. ARCHITECTURE COMPONENTS

### 6.1 Mixture of Experts
- [EXISTS] MoE Layer / Router / Experts / Gating
- [EXISTS] Enhanced MoE / MoE Transformer
- [EXISTS] Mixture-of-Depths (MoD) — Skip layers per-token via routing (Google DeepMind, 2024)
- [EXISTS] Mixture-of-Agents (MoA) — Multi-LLM layered collaboration (Together AI, 2024)
- [EXISTS] DeepSeek MoE — Shared + routed experts with fine-grained segmentation (2024)
- [EXISTS] Switch Transformer — Top-1 expert routing with load balancing
- [EXISTS] SwitchAll — Fully MoE attention + FFN (NeurIPS 2024)

### 6.2 Normalization
- [EXISTS] Normalization layers (basic)
- [EXISTS] RMSNorm — Root mean square normalization (standard for modern LLMs)
- [EXISTS] DeepNorm — Scaling to 1000-layer transformers (Microsoft, 2022)
- [EXISTS] QK-Norm — Query-Key normalization for attention stability
- [EXISTS] HybridNorm — Pre-Norm attention + Post-Norm FFN (2025)
- [EXISTS] DyT (Dynamic Tanh) — Normalization-free transformers (CVPR 2025)

### 6.3 Activation Functions
- [EXISTS] Custom activations (basic)
- [EXISTS] SwiGLU — Swish-gated linear unit (dominant in modern LLMs)
- [EXISTS] GeGLU — GELU-gated linear unit (used in Gemma)
- [EXISTS] ReGLU — ReLU-gated linear unit

### 6.4 Layers
- [EXISTS] Depthwise Separable Conv
- [EXISTS] SE Block
- [EXISTS] Drop Path

---

## 7. INFERENCE OPTIMIZATIONS

- [EXISTS] KV Cache (with prefix caching)
- [EXISTS] Speculative Decoding
- [EXISTS] Continuous Batching
- [EXISTS] Multi-Token Prediction
- [EXISTS] EAGLE-3 — Feature-based speculative decoding, 3-6.5x speedup (NeurIPS 2025)
- [EXISTS] Medusa — Multi-head parallel token prediction (ICML 2024)
- [EXISTS] Lookahead Decoding — Jacobi iteration for parallel n-gram generation (ICML 2024)
- [EXISTS] PagedAttention — Virtual memory for KV cache management (vLLM)
- [EXISTS] Quantized KV Cache — INT8/FP8 KV cache for memory reduction

---

## 8. COMPUTER VISION

### 8.1 Vision Transformers
- [EXISTS] ViT
- [EXISTS] Swin Transformer
- [EXISTS] HiViT
- [EXISTS] EfficientNet
- [EXISTS] Compact CNN
- [EXISTS] ResNet / VGG
- [EXISTS] DINOv2 — Self-supervised ViT with self-distillation (Meta, 2023)
- [EXISTS] SigLIP — Sigmoid loss CLIP, efficient vision-language pre-training (Google, 2023)
- [EXISTS] EVA-02 — MIM pre-trained ViT with RoPE + SwiGLU (BAAI, 2023)
- [EXISTS] InternVL — Open-source multimodal ViT family (CVPR 2024)

### 8.2 Object Detection
- [EXISTS] DETR
- [EXISTS] Faster R-CNN / Cascade R-CNN / Mask R-CNN / Keypoint R-CNN
- [EXISTS] RT-DETR — Real-time DETR beating YOLOs (CVPR 2024)
- [EXISTS] YOLO-World — Open-vocabulary YOLO with text input (CVPR 2024)
- [EXISTS] Grounding DINO — Open-set detection with text queries (ECCV 2024)
- [EXISTS] YOLOv10 — NMS-free end-to-end detection (2024)

### 8.3 Segmentation
- [EXISTS] SAM (Segment Anything Model) — Promptable segmentation foundation model (Meta, 2023)
- [EXISTS] SAM 2 — Video segmentation with streaming memory (Meta, 2024)
- [EXISTS] MedSAM — Universal medical image segmentation (Nature Comms, 2024)

### 8.4 NeRF / 3D
- [EXISTS] NeRF / Fast NeRF / Mip-NeRF / NeRF++
- [EXISTS] Gaussian Splatting (with loss, covariance, renderer)
- [EXISTS] Zip-NeRF — Anti-aliased grid-based NeRF, 24x faster (ICCV 2023)
- [EXISTS] DreamGaussian — Single-image to 3D in 2 minutes (ICLR 2024)
- [EXISTS] SuGaR — Surface-aligned Gaussian Splatting for mesh extraction (CVPR 2024)
- [EXISTS] GaussianEditor — Text-guided 3D scene editing (CVPR 2024)
- [EXISTS] LRM (Large Reconstruction Model) — Single-image to NeRF in 5 seconds (ICLR 2024)
- [EXISTS] ProlificDreamer — VSD for high-fidelity text-to-3D (NeurIPS 2023)

---

## 9. GENERATIVE MODELS

### 9.1 Diffusion / Flow
- [EXISTS] Base Diffusion / Conditional Diffusion / Stable Diffusion / UNet
- [EXISTS] DiT (Diffusion Transformer) — Transformer replaces U-Net in diffusion (ICCV 2023)
- [EXISTS] MMDiT (Multimodal Diffusion Transformer) — Dual-stream text-image transformer (SD3, 2024)
- [EXISTS] Consistency Models — Single-step generation from diffusion ODE (ICML 2023)
- [EXISTS] Latent Consistency Models (LCM) — 2-4 step generation from SD (2023)
- [EXISTS] Rectified Flow — Straight-path ODE transport (ICLR 2023, powers SD3/FLUX)
- [EXISTS] Flow Matching — Simulation-free CNF training (ICLR 2023)
- [EXISTS] PixArt-alpha — Efficient DiT training at 10% cost (2023)

### 9.2 GANs
- [EXISTS] Base GAN / Conditional GAN / CycleGAN / WGAN

### 9.3 VAE
- [EXISTS] VAE (encoder/decoder)

### 9.4 Video Generation
- [EXISTS] CogVideoX — Expert transformer for text-to-video (ICLR 2025)
- [EXISTS] VideoPoet — Autoregressive multimodal video LLM (Google, ICML 2024)

### 9.5 Audio / Speech
- [EXISTS] VALL-E — Neural codec language model for zero-shot TTS (Microsoft, 2023)
- [EXISTS] Voicebox — Flow-matching speech generation (Meta, ICLR 2024)
- [EXISTS] SoundStorm — Non-autoregressive parallel audio generation (Google, 2023)
- [EXISTS] MusicGen — Controllable music generation (Meta, NeurIPS 2023)
- [EXISTS] NaturalSpeech 3 — Factorized codec + diffusion TTS (ICML 2024)

---

## 10. NLP / LLM

### 10.1 Language Models
- [EXISTS] GPT / GPT-4o / LLaMA / Falcon / Bloom / Qwen / Edge LLM / Base LLM
- [EXISTS] T5 / Longformer
- [EXISTS] LSTM / GRU / Bidirectional RNN

### 10.2 Reasoning
- [EXISTS] Chain-of-Thought
- [EXISTS] Tree of Thoughts (ToT) — BFS/DFS search over reasoning paths (NeurIPS 2023)
- [EXISTS] Graph of Thoughts (GoT) — DAG-based reasoning with merging/refining (2023)
- [EXISTS] Self-Consistency (CoT-SC) — Majority vote over multiple reasoning chains (ICLR 2023)
- [EXISTS] ReAct — Interleaved reasoning + action loop (ICLR 2023)

### 10.3 RAG
- [EXISTS] RAG Module / Document Encoder / Retriever
- [EXISTS] Self-RAG — Self-reflective retrieval with reflection tokens (ICLR 2024)
- [EXISTS] CRAG (Corrective RAG) — Confidence-gated retrieval actions (ICML 2024)
- [EXISTS] GraphRAG — Knowledge graph + community-based retrieval (Microsoft, 2024)
- [EXISTS] RAPTOR — Recursive tree-structured summarization for retrieval (ICLR 2024)
- [EXISTS] Adaptive RAG — Complexity-based retrieval routing (2024)

### 10.4 Fine-Tuning (PEFT)
- [EXISTS] SFT (Supervised Fine-Tuning)
- [EXISTS] LoRA — Low-rank adaptation (ICLR 2022, foundational)
- [EXISTS] QLoRA — 4-bit quantized LoRA (NeurIPS 2023)
- [EXISTS] DoRA — Weight-decomposed LoRA (ICML 2024 Oral)
- [EXISTS] LoRA+ — Asymmetric learning rates (ICML 2024)
- [EXISTS] GaLore — Gradient low-rank projection, 7B on single 24GB GPU (ICML 2024)
- [EXISTS] LISA — Layerwise importance sampled AdamW (NeurIPS 2024)
- [EXISTS] AdaLoRA — Adaptive rank allocation (ICLR 2023)
- [EXISTS] rsLoRA — Rank-stabilized scaling (2023)

### 10.5 Quantization
- [EXISTS] GPTQ — Second-order weight quantization (ICLR 2023)
- [EXISTS] AWQ — Activation-aware weight quantization (MLSys 2024 Best Paper)
- [EXISTS] QuIP# — 2-bit quantization with E8 lattice codebooks (ICML 2024)
- [EXISTS] SqueezeLLM — Dense-and-sparse hybrid quantization (ICML 2024)
- [EXISTS] AQLM — Additive multi-codebook quantization (ICML 2024)

### 10.6 Pruning / Sparsity
- [EXISTS] SparseGPT — One-shot 50-60% pruning (ICML 2023)
- [EXISTS] Wanda — Prune by weight x activation magnitude (ICLR 2024)
- [EXISTS] SliceGPT — Delete rows/columns via orthogonal transforms (ICLR 2024)
- [EXISTS] ShortGPT — Layer removal by block influence score (2024)

### 10.7 Knowledge Distillation
- [EXISTS] Knowledge Distiller (soft/hard targets)
- [EXISTS] Rationale-based KD — Distill reasoning chains, not just outputs (2023)
- [EXISTS] Minitron — Pruning + KD for compact LLMs (NVIDIA, NeurIPS 2024)

### 10.8 Structured Generation
- [EXISTS] Grammar-Constrained Decoding — Formal grammar constraints during generation
- [EXISTS] JSON Schema Decoder — FSM-based structured output (Outlines-style)

### 10.9 Embedding Models
- [EXISTS] Matryoshka Representation Learning — Nested multi-granularity embeddings (NeurIPS 2022)
- [EXISTS] BGE-M3 — Dense + sparse + ColBERT in one model (BAAI, 2024)

### 10.10 Tokenization
- [EXISTS] Byte Latent Transformer (BLT) — Tokenizer-free, entropy-based dynamic patching (Meta, 2024)
- [EXISTS] MambaByte — Byte-level Mamba without tokenization (COLM 2024)

---

## 11. TRAINING INFRASTRUCTURE

### 11.1 Optimizers
- [EXISTS] Adam / AdamW (used in agents)
- [EXISTS] Lion — Sign-based momentum, halves optimizer memory (NeurIPS 2023)
- [EXISTS] Sophia — Second-order with diagonal Hessian, 2x speedup (ICLR 2024)
- [EXISTS] Prodigy — Learning-rate-free optimizer (ICML 2024)
- [EXISTS] Schedule-Free AdamW — Eliminates LR schedules entirely (2024)
- [EXISTS] SOAP — Adam + Shampoo preconditioner (2024)
- [EXISTS] Muon — Momentum orthogonalized by Newton-Schulz (2024-2025)

### 11.2 Learning Rate Schedules
- [EXISTS] Cosine Annealing (in PPO)
- [EXISTS] WSD (Warmup-Stable-Decay) — No fixed compute budget needed (DeepSeek-V3, 2024)
- [EXISTS] Cosine with Warm Restarts (SGDR) — Periodic LR resets

### 11.3 Mixed Precision
- [EXISTS] FP8 / Gradient Scaling / Config
- [EXISTS] MXFP8 (Microscaling FP8) — Block-level scaling for better FP8 (OCP, 2024)
- [EXISTS] FP4 / MXFP4 Training — 4-bit training with microscaling (2025)

### 11.4 Distributed Training
- [EXISTS] Distributed training support
- [EXISTS] FSDP2 — Next-gen PyTorch fully sharded data parallelism
- [EXISTS] ZeRO++ — 4x communication reduction (Microsoft, 2023)
- [EXISTS] Context Parallelism — Sequence-length parallelism for long context

### 11.5 Loss Functions
- [EXISTS] Cross-entropy / KL / Distillation losses
- [EXISTS] InfoNCE — Foundational contrastive loss
- [EXISTS] SigLIP Loss — Sigmoid contrastive (no global softmax)
- [EXISTS] VICReg — Variance-invariance-covariance regularization

### 11.6 Gradient Methods
- [EXISTS] Selective Activation Checkpointing — Per-operation save/recompute (PyTorch 2.4+)
- [EXISTS] Activation Offloading — CPU offload during forward, prefetch for backward

---

## 12. SELF-SUPERVISED LEARNING

- [EXISTS] DINOv2 — Self-distillation ViT (Meta, 2023)
- [EXISTS] I-JEPA — Joint-embedding prediction in representation space (Meta, CVPR 2023)
- [EXISTS] V-JEPA 2 — Video world model from 1M+ hours (Meta, 2025)
- [EXISTS] data2vec 2.0 — Efficient multimodal SSL (Meta, ICML 2023)
- [EXISTS] MAE (Masked Autoencoder) — Masked image/video/audio modeling
- [EXISTS] Barlow Twins — Self-supervised via redundancy reduction
- [EXISTS] VICReg — Non-contrastive SSL with variance-covariance regularization

---

## 13. MULTIMODAL MODELS

- [EXISTS] LLaVA-RLHF / PaLM-E / HiViLT / Enhanced Transformer
- [EXISTS] NVLM (base + processor)
- [EXISTS] LLaVA-NeXT / LLaVA-OneVision — Open-source multimodal LLM family (2024)
- [EXISTS] Qwen2-VL — Dynamic resolution + M-RoPE (Alibaba, 2024)
- [EXISTS] Molmo — Fully open VLM (AI2, 2024)
- [EXISTS] Phi-3-Vision — Lightweight multimodal with 128K context (Microsoft, 2024)
- [EXISTS] BiomedCLIP — Biomedical vision-language (Microsoft, 2023)

---

## 14. GRAPH NEURAL NETWORKS

- [EXISTS] Base GNN / Message Passing / Hierarchical / Attention-based
- [EXISTS] GPS (Graph Transformer) — Modular graph transformer framework (NeurIPS 2022)
- [EXISTS] Exphormer — Sparse graph transformer with expander graphs (ICML 2023)
- [EXISTS] Graph Attention Network v2 (GATv2) — Dynamic attention for graphs
- [EXISTS] GraphSAGE — Inductive representation learning on large graphs

---

## 15. WORLD MODELS

- [EXISTS] DreamerV3 — Latent world model, 150+ tasks (Nature, 2025)
- [EXISTS] I-JEPA — Image joint-embedding predictive architecture (Meta, 2023)
- [EXISTS] V-JEPA 2 — Video world model for robot planning (Meta, 2025)
- [EXISTS] Genie / Genie 2 — Interactive environment generation from video (Google, 2024)

---

## 16. CONTINUAL LEARNING

- [EXISTS] EVCL — Elastic variational continual learning (ICML 2024)
- [EXISTS] Self-Synthesized Rehearsal — LLM self-generates replay data (ACL 2024)
- [EXISTS] Prompt-Based CL (L2P, DualPrompt, CODA-Prompt) — Learnable prompts per task
- [EXISTS] EWC (Elastic Weight Consolidation) — Classic regularization approach

---

## 17. AUTONOMOUS DRIVING

- [EXISTS] Perception / Motion Prediction / Behavior / Planning / Decision / Sensor Fusion
- [EXISTS] UniAD — Unified end-to-end driving (CVPR 2023 Best Paper)
- [EXISTS] VAD — Vectorized autonomous driving (ICCV 2023)
- [EXISTS] DriveTransformer — Unified transformer for scalable E2E driving (2025)

---

## 18. IMITATION LEARNING

- [EXISTS] GAIL (Generative Adversarial Imitation Learning) — Adversarial policy matching
- [EXISTS] DAgger — Dataset aggregation with expert queries
- [EXISTS] MEGA-DAgger — Multi-expert DAgger for imperfect demos (2024)
- [EXISTS] AIRL — Adversarial inverse RL with transferable rewards

---

## 19. TEST-TIME COMPUTE

- [EXISTS] Test-Time Training (TTT) Layers — Hidden state as ML model updated at inference (2024)
- [EXISTS] Compute-Optimal Scaling — Adaptive per-prompt test-time compute allocation (ICLR 2025)
- [EXISTS] Best-of-N with PRM — Sample multiple, verify with process reward model

---

## 20. DIFFUSION TRAINING ADVANCES

- [EXISTS] Flow Matching — Simulation-free CNF training (ICLR 2023)
- [EXISTS] Rectified Flow — Straight-path ODE (ICLR 2023)
- [EXISTS] Consistency Training (iCT) — Improved standalone consistency training (2023)
- [EXISTS] Easy Consistency Tuning (ECT) — Bootstrap from diffusion models (ICLR 2025)

---

## PRIORITY IMPLEMENTATION ORDER

### Tier 1 — High Impact, Widely Used
1. LoRA / QLoRA / DoRA (PEFT methods)
2. Mamba-2 (SSD)
3. SwiGLU / GeGLU activations
4. RMSNorm
5. DiT (Diffusion Transformer)
6. SAM (Segment Anything)
7. Flow Matching / Rectified Flow
8. DreamerV3 (world model)
9. KTO / SimPO / ORPO (alignment)
10. Lion / Sophia optimizers

### Tier 2 — Important for Research Coverage
11. RWKV-6/7
12. Griffin / Jamba hybrid architectures
13. Self-RAG / GraphRAG
14. GPTQ / AWQ quantization
15. SparseGPT / Wanda pruning
16. Tree of Thoughts / ReAct
17. MAPPO / QMIX (multi-agent RL)
18. Consistency Models
19. PagedAttention / EAGLE speculative decoding
20. GPS graph transformer

### Tier 3 — Specialized / Emerging
21. V-JEPA / I-JEPA
22. MLA (Multi-Head Latent Attention)
23. GaLore / LISA (memory-efficient training)
24. Byte Latent Transformer
25. Mixture-of-Depths
26. Gated Delta Networks
27. Cal-QL / IDQL (offline RL)
28. RT-DETR / YOLO-World
29. DreamGaussian / SuGaR
30. VALL-E / Voicebox (speech)

# Nexus AI Module - Comprehensive Research Implementation Todo

> Cross-referenced against latest academic research (2023-2025).
> `[EXISTS]` = already implemented, `[ ]` = not yet implemented.

---

## 1. REINFORCEMENT LEARNING

### 1.1 Value-Based Methods
- [EXISTS] DQN (Deep Q-Network)
- [EXISTS] Double DQN (DDQN)
- [EXISTS] Dueling DQN
- [ ] Rainbow DQN — Combines 6 DQN improvements (prioritized replay, n-step, distributional, noisy nets, dueling, double)
- [ ] C51 (Categorical DQN) — Distributional RL with categorical value distribution
- [ ] QR-DQN (Quantile Regression DQN) — Distributional RL with quantile-based value estimation

### 1.2 Policy Gradient / Actor-Critic
- [EXISTS] REINFORCE (with baseline)
- [EXISTS] A2C (Advantage Actor-Critic)
- [EXISTS] PPO (Proximal Policy Optimization)
- [EXISTS] DDPG (Deep Deterministic Policy Gradient)
- [EXISTS] TD3 (Twin Delayed DDPG)
- [EXISTS] SAC (Soft Actor-Critic)
- [ ] TRPO (Trust Region Policy Optimization) — Predecessor to PPO with guaranteed monotonic improvement
- [ ] EPO (Evolutionary Policy Optimization) — Hybrid evolutionary + policy gradient (2025)

### 1.3 Offline RL
- [EXISTS] IQL (Implicit Q-Learning)
- [EXISTS] CQL (Conservative Q-Learning)
- [ ] Cal-QL (Calibrated Q-Learning) — Conservative but calibrated values for offline-to-online (NeurIPS 2023)
- [ ] IDQL (Implicit Diffusion Q-Learning) — Diffusion policies + IQL (2023)
- [ ] ReBRAC — Minimalist TD3+BC with design improvements (NeurIPS 2023)
- [ ] EDAC (Ensemble-Diversified Actor Critic) — Diversified Q-ensemble for offline RL
- [ ] TD3+BC — Simple offline RL baseline (TD3 + behavior cloning regularization)
- [ ] AWR (Advantage Weighted Regression) — Simple weighted behavior cloning for offline RL

### 1.4 LLM Alignment RL
- [EXISTS] DPO (Direct Preference Optimization)
- [EXISTS] GRPO (Group Relative Policy Optimization)
- [EXISTS] Reward Model (Ensemble-based)
- [ ] KTO (Kahneman-Tversky Optimization) — Binary labels only, prospect theory-based (ICML 2024)
- [ ] SimPO (Simple Preference Optimization) — Reference-free, length-normalized reward (NeurIPS 2024)
- [ ] ORPO (Odds Ratio Preference Optimization) — Monolithic SFT+alignment in one stage (EMNLP 2024)
- [ ] IPO (Identity Preference Optimization) — Bounded DPO with explicit regularization
- [ ] SPIN (Self-Play Fine-Tuning) — Self-play without additional human data (ICML 2024)
- [ ] RLOO (REINFORCE Leave-One-Out) — Simple REINFORCE with leave-one-out baseline (ACL 2024)
- [ ] ReMax — Greedy-decoding baseline for RLHF, 46% memory savings (ICML 2024)
- [ ] RLVR (RL with Verifiable Rewards) — Binary correctness from symbolic verifiers (DeepSeek-R1, 2025)

### 1.5 Multi-Agent RL
- [ ] MAPPO (Multi-Agent PPO) — PPO with centralized critic for cooperative MARL
- [ ] QMIX — Monotonic value factorization for cooperative MARL
- [ ] WQMIX (Weighted QMIX) — Relaxed monotonicity via importance weighting
- [ ] MADDPG (Multi-Agent DDPG) — Centralized training, decentralized execution for mixed cooperative-competitive
- [ ] QPLEX — Duplex dueling for complete IGM decomposition

### 1.6 Model-Based RL
- [ ] DreamerV3 — Universal world model, learns latent dynamics, 150+ tasks (Nature 2025)
- [ ] TD-MPC2 — Scalable MPC with learned latent dynamics (ICLR 2024)
- [ ] MBPO (Model-Based Policy Optimization) — Short branched rollouts for sample efficiency

### 1.7 Exploration
- [EXISTS] ICM (Intrinsic Curiosity Module)
- [ ] RND (Random Network Distillation) — Exploration via prediction error on random network
- [ ] Go-Explore — Archive-based exploration for hard-exploration tasks

### 1.8 Sequence-Based RL
- [EXISTS] Decision Transformer
- [ ] Elastic Decision Transformer (EDT) — Adaptive history length for trajectory stitching (NeurIPS 2023)
- [ ] Online Decision Transformer — DT fine-tuned with online interaction

### 1.9 Reward Modeling
- [EXISTS] Enhanced Reward Model (ensemble)
- [EXISTS] PRM (Probabilistic Roadmap)
- [ ] Process Reward Model (PRM) — Step-level verification for reasoning (OpenAI, 2023)
- [ ] Outcome Reward Model (ORM) — Final-answer-only reward evaluation
- [ ] Generative Reward Model — LLM-as-judge reward signal (RLAIF)

### 1.10 Planning
- [EXISTS] MCTS (Monte Carlo Tree Search)
- [EXISTS] PRM Agent (A* planning)
- [ ] AlphaZero-style Self-Play — Combines MCTS with neural evaluation

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
- [ ] FlashAttention-3 — Hopper GPU asynchrony + FP8 (2024)
- [ ] PagedAttention — OS-inspired virtual memory for KV cache (vLLM, 2023)
- [ ] Multi-Head Latent Attention (MLA) — Low-rank KV compression, 93% cache reduction (DeepSeek-V2, 2024)
- [ ] Neighborhood Attention (NATTEN) — Sliding-window self-attention for vision (CVPR 2023)
- [ ] SwitchHead (MoE Attention) — Expert routing for attention heads (NeurIPS 2024)

### 2.2 Chunked / Prefill
- [EXISTS] Chunked Prefill

---

## 3. STATE SPACE MODELS (SSMs)

- [EXISTS] Mamba (Selective SSM / S6)
- [EXISTS] HGRN (Hierarchical GRN)
- [EXISTS] DeltaNet
- [EXISTS] RetNet
- [EXISTS] Linear RNN
- [ ] Mamba-2 (SSD) — State Space Duality, 2-8x faster than Mamba-1 (ICML 2024)
- [ ] S4 (Structured State Space) — Foundational SSM with HiPPO initialization
- [ ] S4D (Diagonal State Spaces) — Simplified S4 with diagonal matrices
- [ ] S5 (Simplified State Space) — MIMO SSM with parallel scans (ICLR 2023 Oral)
- [ ] Liquid-S4 — Input-dependent state transitions (ICLR 2023)
- [ ] Gated Delta Networks (GDN) — Combines Mamba2 gating + DeltaNet delta rule (ICLR 2025)
- [ ] RWKV-6 (Finch) — RNN with matrix-valued states and dynamic recurrence (COLM 2024)
- [ ] RWKV-7 (Goose) — Generalized delta rule with vector-valued gating (COLM 2025)
- [EXISTS] RWKV (basic WKV component)

---

## 4. EFFICIENT / HYBRID ARCHITECTURES

- [ ] Griffin — Gated linear recurrences + local attention hybrid (Google DeepMind, 2024)
- [ ] Hyena — Subquadratic attention replacement via long convolutions (ICML 2023)
- [ ] Based — Linear attention + sliding window, 24x throughput vs FlashAttention-2 (ICML 2024)
- [ ] Jamba — Transformer-Mamba-MoE hybrid, 256K context (AI21, ICLR 2025)
- [ ] StripedHyena — Attention-Hyena hybrid, 128K context (Together AI, 2023)
- [ ] Zamba — Mamba backbone + shared attention block (Zyphra, 2024)
- [ ] GoldFinch — RWKV-Transformer hybrid, 756-2550x KV cache compression (2024)
- [ ] RecurrentGemma — Griffin-based open model (Google, 2024)
- [ ] Hawk — Pure gated linear recurrence model (Google, 2024)

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
- [ ] NTK-Aware RoPE Scaling — Non-uniform frequency scaling for context extension
- [ ] LongRoPE — Evolutionary search for 2M+ token context (Microsoft, 2024)
- [ ] FIRE — Functional interpolation for relative positions (CMU/Google, 2023)
- [ ] Resonance RoPE — Integer-wavelength snapping for better extrapolation (ACL 2024)
- [ ] CLEX — Continuous length extrapolation (EMNLP 2024)

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
- [ ] EAGLE-3 — Feature-based speculative decoding, 3-6.5x speedup (NeurIPS 2025)
- [ ] Medusa — Multi-head parallel token prediction (ICML 2024)
- [ ] Lookahead Decoding — Jacobi iteration for parallel n-gram generation (ICML 2024)
- [ ] PagedAttention — Virtual memory for KV cache management (vLLM)
- [ ] Quantized KV Cache — INT8/FP8 KV cache for memory reduction

---

## 8. COMPUTER VISION

### 8.1 Vision Transformers
- [EXISTS] ViT
- [EXISTS] Swin Transformer
- [EXISTS] HiViT
- [EXISTS] EfficientNet
- [EXISTS] Compact CNN
- [EXISTS] ResNet / VGG
- [ ] DINOv2 — Self-supervised ViT with self-distillation (Meta, 2023)
- [ ] SigLIP — Sigmoid loss CLIP, efficient vision-language pre-training (Google, 2023)
- [ ] EVA-02 — MIM pre-trained ViT with RoPE + SwiGLU (BAAI, 2023)
- [ ] InternVL — Open-source multimodal ViT family (CVPR 2024)

### 8.2 Object Detection
- [EXISTS] DETR
- [EXISTS] Faster R-CNN / Cascade R-CNN / Mask R-CNN / Keypoint R-CNN
- [ ] RT-DETR — Real-time DETR beating YOLOs (CVPR 2024)
- [ ] YOLO-World — Open-vocabulary YOLO with text input (CVPR 2024)
- [ ] Grounding DINO — Open-set detection with text queries (ECCV 2024)
- [ ] YOLOv10 — NMS-free end-to-end detection (2024)

### 8.3 Segmentation
- [ ] SAM (Segment Anything Model) — Promptable segmentation foundation model (Meta, 2023)
- [ ] SAM 2 — Video segmentation with streaming memory (Meta, 2024)
- [ ] MedSAM — Universal medical image segmentation (Nature Comms, 2024)

### 8.4 NeRF / 3D
- [EXISTS] NeRF / Fast NeRF / Mip-NeRF / NeRF++
- [EXISTS] Gaussian Splatting (with loss, covariance, renderer)
- [ ] Zip-NeRF — Anti-aliased grid-based NeRF, 24x faster (ICCV 2023)
- [ ] DreamGaussian — Single-image to 3D in 2 minutes (ICLR 2024)
- [ ] SuGaR — Surface-aligned Gaussian Splatting for mesh extraction (CVPR 2024)
- [ ] GaussianEditor — Text-guided 3D scene editing (CVPR 2024)
- [ ] LRM (Large Reconstruction Model) — Single-image to NeRF in 5 seconds (ICLR 2024)
- [ ] ProlificDreamer — VSD for high-fidelity text-to-3D (NeurIPS 2023)

---

## 9. GENERATIVE MODELS

### 9.1 Diffusion / Flow
- [EXISTS] Base Diffusion / Conditional Diffusion / Stable Diffusion / UNet
- [ ] DiT (Diffusion Transformer) — Transformer replaces U-Net in diffusion (ICCV 2023)
- [ ] MMDiT (Multimodal Diffusion Transformer) — Dual-stream text-image transformer (SD3, 2024)
- [ ] Consistency Models — Single-step generation from diffusion ODE (ICML 2023)
- [ ] Latent Consistency Models (LCM) — 2-4 step generation from SD (2023)
- [ ] Rectified Flow — Straight-path ODE transport (ICLR 2023, powers SD3/FLUX)
- [ ] Flow Matching — Simulation-free CNF training (ICLR 2023)
- [ ] PixArt-alpha — Efficient DiT training at 10% cost (2023)

### 9.2 GANs
- [EXISTS] Base GAN / Conditional GAN / CycleGAN / WGAN

### 9.3 VAE
- [EXISTS] VAE (encoder/decoder)

### 9.4 Video Generation
- [ ] CogVideoX — Expert transformer for text-to-video (ICLR 2025)
- [ ] VideoPoet — Autoregressive multimodal video LLM (Google, ICML 2024)

### 9.5 Audio / Speech
- [ ] VALL-E — Neural codec language model for zero-shot TTS (Microsoft, 2023)
- [ ] Voicebox — Flow-matching speech generation (Meta, ICLR 2024)
- [ ] SoundStorm — Non-autoregressive parallel audio generation (Google, 2023)
- [ ] MusicGen — Controllable music generation (Meta, NeurIPS 2023)
- [ ] NaturalSpeech 3 — Factorized codec + diffusion TTS (ICML 2024)

---

## 10. NLP / LLM

### 10.1 Language Models
- [EXISTS] GPT / GPT-4o / LLaMA / Falcon / Bloom / Qwen / Edge LLM / Base LLM
- [EXISTS] T5 / Longformer
- [EXISTS] LSTM / GRU / Bidirectional RNN

### 10.2 Reasoning
- [EXISTS] Chain-of-Thought
- [ ] Tree of Thoughts (ToT) — BFS/DFS search over reasoning paths (NeurIPS 2023)
- [ ] Graph of Thoughts (GoT) — DAG-based reasoning with merging/refining (2023)
- [ ] Self-Consistency (CoT-SC) — Majority vote over multiple reasoning chains (ICLR 2023)
- [ ] ReAct — Interleaved reasoning + action loop (ICLR 2023)

### 10.3 RAG
- [EXISTS] RAG Module / Document Encoder / Retriever
- [ ] Self-RAG — Self-reflective retrieval with reflection tokens (ICLR 2024)
- [ ] CRAG (Corrective RAG) — Confidence-gated retrieval actions (ICML 2024)
- [ ] GraphRAG — Knowledge graph + community-based retrieval (Microsoft, 2024)
- [ ] RAPTOR — Recursive tree-structured summarization for retrieval (ICLR 2024)
- [ ] Adaptive RAG — Complexity-based retrieval routing (2024)

### 10.4 Fine-Tuning (PEFT)
- [EXISTS] SFT (Supervised Fine-Tuning)
- [ ] LoRA — Low-rank adaptation (ICLR 2022, foundational)
- [ ] QLoRA — 4-bit quantized LoRA (NeurIPS 2023)
- [ ] DoRA — Weight-decomposed LoRA (ICML 2024 Oral)
- [ ] LoRA+ — Asymmetric learning rates (ICML 2024)
- [ ] GaLore — Gradient low-rank projection, 7B on single 24GB GPU (ICML 2024)
- [ ] LISA — Layerwise importance sampled AdamW (NeurIPS 2024)
- [ ] AdaLoRA — Adaptive rank allocation (ICLR 2023)
- [ ] rsLoRA — Rank-stabilized scaling (2023)

### 10.5 Quantization
- [ ] GPTQ — Second-order weight quantization (ICLR 2023)
- [ ] AWQ — Activation-aware weight quantization (MLSys 2024 Best Paper)
- [ ] QuIP# — 2-bit quantization with E8 lattice codebooks (ICML 2024)
- [ ] SqueezeLLM — Dense-and-sparse hybrid quantization (ICML 2024)
- [ ] AQLM — Additive multi-codebook quantization (ICML 2024)

### 10.6 Pruning / Sparsity
- [ ] SparseGPT — One-shot 50-60% pruning (ICML 2023)
- [ ] Wanda — Prune by weight x activation magnitude (ICLR 2024)
- [ ] SliceGPT — Delete rows/columns via orthogonal transforms (ICLR 2024)
- [ ] ShortGPT — Layer removal by block influence score (2024)

### 10.7 Knowledge Distillation
- [EXISTS] Knowledge Distiller (soft/hard targets)
- [ ] Rationale-based KD — Distill reasoning chains, not just outputs (2023)
- [ ] Minitron — Pruning + KD for compact LLMs (NVIDIA, NeurIPS 2024)

### 10.8 Structured Generation
- [ ] Grammar-Constrained Decoding — Formal grammar constraints during generation
- [ ] JSON Schema Decoder — FSM-based structured output (Outlines-style)

### 10.9 Embedding Models
- [ ] Matryoshka Representation Learning — Nested multi-granularity embeddings (NeurIPS 2022)
- [ ] BGE-M3 — Dense + sparse + ColBERT in one model (BAAI, 2024)

### 10.10 Tokenization
- [ ] Byte Latent Transformer (BLT) — Tokenizer-free, entropy-based dynamic patching (Meta, 2024)
- [ ] MambaByte — Byte-level Mamba without tokenization (COLM 2024)

---

## 11. TRAINING INFRASTRUCTURE

### 11.1 Optimizers
- [EXISTS] Adam / AdamW (used in agents)
- [ ] Lion — Sign-based momentum, halves optimizer memory (NeurIPS 2023)
- [ ] Sophia — Second-order with diagonal Hessian, 2x speedup (ICLR 2024)
- [ ] Prodigy — Learning-rate-free optimizer (ICML 2024)
- [ ] Schedule-Free AdamW — Eliminates LR schedules entirely (2024)
- [ ] SOAP — Adam + Shampoo preconditioner (2024)
- [ ] Muon — Momentum orthogonalized by Newton-Schulz (2024-2025)

### 11.2 Learning Rate Schedules
- [EXISTS] Cosine Annealing (in PPO)
- [ ] WSD (Warmup-Stable-Decay) — No fixed compute budget needed (DeepSeek-V3, 2024)
- [ ] Cosine with Warm Restarts (SGDR) — Periodic LR resets

### 11.3 Mixed Precision
- [EXISTS] FP8 / Gradient Scaling / Config
- [ ] MXFP8 (Microscaling FP8) — Block-level scaling for better FP8 (OCP, 2024)
- [ ] FP4 / MXFP4 Training — 4-bit training with microscaling (2025)

### 11.4 Distributed Training
- [EXISTS] Distributed training support
- [ ] FSDP2 — Next-gen PyTorch fully sharded data parallelism
- [ ] ZeRO++ — 4x communication reduction (Microsoft, 2023)
- [ ] Context Parallelism — Sequence-length parallelism for long context

### 11.5 Loss Functions
- [EXISTS] Cross-entropy / KL / Distillation losses
- [ ] InfoNCE — Foundational contrastive loss
- [ ] SigLIP Loss — Sigmoid contrastive (no global softmax)
- [ ] VICReg — Variance-invariance-covariance regularization

### 11.6 Gradient Methods
- [ ] Selective Activation Checkpointing — Per-operation save/recompute (PyTorch 2.4+)
- [ ] Activation Offloading — CPU offload during forward, prefetch for backward

---

## 12. SELF-SUPERVISED LEARNING

- [ ] DINOv2 — Self-distillation ViT (Meta, 2023)
- [ ] I-JEPA — Joint-embedding prediction in representation space (Meta, CVPR 2023)
- [ ] V-JEPA 2 — Video world model from 1M+ hours (Meta, 2025)
- [ ] data2vec 2.0 — Efficient multimodal SSL (Meta, ICML 2023)
- [ ] MAE (Masked Autoencoder) — Masked image/video/audio modeling
- [ ] Barlow Twins — Self-supervised via redundancy reduction
- [ ] VICReg — Non-contrastive SSL with variance-covariance regularization

---

## 13. MULTIMODAL MODELS

- [EXISTS] LLaVA-RLHF / PaLM-E / HiViLT / Enhanced Transformer
- [EXISTS] NVLM (base + processor)
- [ ] LLaVA-NeXT / LLaVA-OneVision — Open-source multimodal LLM family (2024)
- [ ] Qwen2-VL — Dynamic resolution + M-RoPE (Alibaba, 2024)
- [ ] Molmo — Fully open VLM (AI2, 2024)
- [ ] Phi-3-Vision — Lightweight multimodal with 128K context (Microsoft, 2024)
- [ ] BiomedCLIP — Biomedical vision-language (Microsoft, 2023)

---

## 14. GRAPH NEURAL NETWORKS

- [EXISTS] Base GNN / Message Passing / Hierarchical / Attention-based
- [ ] GPS (Graph Transformer) — Modular graph transformer framework (NeurIPS 2022)
- [ ] Exphormer — Sparse graph transformer with expander graphs (ICML 2023)
- [ ] Graph Attention Network v2 (GATv2) — Dynamic attention for graphs
- [ ] GraphSAGE — Inductive representation learning on large graphs

---

## 15. WORLD MODELS

- [ ] DreamerV3 — Latent world model, 150+ tasks (Nature, 2025)
- [ ] I-JEPA — Image joint-embedding predictive architecture (Meta, 2023)
- [ ] V-JEPA 2 — Video world model for robot planning (Meta, 2025)
- [ ] Genie / Genie 2 — Interactive environment generation from video (Google, 2024)

---

## 16. CONTINUAL LEARNING

- [ ] EVCL — Elastic variational continual learning (ICML 2024)
- [ ] Self-Synthesized Rehearsal — LLM self-generates replay data (ACL 2024)
- [ ] Prompt-Based CL (L2P, DualPrompt, CODA-Prompt) — Learnable prompts per task
- [ ] EWC (Elastic Weight Consolidation) — Classic regularization approach

---

## 17. AUTONOMOUS DRIVING

- [EXISTS] Perception / Motion Prediction / Behavior / Planning / Decision / Sensor Fusion
- [ ] UniAD — Unified end-to-end driving (CVPR 2023 Best Paper)
- [ ] VAD — Vectorized autonomous driving (ICCV 2023)
- [ ] DriveTransformer — Unified transformer for scalable E2E driving (2025)

---

## 18. IMITATION LEARNING

- [ ] GAIL (Generative Adversarial Imitation Learning) — Adversarial policy matching
- [ ] DAgger — Dataset aggregation with expert queries
- [ ] MEGA-DAgger — Multi-expert DAgger for imperfect demos (2024)
- [ ] AIRL — Adversarial inverse RL with transferable rewards

---

## 19. TEST-TIME COMPUTE

- [ ] Test-Time Training (TTT) Layers — Hidden state as ML model updated at inference (2024)
- [ ] Compute-Optimal Scaling — Adaptive per-prompt test-time compute allocation (ICLR 2025)
- [ ] Best-of-N with PRM — Sample multiple, verify with process reward model

---

## 20. DIFFUSION TRAINING ADVANCES

- [ ] Flow Matching — Simulation-free CNF training (ICLR 2023)
- [ ] Rectified Flow — Straight-path ODE (ICLR 2023)
- [ ] Consistency Training (iCT) — Improved standalone consistency training (2023)
- [ ] Easy Consistency Tuning (ECT) — Bootstrap from diffusion models (ICLR 2025)

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

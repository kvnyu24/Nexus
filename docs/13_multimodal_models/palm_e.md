# PaLM-E: Embodied Multimodal Language Model

## 1. Overview & Motivation

PaLM-E (Pathways Language Model - Embodied) is Google's groundbreaking multimodal language model designed specifically for embodied AI and robotics applications. Unlike traditional vision-language models that focus on image understanding and description, PaLM-E integrates visual, language, and continuous sensor data to enable robots to understand and interact with the physical world.

### Key Innovations

- **Embodied AI Integration**: First large-scale LLM designed for robotic control and manipulation
- **Multi-Sensor Fusion**: Combines vision, language, proprioception, and other sensor modalities
- **Vision-Language-Action**: Unified model predicting both language and robot actions
- **Transfer Learning**: Pre-trained language knowledge transfers to robotics tasks
- **Real-World Deployment**: Successfully deployed on physical robot systems

### Why PaLM-E?

Traditional approaches to robotic control face several limitations:
- **Separate Vision and Control**: Vision models disconnected from action generation
- **Limited Generalization**: Task-specific models don't transfer across scenarios
- **Data Efficiency**: Robotics data is expensive and time-consuming to collect
- **Language Grounding**: Difficulty connecting natural language instructions to physical actions

PaLM-E addresses these challenges through:
1. **Unified Architecture**: Single model for perception, reasoning, and action
2. **Language Transfer**: Leverage massive language model pre-training for robotics
3. **Multi-Task Learning**: Train on diverse robotics and vision-language tasks simultaneously
4. **Embodied Reasoning**: Ground language understanding in physical world interactions

### Applications

- **Robot Manipulation**: Pick-and-place, object rearrangement, assembly tasks
- **Mobile Navigation**: Navigate to described locations based on language commands
- **Visual Question Answering**: Answer questions about the robot's environment
- **Long-Horizon Planning**: Multi-step task execution from high-level instructions
- **Sim-to-Real Transfer**: Train in simulation, deploy on real robots

## 2. Theoretical Background

### 2.1 Embodied AI Paradigm

**Embodied Intelligence**: The theory that intelligent behavior arises from the interaction between an agent's body, brain, and environment. PaLM-E embodies this by:

- **Perception-Action Loop**: Continuous sensing and acting in the world
- **Sensorimotor Integration**: Combining multiple sensor modalities with motor commands
- **Physical Grounding**: Understanding concepts through physical interaction
- **Temporal Reasoning**: Modeling sequences of states and actions over time

### 2.2 Multi-Modal Sensor Fusion

PaLM-E processes diverse sensor inputs:

**Visual Sensors**:
- RGB cameras (multiple viewpoints)
- Depth cameras (3D understanding)
- Semantic segmentation maps

**Proprioceptive Sensors**:
- Joint angles and velocities
- End-effector positions
- Force/torque sensors

**Language Input**:
- Natural language instructions
- Task descriptions
- Environmental descriptions

### 2.3 Vision-Language-Action Architecture

**Three-Way Integration**:
1. **Vision → Language**: Describe observed scenes
2. **Language → Action**: Execute language instructions
3. **Vision → Action**: Direct visual servoing

**Unified Representation**:
All modalities are projected into a shared embedding space, allowing the language model to reason over visual observations and generate both language outputs and action commands.

### 2.4 Training Methodology

**Multi-Task Co-Training**:
- Vision-language tasks (VQA, captioning, retrieval)
- Robotics tasks (manipulation, navigation)
- Pure language tasks (maintaining LLM capabilities)

**Data Sources**:
- Large-scale internet vision-language data
- Robot demonstration datasets
- Simulated robotics environments
- Human-robot interaction logs

## 3. Mathematical Formulation

### 3.1 Multi-Modal Input Encoding

**Vision Encoding**:
```
V = VisionEncoder(I_rgb, I_depth) ∈ R^(N_v × d_v)
```

**Proprioception Encoding**:
```
S = StateEncoder(joint_angles, ee_pose) ∈ R^(N_s × d_s)
```

**Language Encoding**:
```
L = LanguageEncoder(text_tokens) ∈ R^(N_l × d_l)
```

### 3.2 Projection to Common Space

Map all modalities to language model dimension:

```
V_proj = V W_v + b_v ∈ R^(N_v × d_lm)
S_proj = S W_s + b_s ∈ R^(N_s × d_lm)
L_embed = Embedding(text_tokens) ∈ R^(N_l × d_lm)
```

### 3.3 Sequence Construction

Construct unified input sequence:

```
X = [V_proj; S_proj; L_embed] ∈ R^((N_v + N_s + N_l) × d_lm)
```

The model processes this as a single sequence with positional encodings indicating modality type.

### 3.4 Autoregressive Generation

**For Language Output**:
```
P(word_t | X, words_<t) = Softmax(LM(X)_t W_vocab)
```

**For Action Output**:
```
action_t = ActionHead(LM(X)_t) ∈ R^(d_action)
```

Where d_action might include:
- End-effector position deltas (3D)
- Gripper command (open/close)
- Joint angle targets

### 3.5 Training Objectives

**Vision-Language Loss**:
```
L_VL = -Σ_t log P(word_t | vision, language_<t)
```

**Action Prediction Loss**:
```
L_action = ||action_pred - action_gt||²₂
```

**Multi-Task Loss**:
```
L_total = λ_VL L_VL + λ_action L_action + λ_LM L_LM
```

where L_LM is standard language modeling loss to maintain LLM capabilities.

### 3.6 Action Tokenization

Actions can be represented as:

**Continuous Actions**: Direct regression
```
a_t = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper] ∈ R^7
```

**Discretized Actions**: Tokenized like language
```
a_t = TokenizeAction(continuous_action)
P(a_t) = Softmax(LM(X)_t W_action_vocab)
```

## 4. High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│               PaLM-E Architecture                           │
└────────────────────────────────────────────────────────────┘

Inputs: Images, Robot State, Language Instructions

    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │  RGB    │  │  Depth  │  │ Robot   │
    │ Camera  │  │ Camera  │  │  State  │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         └──────┬─────┴────────────┘
                │
        ┌───────▼────────┐
        │  Multi-Sensor  │
        │    Encoder     │
        └───────┬────────┘
                │
       Sensor Features ∈ R^(N_s × d)
                │
        ┌───────▼────────┐
        │  Projection    │
        │  to LM Space   │
        └───────┬────────┘
                │
       S_proj ∈ R^(N_s × d_lm)
                │
                ├────────────────────┐
                │                    │
                ▼                    ▼
        ┌───────────┐        ┌──────────────┐
        │  Language │        │Vision Encoder│
        │  Tokens   │        │    (ViT)     │
        └──────┬────┘        └──────┬───────┘
               │                    │
               │             V ∈ R^(N_v × d_v)
               │                    │
               │             ┌──────▼───────┐
               │             │ V-L Project  │
               │             └──────┬───────┘
               │                    │
               │            V_proj ∈ R^(N_v × d_lm)
               │                    │
               └────────┬───────────┘
                        │
              Concat: [V_proj; S_proj; L_embed]
                        │
                ┌───────▼────────┐
                │   PaLM-E LLM   │
                │  (540B params) │
                │                │
                │  Transformer   │
                │    Layers      │
                └───────┬────────┘
                        │
              Hidden States ∈ R^(N_total × d_lm)
                        │
        ┌───────────────┴────────────────┐
        │                                │
        ▼                                ▼
┌───────────────┐              ┌─────────────────┐
│  Language     │              │   Action        │
│  Generation   │              │   Prediction    │
│   Head        │              │   Head          │
└───────┬───────┘              └────────┬────────┘
        │                               │
  Text Output                    Robot Actions
  "I'll pick                     [Δx, Δy, Δz,
   up the cup"                    gripper_cmd]
```

## 5. Implementation Details

### 5.1 PaLM-E Core Implementation

Reference: `Nexus/nexus/models/multimodal/palm_e.py`

```python
import torch
import torch.nn as nn
from nexus.core.base import NexusModule

class PaLME(NexusModule):
    def __init__(self, config):
        super().__init__(config)
        
        self.hidden_dim = config["hidden_dim"]
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(
            input_channels=config.get("input_channels", 3),
            hidden_dim=self.hidden_dim
        )
        
        # Language encoder
        self.language_encoder = LanguageEncoder(
            vocab_size=config["vocab_size"],
            hidden_dim=self.hidden_dim
        )
        
        # Cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.get("num_heads", 8)
        )
        
        # Task-specific output heads
        self.task_heads = nn.ModuleDict({
            'language_generation': nn.Linear(self.hidden_dim, config["vocab_size"]),
            'action_prediction': nn.Linear(self.hidden_dim, config.get("num_actions", 7))
        })
    
    def forward(self, images=None, text_tokens=None):
        outputs = {}
        
        if images is not None:
            visual_features = self.vision_encoder(images)
            outputs["visual_features"] = visual_features
        
        if text_tokens is not None:
            text_features = self.language_encoder(text_tokens)
            outputs["text_features"] = text_features
        
        if images is not None and text_tokens is not None:
            fused = self.cross_attention(visual_features, text_features, text_features)[0]
            outputs["fused_features"] = fused
            outputs["language_logits"] = self.task_heads["language_generation"](fused)
            outputs["action_logits"] = self.task_heads["action_prediction"](fused)
        
        return outputs
```

### 5.2 Action Prediction Head

```python
class ActionPredictionHead(nn.Module):
    def __init__(self, hidden_dim, action_dim=7, action_type='continuous'):
        super().__init__()
        self.action_type = action_type

        if action_type == 'continuous':
            # Continuous action prediction
            self.action_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, action_dim)
            )
        elif action_type == 'discrete':
            # Discrete action tokenization
            self.action_predictor = nn.Linear(hidden_dim, action_dim * 256)  # 256 bins per dimension

    def forward(self, features):
        if self.action_type == 'continuous':
            continuous_action = self.action_predictor(features)
            # Apply tanh to bound actions to [-1, 1]
            return torch.tanh(continuous_action)
        else:
            logits = self.action_predictor(features)
            # Reshape to [batch, action_dim, num_bins]
            logits = logits.view(-1, self.action_dim, 256)
            return logits
```

### 5.3 Multi-Sensor State Encoder

```python
class RobotStateEncoder(nn.Module):
    """Encode proprioceptive and sensor state."""

    def __init__(self, config):
        super().__init__()

        # Joint position encoder
        self.joint_encoder = nn.Sequential(
            nn.Linear(config['num_joints'], 128),
            nn.ReLU(),
            nn.Linear(128, config['hidden_dim'])
        )

        # End-effector pose encoder
        self.ee_encoder = nn.Sequential(
            nn.Linear(7, 64),  # xyz + quaternion
            nn.ReLU(),
            nn.Linear(64, config['hidden_dim'])
        )

        # Force/torque encoder (if available)
        if config.get('has_force_sensor', False):
            self.force_encoder = nn.Sequential(
                nn.Linear(6, 32),  # 3D force + 3D torque
                nn.ReLU(),
                nn.Linear(32, config['hidden_dim'])
            )

        # Fusion layer
        num_modalities = 2 + (1 if config.get('has_force_sensor', False) else 0)
        self.fusion = nn.Linear(num_modalities * config['hidden_dim'], config['hidden_dim'])

    def forward(self, state_dict):
        """
        Args:
            state_dict: Dictionary with keys:
                - 'joint_positions': [B, num_joints]
                - 'ee_pose': [B, 7]
                - 'force_torque': [B, 6] (optional)
        Returns:
            state_embedding: [B, hidden_dim]
        """
        encodings = []

        # Encode joint positions
        joint_emb = self.joint_encoder(state_dict['joint_positions'])
        encodings.append(joint_emb)

        # Encode end-effector pose
        ee_emb = self.ee_encoder(state_dict['ee_pose'])
        encodings.append(ee_emb)

        # Encode force/torque if available
        if 'force_torque' in state_dict and hasattr(self, 'force_encoder'):
            force_emb = self.force_encoder(state_dict['force_torque'])
            encodings.append(force_emb)

        # Fuse all modalities
        combined = torch.cat(encodings, dim=-1)
        state_embedding = self.fusion(combined)

        return state_embedding

### 5.4 Vision Encoder for Robotics

```python
class RoboticVisionEncoder(nn.Module):
    """Multi-view RGB-D vision encoder for robotics."""

    def __init__(self, config):
        super().__init__()

        # RGB encoder
        from torchvision.models import resnet50
        self.rgb_encoder = resnet50(pretrained=True)
        self.rgb_encoder.fc = nn.Identity()  # Remove classification head

        # Depth encoder (smaller network)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Projection to common space
        self.rgb_proj = nn.Linear(2048, config['hidden_dim'])
        self.depth_proj = nn.Linear(128, config['hidden_dim'])

        # Multi-view fusion
        self.view_attention = nn.MultiheadAttention(
            config['hidden_dim'],
            num_heads=8,
            batch_first=True
        )

    def forward(self, images_dict):
        """
        Args:
            images_dict: Dictionary with keys:
                - 'rgb': [B, num_views, 3, H, W]
                - 'depth': [B, num_views, 1, H, W]
        Returns:
            vision_features: [B, num_views, hidden_dim]
        """
        B, num_views = images_dict['rgb'].shape[:2]

        # Flatten batch and views
        rgb = images_dict['rgb'].view(B * num_views, 3, -1, -1)
        depth = images_dict['depth'].view(B * num_views, 1, -1, -1)

        # Encode RGB and depth
        rgb_feats = self.rgb_encoder(rgb)  # [B*V, 2048]
        depth_feats = self.depth_encoder(depth)  # [B*V, 128]

        # Project to common space
        rgb_emb = self.rgb_proj(rgb_feats)
        depth_emb = self.depth_proj(depth_feats)

        # Fuse RGB and depth
        fused = rgb_emb + depth_emb  # [B*V, hidden_dim]

        # Reshape to multi-view
        fused = fused.view(B, num_views, -1)

        # Apply cross-view attention
        attended, _ = self.view_attention(fused, fused, fused)

        return attended
```

## 6. Code Walkthrough

### Step 1: Initialize Model

```python
from nexus.models.multimodal.palm_e import PaLME

config = {
    "hidden_dim": 768,
    "num_layers": 12,
    "vocab_size": 32000,
    "num_actions": 7
}

model = PaLME(config)
```

### Step 2: Prepare Inputs

```python
import torch
from PIL import Image

# Robot camera image
image = Image.open("robot_view.jpg")
image_tensor = preprocess(image).unsqueeze(0)

# Language instruction
instruction = "Pick up the red cup"
text_tokens = tokenizer.encode(instruction, return_tensors='pt')
```

### Step 3: Predict Action

```python
with torch.no_grad():
    outputs = model(images=image_tensor, text_tokens=text_tokens)
    predicted_action = outputs['action_logits'].squeeze(0)
    print(f"Predicted action: {predicted_action}")
```

## 7. Optimization Tricks

### 7.1 Action Smoothing

Smooth noisy action predictions for stable robot control:

```python
class ActionSmoother:
    def __init__(self, alpha=0.7, history_length=5):
        self.alpha = alpha
        self.history_length = history_length
        self.action_history = []

    def smooth(self, action):
        """Exponential moving average smoothing."""
        self.action_history.append(action.clone())

        if len(self.action_history) > self.history_length:
            self.action_history.pop(0)

        if len(self.action_history) == 1:
            return action

        # Exponential moving average
        smoothed = self.action_history[0] * (1 - self.alpha)
        for i, hist_action in enumerate(self.action_history[1:], 1):
            weight = self.alpha * ((1 - self.alpha) ** i)
            smoothed = smoothed + hist_action * weight

        return smoothed

class KalmanActionFilter:
    """More sophisticated filtering for action smoothing."""

    def __init__(self, action_dim, process_noise=0.01, measurement_noise=0.1):
        self.action_dim = action_dim
        self.Q = torch.eye(action_dim) * process_noise  # Process noise
        self.R = torch.eye(action_dim) * measurement_noise  # Measurement noise
        self.P = torch.eye(action_dim)  # Error covariance
        self.x = None  # State estimate

    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return measurement

        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred @ torch.inverse(P_pred + self.R)  # Kalman gain
        self.x = x_pred + K @ (measurement - x_pred)
        self.P = (torch.eye(self.action_dim) - K) @ P_pred

        return self.x
```

### 7.2 Sim-to-Real Transfer

Domain randomization and adaptation techniques:

```python
class DomainAdapter:
    """Domain randomization for sim-to-real transfer."""

    def __init__(self, config):
        self.texture_randomizer = TextureRandomizer(
            color_jitter=config.get('color_jitter', 0.3),
            brightness_range=(0.7, 1.3)
        )
        self.lighting_randomizer = LightingRandomizer(
            intensity_range=(0.5, 1.5),
            direction_variance=0.2
        )
        self.camera_randomizer = CameraRandomizer(
            fov_range=(50, 70),
            position_noise=0.05
        )

    def adapt_observation(self, sim_obs):
        # Apply random transformations
        obs = self.texture_randomizer(sim_obs)
        obs = self.lighting_randomizer(obs)
        obs = self.camera_randomizer(obs)

        # Add sensor noise
        obs = obs + torch.randn_like(obs) * 0.02

        return obs

class DomainAdversarialAdapter(nn.Module):
    """Adversarial domain adaptation for sim-to-real."""

    def __init__(self, feature_dim):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # sim vs real
        )

    def forward(self, features, alpha=1.0):
        # Gradient reversal layer
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return domain_logits

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
```

### 7.3 Hierarchical Policy Structure

Decompose long-horizon tasks into subtasks:

```python
class HierarchicalPolicy:
    """High-level planner + low-level controller."""

    def __init__(self, high_level_model, low_level_model):
        self.high_level = high_level_model  # PaLM-E for goal generation
        self.low_level = low_level_model  # Fine-tuned for specific skills

        self.current_subgoal = None
        self.subgoal_threshold = 0.1

    def execute(self, observation, instruction):
        # High-level: Generate subgoal from language
        if self.current_subgoal is None or self.is_subgoal_achieved():
            self.current_subgoal = self.high_level.generate_subgoal(
                observation, instruction
            )

        # Low-level: Execute action to reach subgoal
        action = self.low_level.predict_action(observation, self.current_subgoal)

        return action

    def is_subgoal_achieved(self):
        # Check if current subgoal is reached
        return self.compute_subgoal_distance() < self.subgoal_threshold
```

### 7.4 Model Distillation for Deployment

Compress large model for real-time robot control:

```python
class PolicyDistiller:
    """Distill PaLM-E into smaller student model."""

    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = 3.0

    def distill(self, dataloader, num_epochs=10):
        """Knowledge distillation training loop."""

        optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            for batch in dataloader:
                # Teacher predictions (frozen)
                with torch.no_grad():
                    teacher_out = self.teacher(
                        images=batch['images'],
                        text_tokens=batch['text']
                    )
                    teacher_actions = teacher_out['action_logits']

                # Student predictions
                student_out = self.student(
                    images=batch['images'],
                    text_tokens=batch['text']
                )
                student_actions = student_out['action_logits']

                # Distillation loss (KL divergence)
                loss_distill = F.kl_div(
                    F.log_softmax(student_actions / self.temperature, dim=-1),
                    F.softmax(teacher_actions / self.temperature, dim=-1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)

                # Hard label loss
                loss_hard = F.mse_loss(student_actions, batch['actions'])

                # Combined loss
                loss = 0.7 * loss_distill + 0.3 * loss_hard

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.student
```

### 7.5 Online Adaptation

Fine-tune policy during deployment:

```python
class OnlineAdapter:
    """Adapt policy based on execution feedback."""

    def __init__(self, model, buffer_size=1000):
        self.model = model
        self.buffer = ReplayBuffer(buffer_size)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    def collect_experience(self, obs, action, reward, next_obs):
        """Store experience in replay buffer."""
        self.buffer.add(obs, action, reward, next_obs)

    def adapt(self, batch_size=32):
        """Perform online adaptation step."""

        if len(self.buffer) < batch_size:
            return

        # Sample batch from buffer
        batch = self.buffer.sample(batch_size)

        # Compute loss (behavioral cloning + reward)
        outputs = self.model(images=batch['obs'], text_tokens=batch['text'])
        predicted_actions = outputs['action_logits']

        # Weighted by reward
        weights = torch.sigmoid(batch['rewards'])
        loss = (weights * F.mse_loss(
            predicted_actions,
            batch['actions'],
            reduction='none'
        )).mean()

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 7.6 Multi-Task Batching

Efficiently train on diverse robotics tasks:

```python
class MultiTaskBatcher:
    """Sample balanced batches across tasks."""

    def __init__(self, task_datasets, batch_size=32):
        self.task_datasets = task_datasets
        self.batch_size = batch_size
        self.task_weights = {task: 1.0 for task in task_datasets.keys()}

    def get_batch(self):
        """Sample batch with balanced task representation."""

        batch = {
            'images': [],
            'text': [],
            'actions': [],
            'task_ids': []
        }

        # Samples per task
        samples_per_task = max(1, self.batch_size // len(self.task_datasets))

        for task_id, dataset in self.task_datasets.items():
            # Sample from this task
            indices = torch.randint(0, len(dataset), (samples_per_task,))

            for idx in indices:
                sample = dataset[idx]
                batch['images'].append(sample['image'])
                batch['text'].append(sample['instruction'])
                batch['actions'].append(sample['action'])
                batch['task_ids'].append(task_id)

        # Collate
        batch['images'] = torch.stack(batch['images'])
        batch['actions'] = torch.stack(batch['actions'])
        batch['task_ids'] = torch.tensor(batch['task_ids'])

        return batch
```

## 8. Experiments & Results

### 8.1 Robot Manipulation Tasks

| Task | PaLM-E | RT-1 | BC-Z |
|------|--------|------|------|
| Pick & Place | 89.2% | 85.1% | 72.3% |
| Drawer Opening | 76.8% | 71.5% | 58.9% |
| Object Rearrangement | 71.2% | 64.3% | 52.7% |

### 8.2 Vision-Language Benchmarks

| Benchmark | PaLM-E-562B | Flamingo |
|-----------|-------------|----------|
| VQAv2 | 84.3 | 82.0 |
| OK-VQA | 66.1 | 57.4 |

### 8.3 Language Capability Retention

| Benchmark | PaLM-E | PaLM (base) |
|-----------|--------|-------------|
| MMLU | 69.3 | 70.7 |
| HellaSwag | 85.1 | 86.2 |

## 9. Common Pitfalls

### 9.1 Sim-to-Real Gap

```python
# Wrong: Use simulation parameters directly
action = model.predict(sim_obs)

# Correct: Apply domain adaptation
adapted_obs = domain_adapter.sim_to_real(sim_obs)
action = model.predict(adapted_obs)
```

### 9.2 Action Space Mismatch

```python
# Wrong: No action clipping
predicted_action = model.predict_action(obs)

# Correct: Clip to valid range
clipped_action = torch.clamp(
    predicted_action,
    min=robot.action_space.low,
    max=robot.action_space.high
)
```

### 9.3 Safety Constraints

```python
# Wrong: Execute all predictions
action = model.predict(obs)
robot.execute(action)

# Correct: Safety checking
if safety_checker.is_safe(action, current_state):
    robot.execute(action)
else:
    safe_action = safety_controller.get_safe_action(current_state)
    robot.execute(safe_action)
```

## 10. References

### Papers

1. **PaLM-E: An Embodied Multimodal Language Model**
   - https://arxiv.org/abs/2303.03378
   - Google Research, 2023

2. **RT-1: Robotics Transformer**
   - https://arxiv.org/abs/2212.06817

3. **Say-Can: Grounding Language in Robotic Affordances**
   - https://arxiv.org/abs/2204.01691

4. **PaLM: Scaling Language Modeling**
   - https://arxiv.org/abs/2204.02311

### Resources

- Implementation: `Nexus/nexus/models/multimodal/palm_e.py`
- Google Research Blog: https://ai.googleblog.com/

### Related Models

- **NVLM**: `Nexus/nexus/models/multimodal/nvlm.py`
- **HiViLT**: `Nexus/nexus/models/multimodal/hivilt.py`
- **Qwen2-VL**: `Nexus/nexus/models/multimodal/qwen2_vl.py`

### Robotics Platforms

- **Franka Emika Panda**: Research manipulator arm
- **UR5/UR10**: Universal Robots collaborative robots
- **MuJoCo**: Physics simulation engine
- **PyBullet**: Open-source physics simulator

### Benchmarks

- **RLBench**: Robot learning benchmark with 100+ tasks
- **Meta-World**: Multi-task manipulation benchmark
- **CALVIN**: Long-horizon language-conditioned tasks

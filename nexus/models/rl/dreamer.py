"""
DreamerV3: Mastering Diverse Domains through World Models
Paper: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)

DreamerV3 is a model-based reinforcement learning algorithm that:
- Learns a world model (RSSM) with discrete categorical latent representations
- Uses the world model to generate imagined trajectories (latent rollouts)
- Trains an actor-critic entirely in imagination, without interacting with the
  real environment during policy optimization
- Employs symlog transforms for prediction targets to handle varying scales
- Uses an EMA target network for the critic to stabilize training
- Achieves strong performance across diverse domains (Atari, DMC, Minecraft, etc.)

Key components:
- RSSM (Recurrent State Space Model): combines deterministic recurrent state with
  stochastic discrete categorical latent variables
- World model: encoder, dynamics (prior), posterior, reward/continue/decoder heads
- Actor-Critic: policy and value networks trained on imagined latent trajectories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic transform: sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RSSM(NexusModule):
    """
    Recurrent State Space Model with discrete categorical latent variables.

    The RSSM maintains a combined state consisting of:
    - Deterministic recurrent state h_t (GRU hidden state)
    - Stochastic state z_t (discrete categorical, num_categories^stoch_dim)

    The model supports:
    - Prior (dynamics): p(z_t | h_t) - predicts stochastic state from deterministic
    - Posterior: q(z_t | h_t, o_t) - infers stochastic state with observations
    - Sequence model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})

    Args:
        obs_dim: Dimension of observation input
        action_dim: Dimension of action space
        hidden_dim: Dimension of the deterministic recurrent state
        stoch_dim: Number of categorical distributions in stochastic state
        num_categories: Number of categories per categorical distribution
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        stoch_dim: int = 32,
        num_categories: int = 32,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.num_categories = num_categories
        self.stoch_state_size = stoch_dim * num_categories

        # Sequence model: produces next deterministic state h_t from h_{t-1}, z_{t-1}, a_{t-1}
        self.sequence_input = nn.Linear(
            self.stoch_state_size + action_dim, hidden_dim
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Prior (dynamics predictor): p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_state_size),
        )

        # Posterior: q(z_t | h_t, o_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_state_size),
        )

    def initial_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get initial RSSM state (zeros).

        Args:
            batch_size: Number of parallel environments/episodes

        Returns:
            Dictionary with 'deter' and 'stoch' tensors
        """
        device = next(self.parameters()).device
        return {
            "deter": torch.zeros(batch_size, self.hidden_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_state_size, device=device),
        }

    def observe_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Single observation step: compute prior and posterior.

        Args:
            prev_state: Previous RSSM state dict
            prev_action: Previous action
            obs: Current observation

        Returns:
            Tuple of (new_state, prior_logits, posterior_logits)
        """
        # Sequence model: compute new deterministic state
        x = self.sequence_input(
            torch.cat([prev_state["stoch"], prev_action], dim=-1)
        )
        x = F.silu(x)
        deter = self.gru(x, prev_state["deter"])

        # Prior: p(z_t | h_t)
        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.view(-1, self.stoch_dim, self.num_categories)

        # Posterior: q(z_t | h_t, o_t)
        posterior_logits = self.posterior_net(torch.cat([deter, obs], dim=-1))
        posterior_logits = posterior_logits.view(-1, self.stoch_dim, self.num_categories)

        # Sample stochastic state from posterior using straight-through
        stoch = self._sample_categorical(posterior_logits)

        new_state = {
            "deter": deter,
            "stoch": stoch.view(-1, self.stoch_state_size),
        }

        return new_state, prior_logits, posterior_logits

    def imagine_step(
        self,
        prev_state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Single imagination step: compute prior only (no observation).

        Args:
            prev_state: Previous RSSM state dict
            prev_action: Previous action

        Returns:
            Tuple of (new_state, prior_logits)
        """
        # Sequence model
        x = self.sequence_input(
            torch.cat([prev_state["stoch"], prev_action], dim=-1)
        )
        x = F.silu(x)
        deter = self.gru(x, prev_state["deter"])

        # Prior only
        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.view(-1, self.stoch_dim, self.num_categories)

        stoch = self._sample_categorical(prior_logits)

        new_state = {
            "deter": deter,
            "stoch": stoch.view(-1, self.stoch_state_size),
        }

        return new_state, prior_logits

    def get_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate deterministic and stochastic state into a feature vector."""
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    @property
    def feature_dim(self) -> int:
        """Total dimension of the RSSM feature vector."""
        return self.hidden_dim + self.stoch_state_size

    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from categorical distribution with straight-through gradients.

        Args:
            logits: Unnormalized logits [batch, stoch_dim, num_categories]

        Returns:
            One-hot sampled tensor [batch, stoch_dim, num_categories]
        """
        dist = torch.distributions.OneHotCategorical(logits=logits)
        sample = dist.sample()
        # Straight-through estimator: gradients pass through as if identity
        probs = dist.probs
        return sample + probs - probs.detach()


class WorldModel(NexusModule):
    """
    DreamerV3 World Model.

    Contains the RSSM and all prediction heads:
    - Encoder: maps observations to feature space for the posterior
    - Reward predictor: predicts reward from latent state (using symlog)
    - Continue predictor: predicts episode continuation probability
    - Decoder: reconstructs observations from latent state

    Args:
        config: Configuration dictionary with:
            - obs_dim: Observation dimension
            - action_dim: Action dimension
            - hidden_dim: RSSM deterministic state dimension (default: 512)
            - stoch_dim: Number of categorical distributions (default: 32)
            - num_categories: Categories per distribution (default: 32)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        obs_dim = config["obs_dim"]
        action_dim = config["action_dim"]
        hidden_dim = config.get("hidden_dim", 512)
        stoch_dim = config.get("stoch_dim", 32)
        num_categories = config.get("num_categories", 32)

        # RSSM core
        self.rssm = RSSM(obs_dim, action_dim, hidden_dim, stoch_dim, num_categories)

        feature_dim = self.rssm.feature_dim

        # Encoder: maps observations to a representation for the posterior
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward predictor (symlog transformed targets)
        self.reward_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Continue predictor (episode continuation probability)
        self.continue_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Decoder: reconstructs observations from latent state
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        prev_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the world model over a sequence.

        Args:
            observations: Observation sequence [batch, seq_len, obs_dim]
            actions: Action sequence [batch, seq_len, action_dim]
            prev_state: Initial RSSM state (or None for zeros)

        Returns:
            Dictionary with predictions and latent states
        """
        batch_size, seq_len, _ = observations.shape

        if prev_state is None:
            prev_state = self.rssm.initial_state(batch_size)

        # Encode observations
        encoded_obs = self.encoder(observations)

        # Roll through sequence
        prior_logits_seq = []
        posterior_logits_seq = []
        features_seq = []
        states = []

        state = prev_state
        for t in range(seq_len):
            action_t = actions[:, t]
            obs_t = encoded_obs[:, t]

            state, prior_logits, posterior_logits = self.rssm.observe_step(
                state, action_t, obs_t
            )

            features = self.rssm.get_features(state)
            prior_logits_seq.append(prior_logits)
            posterior_logits_seq.append(posterior_logits)
            features_seq.append(features)
            states.append(state)

        # Stack sequences
        features_seq = torch.stack(features_seq, dim=1)
        prior_logits_seq = torch.stack(prior_logits_seq, dim=1)
        posterior_logits_seq = torch.stack(posterior_logits_seq, dim=1)

        # Predictions from features
        reward_pred = self.reward_predictor(features_seq)
        continue_pred = self.continue_predictor(features_seq)
        obs_pred = self.decoder(features_seq)

        return {
            "features": features_seq,
            "prior_logits": prior_logits_seq,
            "posterior_logits": posterior_logits_seq,
            "reward_pred": reward_pred,
            "continue_pred": continue_pred,
            "obs_pred": obs_pred,
            "final_state": state,
        }


class DreamerActorCritic(NexusModule):
    """
    Actor-Critic for DreamerV3, trained entirely in imagination.

    The actor outputs a policy distribution over actions conditioned on the
    RSSM latent features. The critic estimates the value of latent states
    with an EMA target network for stability.

    Args:
        feature_dim: Dimension of RSSM features (deter + stoch)
        action_dim: Dimension of the action space
        hidden_dim: Hidden layer size
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()

        self.action_dim = action_dim

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # EMA target critic
        self.target_critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad = False

    def get_action_distribution(
        self, features: torch.Tensor
    ) -> torch.distributions.Normal:
        """
        Get the policy distribution for the given features.

        Args:
            features: RSSM feature vector [batch, feature_dim]

        Returns:
            Normal distribution over actions
        """
        x = self.actor(features)
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)

    def get_value(self, features: torch.Tensor) -> torch.Tensor:
        """Compute value estimate from latent features."""
        return self.critic(features)

    def get_target_value(self, features: torch.Tensor) -> torch.Tensor:
        """Compute target value estimate from latent features."""
        with torch.no_grad():
            return self.target_critic(features)

    def update_target(self, ema_decay: float = 0.98):
        """Update target critic with exponential moving average."""
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(
                ema_decay * target_param.data + (1 - ema_decay) * param.data
            )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing policy and value.

        Args:
            features: RSSM feature vector

        Returns:
            Dictionary with action, log_prob, and value
        """
        dist = self.get_action_distribution(features)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = self.critic(features)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
        }


class DreamerAgent(NexusModule):
    """
    DreamerV3 Agent orchestrating world model learning and actor-critic
    training in imagination.

    The agent:
    1. Learns a world model from real environment interactions
    2. Uses the world model to imagine trajectories in latent space
    3. Trains the actor-critic on these imagined trajectories
    4. Uses symlog transforms for reward and value predictions

    Args:
        config: Configuration dictionary with:
            - obs_dim: Observation dimension
            - action_dim: Action dimension
            - hidden_dim: RSSM deterministic state dimension (default: 512)
            - stoch_dim: Number of categorical distributions (default: 32)
            - num_categories: Categories per distribution (default: 32)
            - imagination_horizon: Length of imagined rollouts (default: 15)
            - gamma: Discount factor (default: 0.997)
            - lambda_: GAE lambda for imagination (default: 0.95)
            - model_lr: World model learning rate (default: 1e-4)
            - actor_lr: Actor learning rate (default: 3e-5)
            - critic_lr: Critic learning rate (default: 3e-5)
            - ema_decay: EMA decay for target critic (default: 0.98)
            - free_nats: Free nats for KL balancing (default: 1.0)
            - kl_balance: Balance between prior/posterior KL (default: 0.8)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 512)
        self.imagination_horizon = config.get("imagination_horizon", 15)
        self.gamma = config.get("gamma", 0.997)
        self.lambda_ = config.get("lambda_", 0.95)
        self.ema_decay = config.get("ema_decay", 0.98)
        self.free_nats = config.get("free_nats", 1.0)
        self.kl_balance = config.get("kl_balance", 0.8)

        # World model
        self.world_model = WorldModel(config)

        # Actor-Critic
        feature_dim = self.world_model.rssm.feature_dim
        self.actor_critic = DreamerActorCritic(
            feature_dim, self.action_dim, self.hidden_dim
        )

        # Optimizers
        self.model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=config.get("model_lr", 1e-4),
            eps=1e-8,
        )
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor_critic.actor.parameters()) +
            list(self.actor_critic.actor_mean.parameters()) +
            list(self.actor_critic.actor_log_std.parameters()),
            lr=config.get("actor_lr", 3e-5),
            eps=1e-8,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.actor_critic.critic.parameters(),
            lr=config.get("critic_lr", 3e-5),
            eps=1e-8,
        )

        # Track RSSM state for online interaction
        self._current_state = None

    def select_action(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select an action given an observation using the world model + actor.

        Args:
            observation: Current observation
            deterministic: If True, use the mean action

        Returns:
            Action as a numpy array
        """
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                obs = torch.FloatTensor(observation).unsqueeze(0)
            else:
                obs = observation.unsqueeze(0) if observation.dim() == 1 else observation

            device = next(self.parameters()).device
            obs = obs.to(device)

            # Initialize state if needed
            if self._current_state is None:
                self._current_state = self.world_model.rssm.initial_state(1)
                self._prev_action = torch.zeros(1, self.action_dim, device=device)

            # Observe step to update latent state
            encoded = self.world_model.encoder(obs)
            self._current_state, _, _ = self.world_model.rssm.observe_step(
                self._current_state, self._prev_action, encoded
            )

            # Get features and sample action
            features = self.world_model.rssm.get_features(self._current_state)
            dist = self.actor_critic.get_action_distribution(features)

            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()

            self._prev_action = action
            return action.cpu().numpy()[0]

    def reset_state(self):
        """Reset the internal RSSM state (call at episode start)."""
        self._current_state = None

    def _compute_world_model_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
        """
        Compute world model loss: reconstruction + reward + continue + KL.

        Args:
            observations: [batch, seq_len, obs_dim]
            actions: [batch, seq_len, action_dim]
            rewards: [batch, seq_len, 1]
            continues: [batch, seq_len, 1] (1 - dones)

        Returns:
            Tuple of (total_loss, metrics, model_outputs)
        """
        outputs = self.world_model(observations, actions)

        # Reconstruction loss (symlog space)
        obs_loss = F.mse_loss(
            outputs["obs_pred"], symlog(observations)
        )

        # Reward prediction loss (symlog targets)
        reward_loss = F.mse_loss(
            outputs["reward_pred"], symlog(rewards)
        )

        # Continue prediction loss (binary cross-entropy)
        continue_loss = F.binary_cross_entropy_with_logits(
            outputs["continue_pred"], continues
        )

        # KL divergence with free nats and balancing
        prior_logits = outputs["prior_logits"]
        posterior_logits = outputs["posterior_logits"]

        prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)
        posterior_dist = torch.distributions.OneHotCategorical(logits=posterior_logits)

        # KL(posterior || prior) with free nats
        kl_post = torch.distributions.kl_divergence(posterior_dist, prior_dist)
        kl_post = kl_post.sum(dim=-1)  # sum over stoch_dim
        kl_post = torch.clamp(kl_post, min=self.free_nats).mean()

        # KL(prior || posterior) for balancing
        kl_prior = torch.distributions.kl_divergence(prior_dist, posterior_dist)
        kl_prior = kl_prior.sum(dim=-1)
        kl_prior = torch.clamp(kl_prior, min=self.free_nats).mean()

        # Balanced KL loss
        kl_loss = self.kl_balance * kl_post + (1 - self.kl_balance) * kl_prior

        total_loss = obs_loss + reward_loss + continue_loss + kl_loss

        metrics = {
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item(),
            "model_loss": total_loss.item(),
        }

        return total_loss, metrics, outputs

    def _imagine_trajectory(
        self, start_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory from a starting latent state.

        Uses the actor to select actions and the world model dynamics
        to predict next states, without any real environment interaction.

        Args:
            start_state: Starting RSSM state

        Returns:
            Dictionary with imagined features, actions, rewards, continues
        """
        state = start_state
        features_list = []
        actions_list = []
        reward_preds = []
        continue_preds = []

        for _ in range(self.imagination_horizon):
            features = self.world_model.rssm.get_features(state)
            features_list.append(features)

            # Actor selects action
            dist = self.actor_critic.get_action_distribution(features)
            action = dist.rsample()
            actions_list.append(action)

            # Imagine next state
            state, _ = self.world_model.rssm.imagine_step(state, action)

            # Predict reward and continue
            next_features = self.world_model.rssm.get_features(state)
            reward_pred = self.world_model.reward_predictor(next_features)
            continue_pred = torch.sigmoid(
                self.world_model.continue_predictor(next_features)
            )
            reward_preds.append(reward_pred)
            continue_preds.append(continue_pred)

        # Final features for bootstrapping
        features_list.append(self.world_model.rssm.get_features(state))

        return {
            "features": torch.stack(features_list, dim=1),       # [B, H+1, F]
            "actions": torch.stack(actions_list, dim=1),          # [B, H, A]
            "rewards": torch.stack(reward_preds, dim=1),          # [B, H, 1]
            "continues": torch.stack(continue_preds, dim=1),      # [B, H, 1]
        }

    def _compute_actor_loss(
        self, start_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute actor loss by imagining trajectories and maximizing returns.

        The actor loss flows gradients through the imagined trajectory:
        actor -> action -> dynamics -> rewards -> loss.
        This allows the actor to be trained to maximize imagined returns.

        Args:
            start_state: Starting RSSM state (detached from world model)

        Returns:
            Tuple of (actor_loss, metrics)
        """
        imagined = self._imagine_trajectory(start_state)
        features = imagined["features"]
        rewards = symexp(imagined["rewards"].squeeze(-1))
        continues = imagined["continues"].squeeze(-1)

        # Use target critic for value bootstrapping (no actor gradients through critic)
        with torch.no_grad():
            target_values = self.actor_critic.get_target_value(features).squeeze(-1)

        # Compute lambda-returns (these carry gradients through actions -> dynamics -> rewards)
        lambda_returns = torch.zeros_like(rewards)
        last_return = target_values[:, -1]

        for t in reversed(range(self.imagination_horizon)):
            cont = continues[:, t]
            reward = rewards[:, t]
            next_val = target_values[:, t + 1]

            td_target = reward + self.gamma * cont * next_val
            last_return = (1 - self.lambda_) * td_target + self.lambda_ * (
                reward + self.gamma * cont * last_return
            )
            lambda_returns[:, t] = last_return

        # Actor loss: maximize lambda-returns (symlog for scale invariance)
        # Weighted by discount to prioritize near-term rewards
        weights = torch.cumprod(
            torch.cat([torch.ones_like(continues[:, :1]), continues[:, :-1]], dim=1),
            dim=1,
        )
        actor_loss = -(weights * symlog(lambda_returns)).mean()

        metrics = {
            "actor_loss": actor_loss.item(),
            "imagined_reward": rewards.mean().item(),
            "lambda_return": lambda_returns.mean().item(),
        }

        return actor_loss, metrics

    def _compute_critic_loss(
        self, start_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute critic loss from imagined trajectories.

        The critic is trained to predict lambda-returns computed from the
        EMA target critic (no actor gradients needed).

        Args:
            start_state: Starting RSSM state (detached from world model)

        Returns:
            Tuple of (critic_loss, metrics)
        """
        with torch.no_grad():
            imagined = self._imagine_trajectory(start_state)
            features = imagined["features"]
            rewards = symexp(imagined["rewards"].squeeze(-1))
            continues = imagined["continues"].squeeze(-1)
            target_values = self.actor_critic.get_target_value(features).squeeze(-1)

            # Compute lambda-return targets
            lambda_returns = torch.zeros_like(rewards)
            last_return = target_values[:, -1]

            for t in reversed(range(self.imagination_horizon)):
                cont = continues[:, t]
                reward = rewards[:, t]
                next_val = target_values[:, t + 1]

                td_target = reward + self.gamma * cont * next_val
                last_return = (1 - self.lambda_) * td_target + self.lambda_ * (
                    reward + self.gamma * cont * last_return
                )
                lambda_returns[:, t] = last_return

            critic_targets = symlog(lambda_returns)
            # Features for critic input (exclude last bootstrap step)
            critic_features = features[:, :-1].clone()

        # Critic predicts lambda-returns in symlog space
        critic_values = self.actor_critic.get_value(critic_features).squeeze(-1)
        critic_loss = F.mse_loss(critic_values, critic_targets)

        metrics = {
            "critic_loss": critic_loss.item(),
            "imagined_value": critic_values.mean().item(),
        }

        return critic_loss, metrics

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Full DreamerV3 update: world model + imagination-based actor-critic.

        Args:
            batch: Dictionary containing:
                - observations: [batch, seq_len, obs_dim]
                - actions: [batch, seq_len, action_dim]
                - rewards: [batch, seq_len, 1]
                - dones: [batch, seq_len, 1]

        Returns:
            Dictionary with all loss metrics
        """
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        continues = 1.0 - batch["dones"]

        # --- Update world model ---
        model_loss, model_metrics, model_outputs = self._compute_world_model_loss(
            observations, actions, rewards, continues
        )

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.model_optimizer.step()

        # --- Imagination-based actor-critic training ---
        # Detach starting states from world model graph
        with torch.no_grad():
            start_features = model_outputs["features"]
            batch_size, seq_len, feat_dim = start_features.shape
            flat_features = start_features.reshape(-1, feat_dim)

            # Reconstruct RSSM states from features
            deter_dim = self.world_model.rssm.hidden_dim
            start_state = {
                "deter": flat_features[:, :deter_dim].clone(),
                "stoch": flat_features[:, deter_dim:].clone(),
            }

        # Update actor: imagine trajectories with gradient flow through actions
        actor_loss, actor_metrics = self._compute_actor_loss(start_state)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor_critic.actor.parameters()) +
            list(self.actor_critic.actor_mean.parameters()) +
            list(self.actor_critic.actor_log_std.parameters()),
            100.0,
        )
        self.actor_optimizer.step()

        # Update critic: predict lambda-returns (separate imagination, no actor grad)
        critic_loss, critic_metrics = self._compute_critic_loss(start_state)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), 100.0)
        self.critic_optimizer.step()

        # Update EMA target critic
        self.actor_critic.update_target(self.ema_decay)

        # Combine all metrics
        metrics = {**model_metrics, **actor_metrics, **critic_metrics}
        return metrics

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass through the full agent (world model + actor-critic).

        Args:
            observations: [batch, seq_len, obs_dim]
            actions: [batch, seq_len, action_dim]

        Returns:
            Dictionary with world model outputs and actor-critic predictions
        """
        wm_outputs = self.world_model(observations, actions)
        features = wm_outputs["features"]

        dist = self.actor_critic.get_action_distribution(features[:, -1])
        value = self.actor_critic.get_value(features[:, -1])

        return {
            **wm_outputs,
            "policy_mean": dist.mean,
            "policy_std": dist.stddev,
            "value": value,
        }

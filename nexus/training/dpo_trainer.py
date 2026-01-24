from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from ..core.base import NexusModule
from ..models.rl.preference.reward_model import EnhancedRewardModel
from ..utils.logging import Logger


class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization (DPO) trainer.

    Implements DPO training with optional reward model integration.
    Inherits from BaseTrainer for optimizer setup, checkpointing,
    and training loop infrastructure.
    """

    def __init__(
        self,
        model: NexusModule,
        reference_model: Optional[NexusModule] = None,
        optimizer: str = "adamw",
        learning_rate: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[Logger] = None,
        checkpoint_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the DPO trainer.

        Args:
            model: The policy model to train.
            reference_model: Optional frozen reference model for KL computation.
            optimizer: Optimizer name ('adam', 'sgd', 'adamw').
            learning_rate: Learning rate for the optimizer.
            device: Device to train on ('cuda' or 'cpu').
            logger: Optional logger instance for logging.
            checkpoint_dir: Directory for saving checkpoints.
            config: Configuration dictionary with DPO-specific parameters:
                - beta: DPO temperature parameter (default: 0.1)
                - reference_free: Whether to use reference-free DPO (default: False)
                - kl_weight: Weight for KL divergence loss (default: 1.0)
                - reward_weight: Weight for reward model loss (default: 0.1)
                - use_reward_model: Whether to use reward model (default: True)
        """
        config = config or {}

        # Initialize base trainer (handles optimizer, device, checkpointing)
        super().__init__(
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            device=device,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            config=config
        )

        # Reference model for computing reference log probabilities
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.to(self.device)
            # Set reference model to evaluation mode
            self.reference_model.train(False)
            # Freeze reference model
            for param in self.reference_model.parameters():
                param.requires_grad = False

        # DPO-specific parameters
        self.beta = config.get("beta", 0.1)
        self.reference_free = config.get("reference_free", False)

        # Loss weights
        self.kl_weight = config.get("kl_weight", 1.0)
        self.reward_weight = config.get("reward_weight", 0.1)

        # Optional reward model component
        self.use_reward_model = config.get("use_reward_model", True)
        if self.use_reward_model:
            self.reward_model = EnhancedRewardModel(config)
            self.reward_model.to(self.device)

    def _validate_trainer_config(self) -> None:
        """Validate DPO trainer configuration using ConfigValidatorMixin."""
        super()._validate_trainer_config()

        self.validate_positive(self.config.get("beta", 0.1), "beta")

        if not self.config.get("reference_free", False) and self.reference_model is None:
            self.logger.warning(
                "reference_model is None but reference_free is False. "
                "DPO will use reference-free mode."
            )

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None,
        reward_chosen: Optional[torch.Tensor] = None,
        reward_rejected: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute the DPO loss.

        Args:
            policy_chosen_logps: Log probs of chosen responses from policy.
            policy_rejected_logps: Log probs of rejected responses from policy.
            reference_chosen_logps: Log probs of chosen responses from reference.
            reference_rejected_logps: Log probs of rejected responses from reference.
            reward_chosen: Optional reward scores for chosen responses.
            reward_rejected: Optional reward scores for rejected responses.
            attention_mask: Optional attention mask for loss masking.

        Returns:
            Dictionary containing loss components and total loss.
        """
        # Compute policy losses
        if self.reference_free or reference_chosen_logps is None:
            policy_loss = -F.logsigmoid(policy_chosen_logps - policy_rejected_logps)
        else:
            chosen_diff = policy_chosen_logps - reference_chosen_logps
            rejected_diff = policy_rejected_logps - reference_rejected_logps
            policy_loss = -F.logsigmoid(self.beta * (chosen_diff - rejected_diff))

        if attention_mask is not None:
            policy_loss = policy_loss * attention_mask

        policy_loss = policy_loss.mean()

        losses = {"policy_loss": policy_loss}

        # Add reward losses if available
        if reward_chosen is not None and reward_rejected is not None:
            reward_loss = F.mse_loss(
                reward_chosen - reward_rejected,
                torch.ones_like(reward_chosen)
            )
            losses["reward_loss"] = reward_loss
            losses["total_loss"] = (
                self.kl_weight * policy_loss +
                self.reward_weight * reward_loss
            )
        else:
            losses["total_loss"] = policy_loss

        return losses

    def _compute_log_probs(
        self,
        model: NexusModule,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute sequence log probabilities from a model.

        Args:
            model: The model to compute log probs from.
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.
            labels: Optional labels for computing per-token log probs.

        Returns:
            Log probabilities tensor.
        """
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        if labels is None:
            labels = input_ids

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Sum over sequence length
        if attention_mask is not None:
            mask = attention_mask[..., 1:].contiguous()
            per_token_logps = per_token_logps * mask
            return per_token_logps.sum(dim=-1)

        return per_token_logps.sum(dim=-1)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, float]:
        """Perform a single DPO training step.

        Args:
            batch: Dictionary containing:
                - chosen_input_ids: Input IDs for chosen responses
                - rejected_input_ids: Input IDs for rejected responses
                - chosen_attention_mask: Attention mask for chosen
                - rejected_attention_mask: Attention mask for rejected
            batch_idx: Index of the current batch.

        Returns:
            Dictionary of metrics including 'loss'.
        """
        self.optimizer.zero_grad()

        # Extract batch components
        chosen_input_ids = batch["chosen_input_ids"]
        rejected_input_ids = batch["rejected_input_ids"]
        chosen_attention_mask = batch.get("chosen_attention_mask")
        rejected_attention_mask = batch.get("rejected_attention_mask")

        # Compute policy log probabilities
        policy_chosen_logps = self._compute_log_probs(
            self.model, chosen_input_ids, chosen_attention_mask
        )
        policy_rejected_logps = self._compute_log_probs(
            self.model, rejected_input_ids, rejected_attention_mask
        )

        # Compute reference log probabilities
        reference_chosen_logps = None
        reference_rejected_logps = None
        if self.reference_model is not None and not self.reference_free:
            with torch.no_grad():
                reference_chosen_logps = self._compute_log_probs(
                    self.reference_model, chosen_input_ids, chosen_attention_mask
                )
                reference_rejected_logps = self._compute_log_probs(
                    self.reference_model, rejected_input_ids, rejected_attention_mask
                )

        # Compute optional reward model scores
        reward_chosen = None
        reward_rejected = None
        if self.use_reward_model and hasattr(self, 'reward_model'):
            with torch.no_grad():
                reward_chosen = self.reward_model(chosen_input_ids, chosen_attention_mask)
                reward_rejected = self.reward_model(rejected_input_ids, rejected_attention_mask)

        # Compute DPO loss
        losses = self.compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            reward_chosen=reward_chosen,
            reward_rejected=reward_rejected
        )

        total_loss = losses["total_loss"]
        total_loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Return metrics
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": losses["policy_loss"].item()
        }
        if "reward_loss" in losses:
            metrics["reward_loss"] = losses["reward_loss"].item()

        return metrics

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, float]:
        """Perform a single DPO validation step.

        Args:
            batch: Dictionary containing preference pairs.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary of metrics.
        """
        # Extract batch components
        chosen_input_ids = batch["chosen_input_ids"]
        rejected_input_ids = batch["rejected_input_ids"]
        chosen_attention_mask = batch.get("chosen_attention_mask")
        rejected_attention_mask = batch.get("rejected_attention_mask")

        # Compute policy log probabilities
        policy_chosen_logps = self._compute_log_probs(
            self.model, chosen_input_ids, chosen_attention_mask
        )
        policy_rejected_logps = self._compute_log_probs(
            self.model, rejected_input_ids, rejected_attention_mask
        )

        # Compute reference log probabilities
        reference_chosen_logps = None
        reference_rejected_logps = None
        if self.reference_model is not None and not self.reference_free:
            reference_chosen_logps = self._compute_log_probs(
                self.reference_model, chosen_input_ids, chosen_attention_mask
            )
            reference_rejected_logps = self._compute_log_probs(
                self.reference_model, rejected_input_ids, rejected_attention_mask
            )

        # Compute DPO loss
        losses = self.compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps
        )

        # Compute accuracy: how often policy prefers chosen over rejected
        chosen_rewards = policy_chosen_logps
        rejected_rewards = policy_rejected_logps
        if reference_chosen_logps is not None:
            chosen_rewards = policy_chosen_logps - reference_chosen_logps
            rejected_rewards = policy_rejected_logps - reference_rejected_logps

        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return {
            "loss": losses["total_loss"].item(),
            "policy_loss": losses["policy_loss"].item(),
            "accuracy": accuracy.item() * 100,  # Percentage
            "correct": (chosen_rewards > rejected_rewards).sum().item(),
            "total": chosen_input_ids.size(0)
        }

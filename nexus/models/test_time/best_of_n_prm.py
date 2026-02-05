"""
Best-of-N Sampling with Process Reward Model (PRM).

Generates N candidate solutions and selects the best one using a Process Reward
Model that provides step-by-step verification. Unlike outcome reward models (ORMs)
that only evaluate final answers, PRMs verify each reasoning step, enabling better
selection of reasoning traces.

Key innovations:
- Step-level verification: PRMs evaluate each intermediate step
- Process supervision: catches errors early in reasoning chains
- Better sample selection: identifies correct reasoning even with wrong final answer
- Scalable inference: combines cheap sampling with learned verification
- Works for mathematical reasoning, code generation, planning, etc.

Components:
1. Generator: Samples N candidate solutions (often with chain-of-thought)
2. Process Reward Model (PRM): Evaluates each step in the solution
3. Aggregation: Selects best candidate based on step-level scores
4. Optional: Beam search with PRM-guided pruning

Paper: "Let's Verify Step by Step"
       Lightman et al., OpenAI 2023
       https://arxiv.org/abs/2305.20050

Key insight: Step-by-step verification is more effective than final-answer
verification for complex multi-step reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from nexus.core.base import NexusModule


class ProcessRewardModel(NexusModule):
    """Process Reward Model for step-by-step verification.

    Assigns a reward/correctness score to each step in a reasoning trace.

    Args:
        config: Configuration dictionary with keys:
            - embed_dim (int): Embedding dimension. Default 512
            - hidden_dim (int): Hidden dimension. Default 256
            - num_layers (int): Number of transformer layers. Default 3
            - reward_aggregation (str): How to aggregate step rewards.
              Options: 'mean', 'min', 'product', 'last'. Default 'product'
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embed_dim = config.get('embed_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 3)
        self.reward_aggregation = config.get('reward_aggregation', 'product')

        # Step encoder (processes each step representation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.step_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Step-level reward head
        self.reward_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()  # Reward in [0, 1] per step
        )

    def forward(self,
                step_embeddings: torch.Tensor,
                step_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute step-level and trajectory-level rewards.

        Args:
            step_embeddings: Step representations (B, num_steps, embed_dim)
            step_mask: Valid step mask (B, num_steps). True = valid

        Returns:
            Dictionary containing:
                - step_rewards: Reward per step (B, num_steps)
                - trajectory_reward: Aggregate reward for full trace (B,)
                - encoded_steps: Contextual step encodings (B, num_steps, embed_dim)
        """
        # Encode steps with context
        if step_mask is not None:
            # Convert mask to attention mask (False = attend, True = ignore)
            attn_mask = ~step_mask
        else:
            attn_mask = None

        encoded_steps = self.step_encoder(step_embeddings, src_key_padding_mask=attn_mask)

        # Compute step-level rewards
        step_rewards = self.reward_head(encoded_steps).squeeze(-1)  # (B, num_steps)

        # Mask invalid steps
        if step_mask is not None:
            step_rewards = step_rewards * step_mask

        # Aggregate to trajectory-level reward
        if self.reward_aggregation == 'mean':
            # Average over valid steps
            if step_mask is not None:
                valid_counts = step_mask.sum(dim=1).clamp(min=1)
                trajectory_reward = (step_rewards * step_mask).sum(dim=1) / valid_counts
            else:
                trajectory_reward = step_rewards.mean(dim=1)

        elif self.reward_aggregation == 'min':
            # Minimum (most pessimistic)
            if step_mask is not None:
                step_rewards_masked = step_rewards + (1 - step_mask) * 1e9  # Add large value to invalid
                trajectory_reward = step_rewards_masked.min(dim=1)[0]
            else:
                trajectory_reward = step_rewards.min(dim=1)[0]

        elif self.reward_aggregation == 'product':
            # Product (all steps must be correct)
            if step_mask is not None:
                # Only multiply valid steps
                log_rewards = torch.log(step_rewards + 1e-8) * step_mask
                trajectory_reward = torch.exp(log_rewards.sum(dim=1) / step_mask.sum(dim=1).clamp(min=1))
            else:
                log_rewards = torch.log(step_rewards + 1e-8)
                trajectory_reward = torch.exp(log_rewards.mean(dim=1))

        elif self.reward_aggregation == 'last':
            # Only last step (for tasks where only final answer matters)
            if step_mask is not None:
                # Get last valid step
                last_indices = step_mask.sum(dim=1) - 1
                trajectory_reward = step_rewards[torch.arange(len(step_rewards)), last_indices.long()]
            else:
                trajectory_reward = step_rewards[:, -1]
        else:
            trajectory_reward = step_rewards.mean(dim=1)

        return {
            'step_rewards': step_rewards,
            'trajectory_reward': trajectory_reward,
            'encoded_steps': encoded_steps
        }


class BestOfNSelector(NexusModule):
    """Selects best candidate from N samples using PRM.

    Args:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prm = ProcessRewardModel(config)

    def select_best(self,
                   candidates: List[Dict[str, torch.Tensor]],
                   return_scores: bool = False) -> Tuple[int, Optional[torch.Tensor]]:
        """Select best candidate using PRM scores.

        Args:
            candidates: List of candidate solutions, each a dict with:
                - step_embeddings: Step representations (num_steps, embed_dim)
                - step_mask: Optional valid step mask (num_steps,)
            return_scores: If True, return all scores

        Returns:
            Tuple of:
                - best_idx: Index of best candidate
                - scores: Optional tensor of all scores (N,) if return_scores=True
        """
        scores = []

        for candidate in candidates:
            step_embeds = candidate['step_embeddings'].unsqueeze(0)  # (1, num_steps, embed_dim)
            step_mask = candidate.get('step_mask', None)
            if step_mask is not None:
                step_mask = step_mask.unsqueeze(0)

            with torch.no_grad():
                output = self.prm(step_embeds, step_mask)
                score = output['trajectory_reward'][0]
                scores.append(score)

        scores_tensor = torch.stack(scores)
        best_idx = scores_tensor.argmax().item()

        if return_scores:
            return best_idx, scores_tensor
        else:
            return best_idx, None


class BestOfNWithPRM(NexusModule):
    """Complete Best-of-N sampling system with PRM.

    Args:
        config: Configuration dictionary with keys:
            - embed_dim (int): Embedding dimension. Default 512
            - num_samples (int): Number of candidates to sample. Default 8
            - temperature (float): Sampling temperature. Default 0.8
            - reward_aggregation (str): PRM aggregation method
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_samples = config.get('num_samples', 8)
        self.temperature = config.get('temperature', 0.8)

        # PRM for verification
        self.prm = ProcessRewardModel(config)

        # Selector
        self.selector = BestOfNSelector(config)

    def generate_candidates(self,
                           model: nn.Module,
                           input_ids: torch.Tensor,
                           max_length: int = 512,
                           step_separator_token: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
        """Generate N candidate solutions.

        Args:
            model: Base generation model
            input_ids: Input prompt (1, seq_len)
            max_length: Maximum generation length
            step_separator_token: Token that separates reasoning steps

        Returns:
            List of candidate solutions with step embeddings
        """
        candidates = []

        for _ in range(self.num_samples):
            # Generate with sampling
            # This is a simplified version - actual implementation depends on model
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=self.temperature,
                    do_sample=True
                )

            # Extract step embeddings from generated sequence
            # This assumes model can provide hidden states
            if hasattr(model, 'get_hidden_states'):
                hidden_states = model.get_hidden_states(generated)

                # Split into steps if separator token provided
                if step_separator_token is not None:
                    step_indices = (generated == step_separator_token).nonzero(as_tuple=True)[1]

                    if len(step_indices) > 0:
                        # Extract embeddings at step boundaries
                        step_embeds = hidden_states[0, step_indices, :]
                        step_mask = torch.ones(len(step_indices), dtype=torch.bool)
                    else:
                        # No explicit steps, treat whole generation as one step
                        step_embeds = hidden_states[0, -1:, :]
                        step_mask = torch.ones(1, dtype=torch.bool)
                else:
                    # Use all token embeddings as steps
                    step_embeds = hidden_states[0]
                    step_mask = torch.ones(len(step_embeds), dtype=torch.bool)

                candidates.append({
                    'generated': generated,
                    'step_embeddings': step_embeds,
                    'step_mask': step_mask
                })

        return candidates

    def forward(self,
                model: nn.Module,
                input_ids: torch.Tensor,
                max_length: int = 512,
                step_separator_token: Optional[int] = None) -> Dict[str, Any]:
        """Generate N candidates and select best using PRM.

        Args:
            model: Generation model
            input_ids: Input prompt (1, seq_len)
            max_length: Maximum generation length
            step_separator_token: Token separating steps

        Returns:
            Dictionary containing:
                - best_candidate: Selected best solution
                - best_idx: Index of best candidate
                - all_candidates: All generated candidates
                - scores: PRM scores for all candidates
        """
        # Generate N candidates
        candidates = self.generate_candidates(
            model, input_ids, max_length, step_separator_token
        )

        # Select best using PRM
        best_idx, scores = self.selector.select_best(candidates, return_scores=True)

        return {
            'best_candidate': candidates[best_idx],
            'best_idx': best_idx,
            'all_candidates': candidates,
            'scores': scores
        }

    def train_prm(self,
                  step_embeddings: torch.Tensor,
                  step_labels: torch.Tensor,
                  step_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Train PRM on labeled reasoning traces.

        Args:
            step_embeddings: Step representations (B, num_steps, embed_dim)
            step_labels: Ground truth labels per step (B, num_steps)
                        1 = correct step, 0 = incorrect step
            step_mask: Valid step mask (B, num_steps)

        Returns:
            Training loss
        """
        output = self.prm(step_embeddings, step_mask)
        step_rewards = output['step_rewards']

        # Binary cross-entropy loss per step
        if step_mask is not None:
            loss = F.binary_cross_entropy(
                step_rewards[step_mask],
                step_labels[step_mask]
            )
        else:
            loss = F.binary_cross_entropy(step_rewards, step_labels)

        return loss


class BeamSearchWithPRM(NexusModule):
    """Beam search guided by PRM for efficient generation.

    Instead of generating N full candidates, incrementally build solutions
    with PRM-guided pruning.

    Args:
        config: Configuration dictionary with keys:
            - beam_width (int): Number of beams to maintain. Default 4
            - prune_threshold (float): Prune if step reward below threshold
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.beam_width = config.get('beam_width', 4)
        self.prune_threshold = config.get('prune_threshold', 0.3)

        self.prm = ProcessRewardModel(config)

    def beam_search(self,
                   model: nn.Module,
                   input_ids: torch.Tensor,
                   max_steps: int = 20) -> Dict[str, Any]:
        """Perform PRM-guided beam search.

        Args:
            model: Generation model
            input_ids: Input prompt (1, seq_len)
            max_steps: Maximum reasoning steps

        Returns:
            Best solution found via beam search
        """
        # Initialize beams
        beams = [{
            'sequence': input_ids,
            'step_embeddings': [],
            'score': 1.0
        }]

        for step in range(max_steps):
            candidates = []

            # Expand each beam
            for beam in beams:
                # Generate next step (simplified)
                # In practice, sample multiple continuations per beam
                with torch.no_grad():
                    next_token_logits = model(beam['sequence'])[:, -1, :]
                    top_tokens = torch.topk(next_token_logits, k=self.beam_width)

                for token_idx in range(self.beam_width):
                    new_seq = torch.cat([
                        beam['sequence'],
                        top_tokens.indices[:, token_idx:token_idx+1]
                    ], dim=1)

                    # Get embedding for new step
                    if hasattr(model, 'get_hidden_states'):
                        hidden = model.get_hidden_states(new_seq)[0, -1, :]
                    else:
                        hidden = torch.randn(512)  # Placeholder

                    new_step_embeds = beam['step_embeddings'] + [hidden]

                    # Score with PRM
                    if len(new_step_embeds) > 0:
                        step_tensor = torch.stack(new_step_embeds).unsqueeze(0)
                        prm_output = self.prm(step_tensor)
                        step_score = prm_output['step_rewards'][0, -1].item()

                        # Prune if step score too low
                        if step_score < self.prune_threshold:
                            continue

                        # Update beam score
                        new_score = beam['score'] * step_score
                    else:
                        new_score = beam['score']

                    candidates.append({
                        'sequence': new_seq,
                        'step_embeddings': new_step_embeds,
                        'score': new_score
                    })

            # Keep top beams
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = candidates[:self.beam_width]

            # Check if done (e.g., all beams ended)
            if not beams:
                break

        # Return best beam
        return beams[0] if beams else None


__all__ = [
    'ProcessRewardModel',
    'BestOfNSelector',
    'BestOfNWithPRM',
    'BeamSearchWithPRM'
]

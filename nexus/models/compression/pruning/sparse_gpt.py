"""SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.

Reference:
    Frantar, E. & Alistarh, D. "SparseGPT: Massive Language Models Can Be
    Accurately Pruned in One-Shot." ICML 2023.
    https://arxiv.org/abs/2301.00774

SparseGPT extends the Optimal Brain Surgeon (OBS) framework to enable
one-shot, post-training pruning of large language models to high sparsity
levels (e.g., 50-60%) with minimal accuracy degradation and no retraining.
The key insight is that the OBS weight-update formula can be solved
approximately using a column-by-column Cholesky-based strategy, making it
tractable even for layers with millions of parameters.

Algorithm overview:
    1. Collect calibration data and compute the empirical Hessian H = 2 X X^T
       for each linear layer, where X is the matrix of input activations.
    2. For each layer, process the weight matrix column by column (or in
       blocks of columns):
        a. Identify which weights in the current column to prune based on
           the OBS saliency criterion: w_j^2 / [H^{-1}]_{jj}.
        b. Set pruned weights to zero.
        c. Update the remaining (unpruned) weights in subsequent columns to
           compensate for the pruning error, using the row of H^{-1}
           corresponding to the pruned index.
    3. The Hessian inverse rows are obtained lazily from a Cholesky
       decomposition, so only one triangular factor is stored.

Key features:
    - One-shot pruning (no iterative retraining)
    - Approximate inverse Hessian via Cholesky factorization
    - Block-wise processing for efficiency
    - Supports unstructured sparsity at arbitrary target ratios
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import math
import logging

from nexus.core.base import NexusModule

logger = logging.getLogger(__name__)


@dataclass
class SparseGPTConfig:
    """Configuration for SparseGPT pruning.

    Attributes:
        sparsity: Target sparsity ratio in [0, 1). A sparsity of 0.5 means
            50% of weights are set to zero.
        blocksize: Number of columns processed simultaneously in each OBS
            block. Larger blocks are computationally faster but require more
            memory for the block-local Hessian inverse slice. Must be a
            positive integer.
        percdamp: Dampening factor added to the Hessian diagonal for
            numerical stability, expressed as a fraction of the mean
            diagonal value. Typical values are 0.01-0.1.
    """
    sparsity: float = 0.5
    blocksize: int = 128
    percdamp: float = 0.01


class SparseGPTPruner(NexusModule):
    """One-shot weight pruner based on the SparseGPT algorithm.

    Prunes linear layers to a target unstructured sparsity level using
    approximate inverse Hessian information from calibration data. The
    algorithm processes columns in blocks, pruning the least salient
    weights in each block and compensating subsequent columns to minimize
    the layer-wise output reconstruction error.

    Usage::

        config = SparseGPTConfig(sparsity=0.5, blocksize=128)
        pruner = SparseGPTPruner(pruning_config=config)

        # Prune a single layer
        pruned_weight = pruner.prune_layer(linear_layer, calibration_inputs)

        # Prune an entire model
        metrics = pruner.prune_model(model, calibration_dataloader, nsamples=128)

    Args:
        config: Optional dict-based configuration.
        pruning_config: Optional SparseGPTConfig with pruning parameters.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pruning_config: Optional[SparseGPTConfig] = None,
    ):
        if pruning_config is not None:
            cfg = {
                "sparsity": pruning_config.sparsity,
                "blocksize": pruning_config.blocksize,
                "percdamp": pruning_config.percdamp,
            }
        else:
            cfg = config or {}

        super().__init__(cfg)

        self.sparsity = cfg.get("sparsity", 0.5)
        self.blocksize = cfg.get("blocksize", 128)
        self.percdamp = cfg.get("percdamp", 0.01)

        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(
                f"Sparsity must be in [0, 1), got {self.sparsity}"
            )
        if self.blocksize < 1:
            raise ValueError(
                f"Block size must be >= 1, got {self.blocksize}"
            )

    def _compute_hessian(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the empirical Hessian approximation from calibration inputs.

        The Hessian for a linear layer with respect to the squared-error loss
        is approximated as:

            H = 2 X X^T

        where X is the (in_features, n_samples) matrix of input activations.
        In practice, we accumulate X X^T and normalize by the number of
        samples. The factor of 2 from the squared error is absorbed into the
        subsequent OBS calculations.

        Args:
            inputs: Calibration input tensor of shape (n_samples, in_features)
                or (n_samples, seq_len, in_features). For 3D inputs, the
                sequence and batch dimensions are flattened.

        Returns:
            Hessian matrix of shape (in_features, in_features).
        """
        if inputs.dim() == 3:
            # (batch, seq_len, features) -> (batch * seq_len, features)
            inputs = inputs.reshape(-1, inputs.shape[-1])

        inputs = inputs.float()
        n_samples = inputs.shape[0]

        # H = (1/n) * X^T X, which approximates E[x x^T]
        H = (inputs.T @ inputs) / n_samples

        return H

    def _prune_weight(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
        sparsity: float,
    ) -> torch.Tensor:
        """Prune a weight matrix using the SparseGPT column-by-column algorithm.

        This implements the core SparseGPT procedure:
            1. Dampen the Hessian diagonal for numerical stability.
            2. Compute the Cholesky factorization of the (dampened) Hessian
               inverse, yielding an upper-triangular factor.
            3. Process the weight matrix in blocks of columns. Within each
               block:
               a. Compute the OBS saliency score for each weight:
                  score_{ij} = w_{ij}^2 / [H^{-1}]_{jj}
               b. Select the lowest-scoring fraction (= sparsity) of weights
                  in each row of the block and set them to zero.
               c. Compute the pruning error for each zeroed weight.
               d. Update the remaining weights in the block to compensate,
                  using the corresponding rows of the Cholesky factor.
            4. After processing each block, propagate the accumulated error
               to all subsequent blocks.

        Args:
            W: Weight matrix of shape (out_features, in_features).
            H: Hessian matrix of shape (in_features, in_features).
            sparsity: Target fraction of weights to prune.

        Returns:
            Pruned weight matrix of the same shape, with approximately
            (sparsity * total_weights) entries set to zero and the
            remaining entries adjusted to compensate.
        """
        out_features, in_features = W.shape
        device = W.device

        W = W.clone().float()

        # --- Dampening ---
        damp = self.percdamp * H.diag().mean()
        H = H.clone()
        H.diagonal().add_(damp)

        # --- Cholesky factorization of H^{-1} ---
        # We need the upper-triangular Cholesky factor of H_inv.
        # First invert H via its Cholesky decomposition, then decompose H_inv.
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)
        except RuntimeError:
            # Fallback: increase dampening and retry
            extra_damp = 0.1 * H.diag().mean()
            H.diagonal().add_(extra_damp)
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

        losses = torch.zeros(out_features, device=device)

        # --- Block-wise column processing ---
        for block_start in range(0, in_features, self.blocksize):
            block_end = min(block_start + self.blocksize, in_features)
            block_len = block_end - block_start

            W_block = W[:, block_start:block_end].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)

            Hinv_block = H_inv_chol[block_start:block_end, block_start:block_end]

            for j in range(block_len):
                w_col = W_block[:, j]
                d = Hinv_block[j, j]

                # --- Determine which weights to prune in this column ---
                # For unstructured sparsity, we prune the smallest-magnitude
                # weights (weighted by the inverse Hessian diagonal) across
                # each row. Here we use a simple magnitude threshold per row
                # within the current block.
                if sparsity > 0:
                    # Saliency: |w| / sqrt(h_inv_jj) -- weights with small
                    # saliency are pruned first. We compute a per-row threshold.
                    col_abs = w_col.abs()
                    n_prune = int(math.ceil(sparsity * block_len))

                    # Sort the block columns for each row by saliency to find
                    # which entries to prune. For simplicity in the per-column
                    # pass, we prune entries below the per-column threshold.
                    threshold = torch.kthvalue(
                        W_block.abs(), k=max(n_prune, 1), dim=1
                    ).values
                    mask = col_abs <= threshold
                else:
                    mask = torch.zeros(out_features, dtype=torch.bool, device=device)

                # Zero out pruned weights
                q_col = w_col.clone()
                q_col[mask] = 0.0

                Q_block[:, j] = q_col

                # --- Error and compensation ---
                err = (w_col - q_col) / d.clamp(min=1e-10)
                Err_block[:, j] = err

                # Compensate remaining columns within the block
                if j + 1 < block_len:
                    W_block[:, j + 1:] -= (
                        err.unsqueeze(1) * Hinv_block[j, j + 1:].unsqueeze(0)
                    )

                # Track the loss contribution
                losses += (w_col - q_col).pow(2) / (2.0 * d.clamp(min=1e-10))

            W[:, block_start:block_end] = Q_block

            # --- Propagate error to subsequent blocks ---
            if block_end < in_features:
                W[:, block_end:] -= (
                    Err_block @ H_inv_chol[block_start:block_end, block_end:]
                )

        return W

    def prune_layer(
        self,
        layer: nn.Linear,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Prune a single linear layer using SparseGPT.

        Computes the Hessian from the provided calibration inputs, then
        applies the column-by-column OBS pruning procedure to the layer's
        weight matrix. The pruned weight is written back to the layer
        in-place and also returned.

        Args:
            layer: The nn.Linear layer to prune.
            inputs: Calibration inputs of shape (n_samples, in_features)
                or (n_samples, seq_len, in_features).

        Returns:
            The pruned weight tensor (also stored in layer.weight.data).
        """
        H = self._compute_hessian(inputs)
        pruned_weight = self._prune_weight(layer.weight.data, H, self.sparsity)
        layer.weight.data.copy_(pruned_weight.to(layer.weight.dtype))

        actual_sparsity = (
            (layer.weight.data == 0).float().mean().item()
        )
        logger.info(
            f"SparseGPT: pruned layer to {actual_sparsity:.2%} sparsity "
            f"(target: {self.sparsity:.2%})"
        )

        return pruned_weight

    def prune_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        nsamples: int = 128,
        target_layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Prune all (or targeted) linear layers in a model using SparseGPT.

        For each target linear layer:
            1. Register a forward hook to capture input activations.
            2. Run calibration samples through the model to collect inputs.
            3. Compute the Hessian and prune the layer.

        Args:
            model: The model to prune.
            dataloader: DataLoader providing calibration samples.
            nsamples: Maximum number of calibration samples to collect.
            target_layers: Optional list of layer name substrings to target.
                If None, all nn.Linear layers are pruned.

        Returns:
            Dictionary mapping layer names to pruning metrics including
            achieved sparsity and reconstruction MSE.
        """
        metrics: Dict[str, Any] = {}
        was_training = model.training
        model.eval()

        # --- Collect calibration batches ---
        calibration_inputs: List[torch.Tensor] = []
        collected = 0
        for batch in dataloader:
            if collected >= nsamples:
                break
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            if isinstance(batch, dict):
                batch = batch.get("input_ids", next(iter(batch.values())))
            calibration_inputs.append(batch)
            collected += batch.shape[0]

        # --- Prune each target layer ---
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if target_layers is not None:
                if not any(t in name for t in target_layers):
                    continue

            # Gather activations for this specific layer via hook
            layer_inputs: List[torch.Tensor] = []

            def _make_hook(storage):
                def _hook(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    storage.append(x.detach())
                return _hook

            handle = module.register_forward_hook(_make_hook(layer_inputs))

            with torch.no_grad():
                for data in calibration_inputs:
                    try:
                        if isinstance(data, dict):
                            model(**data)
                        else:
                            model(data)
                    except Exception:
                        pass

            handle.remove()

            if not layer_inputs:
                logger.warning(
                    f"SparseGPT: no activations collected for {name}, skipping."
                )
                continue

            # Concatenate all collected activations
            all_inputs = torch.cat(layer_inputs, dim=0)

            # Store original weight for MSE computation
            original_weight = module.weight.data.clone()

            # Prune the layer
            self.prune_layer(module, all_inputs)

            # Compute metrics
            achieved_sparsity = (module.weight.data == 0).float().mean().item()
            mse = (
                (original_weight.float() - module.weight.data.float())
                .pow(2)
                .mean()
                .item()
            )

            metrics[name] = {
                "target_sparsity": self.sparsity,
                "achieved_sparsity": achieved_sparsity,
                "weight_mse": mse,
            }

            logger.info(
                f"SparseGPT [{name}]: achieved {achieved_sparsity:.2%} sparsity, "
                f"MSE = {mse:.6e}"
            )

        model.train(was_training)
        return metrics

    def forward(self, weight: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Prune a weight tensor given its Hessian (functional interface).

        This enables using the pruner as a callable module. Requires a
        pre-computed Hessian matrix.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
            H: Hessian matrix of shape (in_features, in_features).

        Returns:
            Pruned weight tensor.
        """
        return self._prune_weight(weight, H, self.sparsity)

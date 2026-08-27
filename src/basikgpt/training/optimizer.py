"""AdamW optimizer construction with explicit parameter grouping for weight decay."""

import torch
import torch.nn as nn
from basikgpt.training.config import TrainingConfig


def configure_optimizers(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    """Configures the AdamW optimizer with decoupled weight decay parameter groups.

    Parameter Grouping Policy:
        - Decay (weight_decay = config.weight_decay):
          All 2D+ learnable weight tensors, including:
          * Linear projection matrices (attn.qkv_proj, attn.out_proj, mlp.fc_in, mlp.fc_out)
          * Embedding tables (wte, wpe)
        - No Decay (weight_decay = 0.0):
          All 1D learnable tensors, including:
          * LayerNorm affine scale and bias parameters (ln_1, ln_2, ln_f)
          * Linear projection additive biases (bias)

    Weight Tying Invariant:
        In basikGPT, `model.lm_head.weight` is tied to `model.wte.weight` (same object).
        Parameters are deduplicated by object identity (`id(p)`), ensuring the tied weight
        is registered in the optimizer exactly once.

    Args:
        model: The PyTorch neural network module (e.g. basikGPT.GPT).
        config: TrainingConfig instance containing optimization hyperparameters.

    Returns:
        Configured torch.optim.AdamW instance with 2 parameter groups.

    Raises:
        ValueError: If any trainable parameter is unassigned or duplicated.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen_param_ids: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_id = id(param)
        if param_id in seen_param_ids:
            # Skip tied or duplicated parameter references (e.g. lm_head.weight is wte.weight)
            continue
        seen_param_ids.add(param_id)

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    # Invariant checks: Disjointness and completeness
    decay_set = set(id(p) for p in decay_params)
    no_decay_set = set(id(p) for p in no_decay_params)

    if not decay_set.isdisjoint(no_decay_set):
        raise ValueError("Decay and no-decay parameter groups have overlapping parameters!")

    all_unique_trainable = set(id(p) for p in model.parameters() if p.requires_grad)
    combined_set = decay_set | no_decay_set
    if combined_set != all_unique_trainable:
        missing_count = len(all_unique_trainable - combined_set)
        raise ValueError(f"Parameter grouping coverage mismatch: {missing_count} parameters unassigned!")

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
    return optimizer

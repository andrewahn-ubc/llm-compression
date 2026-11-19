"""Losses shared by probe, local-recovery, and global KD stages."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F


def _expanded_mask(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.shape == reference.shape:
        return mask.to(device=reference.device, dtype=reference.dtype)
    if mask.shape != reference.shape[:-1]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must equal {tuple(reference.shape[:-1])} "
            f"or {tuple(reference.shape)}"
        )
    return mask.to(device=reference.device, dtype=reference.dtype).unsqueeze(-1)


def relative_error(
    prediction: Tensor,
    target: Tensor,
    teacher_input: Tensor,
    *,
    attention_mask: Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Return squared reconstruction error relative to the teacher span's change.

    This computes ``sum(||prediction-target||^2) / sum(||target-teacher_input||^2)``
    over all selected tokens and hidden dimensions. Accumulation is in float32 for
    stable fp16/bf16 training; gradients still flow to ``prediction``.
    """

    if prediction.shape != target.shape or target.shape != teacher_input.shape:
        raise ValueError(
            "prediction, target, and teacher_input must have identical shapes; got "
            f"{tuple(prediction.shape)}, {tuple(target.shape)}, {tuple(teacher_input.shape)}"
        )
    if prediction.ndim < 1:
        raise ValueError("relative_error requires at least one dimension")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    residual_sq = (prediction.float() - target.float()).square()
    change_sq = (target.float() - teacher_input.float()).square()
    if attention_mask is not None:
        mask = _expanded_mask(attention_mask, residual_sq)
        residual_sq = residual_sq * mask
        change_sq = change_sq * mask
    return residual_sq.sum() / change_sq.sum().clamp_min(eps)


def mixed_local_loss(
    teacher_input_prediction: Tensor,
    student_input_prediction: Tensor,
    target: Tensor,
    teacher_input: Tensor,
    lam: float | Tensor,
    *,
    attention_mask: Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Mix local reconstruction on clean teacher and actual student inputs."""

    if isinstance(lam, Tensor):
        if lam.numel() != 1:
            raise ValueError("lam tensor must be scalar")
        if not bool(torch.isfinite(lam.detach()).item()):
            raise ValueError("lam must be finite")
        numeric_lam = float(lam.detach().item())
        weight = lam.to(device=teacher_input_prediction.device)
    else:
        numeric_lam = float(lam)
        weight = numeric_lam
    if not 0.0 <= numeric_lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    teacher_loss = relative_error(
        teacher_input_prediction,
        target,
        teacher_input,
        attention_mask=attention_mask,
        eps=eps,
    )
    student_loss = relative_error(
        student_input_prediction,
        target,
        teacher_input,
        attention_mask=attention_mask,
        eps=eps,
    )
    return weight * teacher_loss + (1.0 - weight) * student_loss


def lambda_schedule(
    step: int,
    total_steps: int,
    *,
    start: float = 1.0,
    end: float = 0.5,
    kind: Literal["fixed", "linear", "cosine", "teacher_only"] = "linear",
) -> float:
    """Return the local-loss mixing weight for a zero-based training step.

    The final configured step reaches ``end``. Steps outside the configured interval are
    clipped, which makes resumed or overrun loops deterministic.
    """

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    for name, value in (("start", start), ("end", end)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if kind not in {"fixed", "linear", "cosine", "teacher_only"}:
        raise ValueError("kind must be one of: fixed, linear, cosine, teacher_only")
    if kind == "teacher_only":
        return 1.0
    if kind == "fixed":
        if start != end:
            raise ValueError("fixed schedule requires start == end")
        return float(start)
    if total_steps == 1:
        return float(end)
    progress = min(step, total_steps - 1) / (total_steps - 1)
    if kind == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return float(start + (end - start) * progress)


def forward_kl(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    temperature: float = 1.0,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Compute token-mean forward KL ``KL(teacher || student) * temperature^2``."""

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher_logits and student_logits must have identical shapes; got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    if teacher_logits.ndim < 2:
        raise ValueError("logits must include at least one token axis and a vocabulary axis")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    teacher_probabilities = F.softmax(
        teacher_logits.detach().float() / temperature, dim=-1
    )
    student_log_probabilities = F.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    token_losses = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=-1)
    if attention_mask is None:
        loss = token_losses.mean()
    else:
        if attention_mask.shape != token_losses.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} must equal token shape "
                f"{tuple(token_losses.shape)}"
            )
        mask = attention_mask.to(device=token_losses.device, dtype=token_losses.dtype)
        selected = mask.sum()
        if not bool((selected > 0).item()):
            raise ValueError("attention_mask must select at least one token")
        loss = (token_losses * mask).sum() / selected
    return loss * (temperature**2)


__all__ = ["forward_kl", "lambda_schedule", "mixed_local_loss", "relative_error"]

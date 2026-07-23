from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DeformationLossOutput:
    loss: Tensor
    logit_mse: Tensor
    attention_js: Tensor
    deformation_size: Tensor
    normalized_drift: Tensor


def sample_smooth_monotone_warp(
    positions: Tensor,
    *,
    max_slope: float = 0.10,
    num_modes: int = 4,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample a smooth endpoint-preserving monotone deformation.

    The normalized warp is u -> u + tau(u), with tau represented by a sine
    series and rescaled so ||tau'||_inf <= max_slope < 1. Endpoints remain fixed.
    Supports positions [L] or [B,L].
    """
    if not (0.0 <= max_slope < 1.0):
        raise ValueError("max_slope must lie in [0,1)")
    if num_modes < 1:
        raise ValueError("num_modes must be positive")
    if positions.ndim not in (1, 2):
        raise ValueError("positions must have shape [L] or [B,L]")

    input_was_1d = positions.ndim == 1
    p = positions[None, :] if input_was_1d else positions
    batch, length = p.shape
    if length < 3:
        return positions.clone()

    p0 = p[:, :1]
    span = (p[:, -1:] - p0).clamp_min(torch.finfo(p.dtype).eps)
    u = (p - p0) / span
    modes = torch.arange(1, num_modes + 1, device=p.device, dtype=p.dtype)
    coeff = torch.randn(batch, num_modes, device=p.device, dtype=p.dtype, generator=generator)
    coeff = coeff / modes.square()[None, :]
    phase = torch.pi * u[:, :, None] * modes[None, None, :]
    tau = (coeff[:, None, :] * torch.sin(phase)).sum(dim=-1)

    du = (u[:, 1:] - u[:, :-1]).clamp_min(torch.finfo(p.dtype).eps)
    slope = (tau[:, 1:] - tau[:, :-1]) / du
    current = slope.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    tau = tau * (max_slope / current)
    warped = p0 + span * (u + tau)

    # Numerical safety: enforce monotonicity without changing endpoints materially.
    eps = torch.finfo(p.dtype).eps * span
    for idx in range(1, length):
        warped[:, idx] = torch.maximum(warped[:, idx], warped[:, idx - 1] + eps[:, 0])
    warped[:, -1] = p[:, -1]
    return warped[0] if input_was_1d else warped


def discrete_deformation_size(positions: Tensor, warped_positions: Tensor) -> Tensor:
    """Discrete analogue of ||tau'||_inf + ||tau''||_inf on [0,1]."""
    if positions.shape != warped_positions.shape:
        raise ValueError("positions and warped_positions must have the same shape")
    p = positions[None, :] if positions.ndim == 1 else positions
    w = warped_positions[None, :] if warped_positions.ndim == 1 else warped_positions
    span = (p[:, -1:] - p[:, :1]).clamp_min(torch.finfo(p.dtype).eps)
    u = (p - p[:, :1]) / span
    tau = (w - p) / span
    du = (u[:, 1:] - u[:, :-1]).clamp_min(torch.finfo(p.dtype).eps)
    grad = (tau[:, 1:] - tau[:, :-1]) / du
    grad_norm = grad.abs().amax(dim=-1)
    if grad.shape[-1] > 1:
        du2 = (0.5 * (du[:, 1:] + du[:, :-1])).clamp_min(torch.finfo(p.dtype).eps)
        hess = (grad[:, 1:] - grad[:, :-1]) / du2
        hess_norm = hess.abs().amax(dim=-1)
    else:
        hess_norm = torch.zeros_like(grad_norm)
    return (grad_norm + hess_norm).mean()


def _center_and_scale(logits: Tensor) -> Tensor:
    centered = logits - logits.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return centered / scale


def _js_divergence_from_logits(a: Tensor, b: Tensor) -> Tensor:
    pa = F.softmax(a.float(), dim=-1)
    pb = F.softmax(b.float(), dim=-1)
    m = 0.5 * (pa + pb)
    log_m = m.clamp_min(1e-12).log()
    kl_a = (pa * (pa.clamp_min(1e-12).log() - log_m)).sum(dim=-1)
    kl_b = (pb * (pb.clamp_min(1e-12).log() - log_m)).sum(dim=-1)
    return 0.5 * (kl_a + kl_b).mean()


class DeformationStabilityRegularizer(nn.Module):
    """Consistency regularizer under sampled smooth coordinate deformations."""

    def __init__(
        self,
        *,
        logit_weight: float = 1.0,
        attention_weight: float = 1.0,
        normalize_by_deformation: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.logit_weight = float(logit_weight)
        self.attention_weight = float(attention_weight)
        self.normalize_by_deformation = bool(normalize_by_deformation)
        self.eps = float(eps)

    def forward(
        self,
        kernel: nn.Module,
        q: Tensor,
        k: Tensor,
        positions: Tensor,
        *,
        warped_positions: Optional[Tensor] = None,
        max_slope: float = 0.10,
        num_modes: int = 4,
    ) -> DeformationLossOutput:
        if warped_positions is None:
            warped_positions = sample_smooth_monotone_warp(
                positions, max_slope=max_slope, num_modes=num_modes
            )
        clean = kernel(q, k, positions)
        warped = kernel(q, k, warped_positions)

        logit_mse = F.mse_loss(_center_and_scale(clean), _center_and_scale(warped))
        attention_js = _js_divergence_from_logits(clean, warped)
        size = discrete_deformation_size(positions, warped_positions)
        loss = self.logit_weight * logit_mse + self.attention_weight * attention_js
        normalized_drift = loss / (size.square() + self.eps)
        if self.normalize_by_deformation:
            loss = normalized_drift
        return DeformationLossOutput(
            loss=loss,
            logit_mse=logit_mse,
            attention_js=attention_js,
            deformation_size=size,
            normalized_drift=normalized_drift,
        )

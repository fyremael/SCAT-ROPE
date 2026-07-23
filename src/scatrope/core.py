from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validate_qk(q: Tensor, k: Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, heads, length, head_dim]")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have identical shapes; got {q.shape} and {k.shape}")
    if q.shape[-1] % 2 != 0:
        raise ValueError("head_dim must be even for planar rotary pairs")


def canonical_positions(length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.arange(length, device=device, dtype=dtype)


def _normalize_positions(positions: Optional[Tensor], q: Tensor) -> Tensor:
    batch, _, length, _ = q.shape
    if positions is None:
        return canonical_positions(length, device=q.device, dtype=q.dtype)
    positions = positions.to(device=q.device, dtype=q.dtype)
    if positions.ndim == 1:
        if positions.shape[0] != length:
            raise ValueError(f"positions length must be {length}; got {positions.shape[0]}")
        return positions
    if positions.ndim == 2:
        if positions.shape != (batch, length):
            raise ValueError(f"batched positions must have shape {(batch, length)}; got {positions.shape}")
        return positions
    raise ValueError("positions must have shape [length] or [batch, length]")


def pairwise_displacement(positions: Tensor) -> Tensor:
    """Return key-minus-query displacement with shape [1,L,L] or [B,L,L]."""
    if positions.ndim == 1:
        return positions[None, None, :] - positions[None, :, None]
    if positions.ndim == 2:
        return positions[:, None, :] - positions[:, :, None]
    raise ValueError("positions must be rank 1 or 2")


def rotate_pairs(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    """Apply norm-preserving 2-D Givens rotations to the last dimension.

    Args:
        x: [B,H,L,D], D even.
        positions: [L] or [B,L], continuous positions allowed.
        inv_freq: [D/2] angular frequencies.
    """
    if x.ndim != 4 or x.shape[-1] % 2 != 0:
        raise ValueError("x must have shape [B,H,L,D] with even D")
    if inv_freq.ndim != 1 or inv_freq.numel() != x.shape[-1] // 2:
        raise ValueError("inv_freq must have D/2 entries")

    if positions.ndim == 1:
        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
    elif positions.ndim == 2:
        angles = positions[:, :, None] * inv_freq[None, None, :]
        cos = angles.cos()[:, None, :, :]
        sin = angles.sin()[:, None, :, :]
    else:
        raise ValueError("positions must have shape [L] or [B,L]")

    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack((y0, y1), dim=-1).flatten(-2)


@dataclass(frozen=True)
class KernelDiagnostics:
    scales: Tensor
    demodulation: Tensor
    band_energy: Tensor


class StandardRoPEKernel(nn.Module):
    """Reference RoPE attention-score kernel."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        pair_idx = torch.arange(head_dim // 2, dtype=torch.float32)
        inv_freq = base ** (-2.0 * pair_idx / head_dim)
        self.head_dim = head_dim
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        return rotate_pairs(x, positions, self.inv_freq.to(dtype=x.dtype, device=x.device))

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        positions: Optional[Tensor] = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        _validate_qk(q, k)
        if q.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}; got {q.shape[-1]}")
        positions = _normalize_positions(positions, q)
        q_rot = self.rotate(q, positions)
        k_rot = self.rotate(k, positions)
        logits = torch.matmul(q_rot, k_rot.transpose(-1, -2)) / self.head_dim**0.5
        if return_diagnostics:
            return logits, {}
        return logits


class ScatteringRotaryKernel(nn.Module):
    """Localized multiscale rotary attention with controlled phase demodulation.

    The head channels are partitioned into frequency-ordered bands. For band b,
    a Gaussian relative-position envelope rho_b(Delta) localizes the rotary
    carrier. Outside that support, the carrier is smoothly stripped and the
    interaction approaches a phase-neutral content correlation:

        score_b = rho_b * <R(p)q_b, R(r)k_b>
                + eta_b * (1-rho_b) * <q_b, k_b>.

    All rotations are orthogonal Givens blocks. The kernel remains exactly
    invariant to a common translation of all positions because both rotary
    inner products and envelopes depend only on relative displacement.

    This is scattering-inspired, not a literal scattering transform: it borrows
    dyadic localization, contractive smooth envelopes, and carrier demodulation,
    while preserving signed/order-sensitive attention interactions.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        num_bands: int = 4,
        min_scale: float = 8.0,
        max_scale: float = 256.0,
        base: float = 10_000.0,
        demodulation: float = 0.75,
        envelope_power: float = 2.0,
        learnable_scales: bool = True,
        learnable_demodulation: bool = True,
        learnable_band_gains: bool = True,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        if num_bands < 1:
            raise ValueError("num_bands must be positive")
        if head_dim % num_bands != 0 or (head_dim // num_bands) % 2 != 0:
            raise ValueError("head_dim must split into num_bands even-width channel groups")
        if not (0.0 <= demodulation <= 1.0):
            raise ValueError("demodulation must lie in [0,1]")
        if min_scale <= 0 or max_scale < min_scale:
            raise ValueError("require 0 < min_scale <= max_scale")
        if envelope_power <= 0:
            raise ValueError("envelope_power must be positive")

        pair_idx = torch.arange(head_dim // 2, dtype=torch.float32)
        inv_freq = base ** (-2.0 * pair_idx / head_dim)
        scales = torch.logspace(
            torch.log10(torch.tensor(float(min_scale))),
            torch.log10(torch.tensor(float(max_scale))),
            num_bands,
        )
        demod = torch.full((num_bands,), float(demodulation)).clamp(1e-4, 1 - 1e-4)

        self.head_dim = head_dim
        self.num_bands = num_bands
        self.band_dim = head_dim // num_bands
        self.envelope_power = float(envelope_power)
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        log_scales = scales.log()
        if learnable_scales:
            self.log_scales = nn.Parameter(log_scales)
        else:
            self.register_buffer("log_scales", log_scales, persistent=True)

        demod_logits = torch.logit(demod)
        if learnable_demodulation:
            self.demod_logits = nn.Parameter(demod_logits)
        else:
            self.register_buffer("demod_logits", demod_logits, persistent=True)

        band_gains = torch.ones(num_bands)
        if learnable_band_gains:
            self.log_band_gains = nn.Parameter(band_gains.log())
        else:
            self.register_buffer("log_band_gains", band_gains.log(), persistent=True)

    @property
    def scales(self) -> Tensor:
        return self.log_scales.exp()

    @property
    def demodulation(self) -> Tensor:
        return self.demod_logits.sigmoid()

    @property
    def band_gains(self) -> Tensor:
        # Mean-one normalization keeps the initial logit scale controlled.
        gains = self.log_band_gains.exp()
        return gains * (self.num_bands / gains.sum().clamp_min(1e-8))

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        return rotate_pairs(x, positions, self.inv_freq.to(dtype=x.dtype, device=x.device))

    def _envelope(self, delta: Tensor, scale: Tensor) -> Tensor:
        # Generalized Gaussian; p=2 is the standard Gaussian/Morlet envelope.
        normalized = (delta.abs() / scale.clamp_min(1e-6)).pow(self.envelope_power)
        return torch.exp(-0.5 * normalized)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        positions: Optional[Tensor] = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        _validate_qk(q, k)
        if q.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}; got {q.shape[-1]}")
        positions = _normalize_positions(positions, q)
        delta = pairwise_displacement(positions)
        q_rot = self.rotate(q, positions)
        k_rot = self.rotate(k, positions)

        scales = self.scales.to(device=q.device, dtype=q.dtype)
        demod = self.demodulation.to(device=q.device, dtype=q.dtype)
        gains = self.band_gains.to(device=q.device, dtype=q.dtype)

        logits = q.new_zeros(q.shape[0], q.shape[1], q.shape[2], q.shape[2])
        band_energy = []
        for band in range(self.num_bands):
            start = band * self.band_dim
            stop = start + self.band_dim
            q_band = q[..., start:stop]
            k_band = k[..., start:stop]
            qr_band = q_rot[..., start:stop]
            kr_band = k_rot[..., start:stop]

            raw = torch.matmul(q_band, k_band.transpose(-1, -2))
            rotary = torch.matmul(qr_band, kr_band.transpose(-1, -2))
            rho = self._envelope(delta, scales[band])
            rho = rho[:, None, :, :]  # [1|B,1,L,L]

            band_score = rho * rotary + demod[band] * (1.0 - rho) * raw
            band_score = gains[band] * band_score
            logits = logits + band_score
            band_energy.append(band_score.square().mean().detach())

        logits = logits / self.head_dim**0.5
        if return_diagnostics:
            diagnostics = {
                "scales": scales.detach(),
                "demodulation": demod.detach(),
                "band_gains": gains.detach(),
                "band_energy": torch.stack(band_energy),
            }
            return logits, diagnostics
        return logits


class MultiheadScatteringAttention(nn.Module):
    """Drop-in research attention block using ScatteringRotaryKernel.

    This implementation is intentionally explicit and easy to instrument. It is
    not yet fused with FlashAttention because the scale-dependent pairwise
    envelopes require a custom score kernel.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        num_bands: int = 4,
        min_scale: float = 8.0,
        max_scale: float = 256.0,
        demodulation: float = 0.75,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.out = nn.Linear(dim, dim, bias=bias)
        self.dropout = float(dropout)
        self.kernel = ScatteringRotaryKernel(
            head_dim,
            num_bands=num_bands,
            min_scale=min_scale,
            max_scale=max_scale,
            demodulation=demodulation,
        )

    def forward(
        self,
        x: Tensor,
        positions: Optional[Tensor] = None,
        *,
        causal: bool = True,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch,length,dim]")
        batch, length, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected dim={self.dim}; got {dim}")

        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        logits = self.kernel(q, k, positions)
        if causal:
            causal_mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
            logits = logits.masked_fill(causal_mask, torch.finfo(logits.dtype).min)
        if attention_mask is not None:
            logits = logits.masked_fill(~attention_mask.to(torch.bool), torch.finfo(logits.dtype).min)

        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).reshape(batch, length, dim)
        y = self.out(y)
        if return_attention:
            return y, weights
        return y

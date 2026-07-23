import math

import torch

from scatrope import (
    DeformationStabilityRegularizer,
    ScatteringRotaryKernel,
    StandardRoPEKernel,
    sample_smooth_monotone_warp,
)


def _qk(batch=2, heads=3, length=32, dim=32):
    torch.manual_seed(7)
    return torch.randn(batch, heads, length, dim), torch.randn(batch, heads, length, dim)


def test_rotation_preserves_norm():
    q, _ = _qk()
    p = torch.arange(q.shape[2], dtype=q.dtype)
    kernel = ScatteringRotaryKernel(32, num_bands=4)
    qr = kernel.rotate(q, p)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=2e-5, rtol=2e-5)


def test_common_translation_invariance():
    q, k = _qk()
    p = torch.arange(q.shape[2], dtype=q.dtype)
    for kernel in (StandardRoPEKernel(32), ScatteringRotaryKernel(32, num_bands=4)):
        a = kernel(q, k, p)
        b = kernel(q, k, p + 137.25)
        assert torch.allclose(a, b, atol=5e-5, rtol=5e-5)


def test_large_scale_limit_matches_rope_when_demodulation_disabled():
    q, k = _qk(length=16)
    p = torch.arange(q.shape[2], dtype=q.dtype)
    rope = StandardRoPEKernel(32)
    scat = ScatteringRotaryKernel(
        32,
        num_bands=4,
        min_scale=1e8,
        max_scale=1e8,
        demodulation=0.0,
        learnable_scales=False,
        learnable_demodulation=False,
        learnable_band_gains=False,
    )
    assert torch.allclose(rope(q, k, p), scat(q, k, p), atol=2e-5, rtol=2e-5)


def test_warp_is_monotone_and_endpoint_preserving():
    p = torch.arange(128, dtype=torch.float32)
    w = sample_smooth_monotone_warp(p, max_slope=0.2)
    assert torch.all(w[1:] > w[:-1])
    assert math.isclose(float(w[0]), float(p[0]), abs_tol=1e-6)
    assert math.isclose(float(w[-1]), float(p[-1]), abs_tol=1e-5)


def test_regularizer_backpropagates():
    q, k = _qk(batch=1, heads=2, length=24, dim=32)
    q.requires_grad_(True)
    k.requires_grad_(True)
    p = torch.arange(q.shape[2], dtype=q.dtype)
    kernel = ScatteringRotaryKernel(32, num_bands=4)
    regularizer = DeformationStabilityRegularizer()
    out = regularizer(kernel, q, k, p, max_slope=0.05)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()

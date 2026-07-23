# ScatRoPE

ScatRoPE is a research implementation of a scattering-inspired rotary attention kernel. It replaces unrestricted global rotary carriers with frequency-ordered, multiscale relative-position bands and smoothly removes carrier phase outside each band's effective support.

## Core operator

For rotary band \(b\), relative displacement \(\Delta=p_j-p_i\), scale \(s_b\), and demodulation coefficient \(\eta_b\),

\[
\rho_b(\Delta)=\exp\left[-\frac12\left|\frac{\Delta}{s_b}\right|^p\right],
\]

\[
A^{(b)}_{ij}=\rho_b(\Delta)\langle R_b(p_i)q_i,R_b(p_j)k_j\rangle
+\eta_b[1-\rho_b(\Delta)]\langle q_i,k_j\rangle.
\]

The first term is a localized rotary carrier. The second is controlled phase demodulation: beyond the band's support, positional phase is stripped rather than allowed to oscillate without bound. Dyadic scales assign narrow support to high-frequency channels and broad support to low-frequency channels.

This is not a literal Mallat scattering transform. It borrows three structural ideas: multiscale localization, nonexpansive smooth envelopes, and carrier demodulation. It deliberately retains signed and directional interactions needed for attention.

## Properties

- Givens rotations preserve query/key norms exactly up to floating-point error.
- A common translation of all positions leaves scores unchanged.
- Continuous positions and batched latent clocks are supported.
- Scale, demodulation, and band gains may be learned.
- A deformation-consistency regularizer is included.
- The explicit kernel is instrumentable but not FlashAttention-fused.

## Install and test

```bash
python -m pip install -e .[dev]
pytest
python benchmarks/deformation_benchmark.py --output-dir results
```

## Minimal use

```python
import torch
from scatrope import ScatteringRotaryKernel, DeformationStabilityRegularizer

B, H, L, D = 2, 8, 512, 64
q = torch.randn(B, H, L, D, device="cuda")
k = torch.randn_like(q)
pos = torch.arange(L, device=q.device, dtype=q.dtype)

kernel = ScatteringRotaryKernel(
    D,
    num_bands=4,
    min_scale=8,
    max_scale=L,
    demodulation=0.75,
).cuda()

logits = kernel(q, k, pos)
regularizer = DeformationStabilityRegularizer(logit_weight=1.0, attention_weight=1.0)
stability = regularizer(kernel, q, k, pos, max_slope=0.08)
loss = task_loss + 0.01 * stability.loss
```

## Recommended first ablation grid

- Encoding: RoPE; localized rotary with \(\eta=0\); ScatRoPE with \(\eta\in\{0.25,0.5,0.75,1\}\).
- Bands: 2, 4, 8.
- Scales: fixed dyadic; learned from dyadic initialization.
- Regularization weight: 0, 1e-3, 1e-2, 1e-1.
- Warps: smooth sinusoidal, piecewise tempo changes, local insert/delete simulations, latent-clock perturbations.
- Report clean task quality, warped task quality, degradation slope, norm drift, per-distance attention drift, runtime, and peak memory.

## Interpretation limits

The synthetic benchmark only checks the intended inductive bias: reduced sensitivity of attention geometry to smooth coordinate warps. It does not establish improved language-model perplexity, long-context retrieval, or audio alignment. Those require trained, multi-seed comparisons against RoPE and strong scaling baselines.

## Included validation results

All five unit tests pass on PyTorch 2.10 CPU. They cover rotary norm preservation, common-translation invariance, recovery of standard RoPE in the infinite-scale/no-demodulation limit, monotone warp generation, and regularizer backpropagation.

In the included synthetic deformation benchmark (`L=128`, `B=2`, `H=4`, `D_h=64`, eight warps per setting), a maximum warp slope of 0.10 produced normalized logit drift of 0.29051 for RoPE and 0.03249 for demodulating ScatRoPE, an 88.8% reduction. Far-distance drift fell from 0.34160 to 0.00714, a 97.9% reduction. The explicit unfused CPU implementation took 2.79 times the RoPE forward time in this configuration.

In the included three-seed two-copy recency task, models trained only on uniform positions. Mean clean/warped accuracy was 0.9505/0.9368 for RoPE, 0.9974/0.9987 for localized ScatRoPE, and 0.9993/0.9987 for demodulating ScatRoPE. This toy task is intentionally diagnostic and may favor localized recency structure; it is not evidence of language-model superiority.

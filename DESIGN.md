# ScatRoPE design note

## Objective

Build a rotary positional mechanism that is exactly relative under global translations but stable, rather than invariant, under smooth nonuniform reparameterizations. The operator should remain compatible with unit-hypersphere models by using only orthogonal planar rotations in the rotary path.

## Why a pairwise kernel is necessary

A nonlinear bounded relative phase \(\psi(p_j-p_i)\) generally cannot be written as \(\phi(p_j)-\phi(p_i)\). Consequently, a faithful localized relative kernel cannot in general be implemented solely by independently rotating queries and keys. ScatRoPE therefore computes per-band pairwise score modulation. This preserves exact origin invariance but currently requires a custom attention-score kernel.

## Frequency-scale coupling

The standard RoPE frequency order is retained. Contiguous high-frequency rotary pairs receive small spatial scales; low-frequency pairs receive large scales. This gives the wavelet-like allocation

- high frequency: precise but local;
- low frequency: coarse but long range.

## Controlled demodulation

Pure localization with \(\eta=0\) suppresses a band outside its window. Nonzero \(\eta\) instead transitions to an unrotated signed content correlation. This is analogous to stripping a carrier to recover a slowly varying envelope, but it avoids the full complex modulus because modulus would destroy sign and orientation information used by attention.

## Stability regularization

For sampled monotone warp \(f(t)=t+\tau(t)\), compute clean and warped attention logits and minimize

\[
\mathcal L_{\mathrm{stab}}
=\lambda_z\|\mathcal N(A)-\mathcal N(A_f)\|_2^2
+\lambda_a\operatorname{JS}(\operatorname{softmax}A,\operatorname{softmax}A_f),
\]

where \(\mathcal N\) removes row mean and normalizes row RMS. The implementation also reports a discrete deformation size based on first and second differences of \(\tau\), but normalization by that size is optional because it can overweight extremely small warps during training.

## Falsification criteria

Reject or revise the mechanism if any of the following hold across controlled multi-seed experiments:

1. Clean-task quality falls materially relative to RoPE at equal compute.
2. Robustness gains disappear after matching runtime or parameter count.
3. Learned scales collapse to the maximum context scale, reducing the method to RoPE-like global carriers.
4. Demodulation erases order-sensitive performance.
5. Stability regularization merely produces diffuse attention or underfitting.
6. A simpler data augmentation or latent-clock method gives equal robustness at lower cost.

## RUNT integration

The rotary transformation is orthogonal and therefore norm-preserving. In a unit-normalized transformer, apply the kernel after normalized Q/K projections. No extra renormalization is needed for the rotation itself. The pairwise score mixture changes attention geometry, not Q/K norms.

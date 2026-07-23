# References and adjacent work

1. Stéphane Mallat. *Group Invariant Scattering*. Communications on Pure and Applied Mathematics, 2012; arXiv:1101.2286. Establishes translation-invariant scattering operators with Lipschitz stability to diffeomorphisms through localized wavelets, modulus cascades, and averaging.
2. Joan Bruna and Stéphane Mallat. *Invariant Scattering Convolution Networks*. IEEE TPAMI, 2013; arXiv:1203.1513.
3. Joakim Andén and Stéphane Mallat. *Deep Scattering Spectrum*. IEEE TSP, 2014; arXiv:1304.6763. Applies scattering stability to time warps in audio.
4. Jianlin Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864; later Neurocomputing 2024.
5. Yusuke Oka et al. *Wavelet-based Positional Representation for Long Context*. ICLR 2025; arXiv:2502.02004. Treats RoPE through a wavelet lens and develops variable-scale relative wavelet positional representations, but does not use the ScatRoPE pairwise demodulation kernel implemented here.
6. Valerio Ruscio et al. *Beyond Position: the Emergence of Wavelet-like Properties in Transformers*. arXiv:2410.18067.
7. Athanasios Zeris. *Energy-Gated Attention and Wavelet Positional Encoding*. arXiv:2605.26355. Uses learned Gaussian-windowed Morlet positional features; reported experiments are small-scale.
8. Feilong Liu. *Rotary Positional Embeddings as Phase Modulation*. arXiv:2602.10959. Frames RoPE as a bank of phase-modulated oscillators and studies long-context phase stability.

ScatRoPE's claim should therefore be narrow: the combination of frequency-ordered relative rotary bands, smooth pairwise carrier stripping, exact common-translation invariance, unit-norm-compatible Givens transport, and an explicit deformation-consistency regularizer. A formal novelty claim requires a broader literature and patent search.

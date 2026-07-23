# ScatRoPE Phase-II falsification protocol

## Purpose

The initial experiments establish that localization and demodulation reduce score drift under smooth positional warps. They do not establish that the resulting positional system preserves exact long-range computation or improves trained language models. Phase II is therefore organized around four claims that may fail independently.

## F1 — Exact long-range positional discrimination

**Question.** Does phase localization erase the ability to distinguish one exact distant offset from adjacent and aliased offsets?

The exact-offset probe gives every key the same content vector. A query is pre-rotated so ordinary RoPE is phase-matched at one designated displacement. Content similarity cannot solve the task. We report top-1 target recovery, reciprocal rank, and the target-to-best-distractor margin by offset.

**Primary failure criterion.** At offsets at least one half of the reference training length, ScatRoPE top-1 accuracy falls more than two percentage points below RoPE, or its mean target margin becomes non-positive.

This probe is intentionally hostile to ScatRoPE. It measures the information sacrificed when remote carrier phase is stripped.

## F2 — Extrapolation beyond the reference length

**Question.** Do fixed multiscale supports remain usable at sequence lengths and relative distances not represented by the reference scale?

The exact-offset probe is repeated at 2x and 4x the reference length without rescaling the learned or fixed ScatRoPE bands. The language model is likewise trained at one context length and evaluated at longer lengths without parameter changes.

**Primary failure criterion.** Long-offset top-1 accuracy falls more than five percentage points below RoPE, or clean perplexity degradation from 1x to 4x is materially steeper than the matched RoPE model.

A separate scale-rescaling policy may later be studied, but it must not be substituted into the primary test because that would conceal fixed-scale extrapolation failure.

## F3 — Language-model perplexity

**Question.** Does reduced deformation sensitivity improve or at least preserve ordinary next-token prediction?

All variants use an identical causal Transformer, tokenizer, token budget, optimizer, initialization seeds, and evaluation windows. The primary matrix is:

1. RoPE
2. RoPE plus deformation regularization
3. fixed-band ScatRoPE
4. fixed-band ScatRoPE plus the identical deformation regularization

The fixed-band primary comparison prevents additional learnable scale parameters from confounding the result. Report clean and warped validation loss/perplexity, clean-to-warped prediction agreement, tokens per second, and peak memory.

**Primary failure criterion.** At the training context length, ScatRoPE clean perplexity exceeds RoPE by more than 2%. Synthetic-corpus results are smoke tests only and cannot satisfy this gate.

## F4 — Benefit beyond equal regularization

**Question.** Is the gain architectural, or can RoPE obtain it from the same deformation-consistency objective?

The regularizer, sampled warps, coefficient, number of kernel evaluations, token budget, and random seeds are matched between `RoPE+Reg` and `ScatRoPE+Reg`.

**Primary success criterion.** At the longest evaluation context, `ScatRoPE+Reg` improves warped perplexity by at least 1% relative to `RoPE+Reg`, while remaining inside the 2% clean-perplexity tolerance.

If both regularized models converge to the same result, the correct conclusion is that the regularizer—not ScatRoPE—is carrying the observed benefit.

## Execution

Harness validation:

```bash
python benchmarks/falsification_suite.py \
  --config configs/falsification_quick.json \
  --output-dir results/falsification/quick
```

Scientific language-model run with identical local byte corpora:

```bash
python benchmarks/falsification_suite.py \
  --config configs/falsification_full.json \
  --train-file data/train.txt \
  --valid-file data/valid.txt \
  --output-dir results/falsification/full
```

The full run should be repeated on at least two corpora with different structural statistics. Recommended first choices are a natural-language corpus and a code corpus, tokenized identically across variants.

## Required reporting

No aggregate score may hide a failed gate. Publish:

- per-offset exact recovery and margin;
- per-length clean and warped perplexity;
- all seed-level records;
- throughput and peak memory;
- clean/warped prediction agreement;
- gate outcomes with fixed thresholds established before the run;
- failures and negative results without post-hoc threshold changes.

# ScatRoPE falsification report

Language-model evidence source: `synthetic_smoke`.

## Decision gates

| gate | status |
|---|---|
| exact_long_range_retention | **fail** |
| unseen_length_extrapolation | **fail** |
| clean_language_model_perplexity | **pass** |
| matched_regularized_benefit | **fail** |

> Synthetic perplexity validates the harness only; it cannot support a language-model claim.

## Language-model summary

| model | eval length | clean PPL | warped PPL | agreement | tokens/s |
|---|---:|---:|---:|---:|---:|
| RoPE | 32 | 63.7405 | 63.7403 | 1.0000 | 62064.1 |
| RoPE | 64 | 65.0701 | 65.0700 | 1.0000 | 62064.1 |
| RoPE+Reg | 32 | 63.7429 | 63.7427 | 1.0000 | 35305.5 |
| RoPE+Reg | 64 | 65.0706 | 65.0705 | 1.0000 | 35305.5 |
| ScatRoPE | 32 | 63.7392 | 63.7392 | 1.0000 | 46901.9 |
| ScatRoPE | 64 | 65.0738 | 65.0738 | 1.0000 | 46901.9 |
| ScatRoPE+Reg | 32 | 63.7391 | 63.7391 | 1.0000 | 19760.7 |
| ScatRoPE+Reg | 64 | 65.0738 | 65.0739 | 1.0000 | 19760.7 |

# ScatRoPE synthetic deformation benchmark

Metric: normalized attention-logit drift under smooth monotone coordinate warps. Lower is better.

| max slope | model | logit drift | near | mid | far | attention JSD | runtime vs RoPE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.020 | RoPE | 0.07325 | 0.01668 | 0.06410 | 0.08952 | 0.0000002 | 1.00x |
| 0.020 | ScatRoPE-localized | 0.00951 | 0.01141 | 0.01018 | 0.00438 | 0.0000000 | 2.75x |
| 0.020 | ScatRoPE-demod | 0.00695 | 0.01090 | 0.00810 | 0.00161 | 0.0000000 | 2.79x |
| 0.050 | RoPE | 0.17351 | 0.04061 | 0.15114 | 0.21247 | 0.0000010 | 1.00x |
| 0.050 | ScatRoPE-localized | 0.02324 | 0.02814 | 0.02473 | 0.01061 | 0.0000000 | 2.75x |
| 0.050 | ScatRoPE-demod | 0.01701 | 0.02691 | 0.01971 | 0.00393 | 0.0000000 | 2.79x |
| 0.100 | RoPE | 0.29051 | 0.07577 | 0.27083 | 0.34160 | 0.0000026 | 1.00x |
| 0.100 | ScatRoPE-localized | 0.04405 | 0.05365 | 0.04693 | 0.01898 | 0.0000000 | 2.75x |
| 0.100 | ScatRoPE-demod | 0.03249 | 0.05180 | 0.03749 | 0.00714 | 0.0000000 | 2.79x |
| 0.200 | RoPE | 0.45899 | 0.14304 | 0.44539 | 0.52501 | 0.0000064 | 1.00x |
| 0.200 | ScatRoPE-localized | 0.08653 | 0.10601 | 0.09156 | 0.03853 | 0.0000001 | 2.75x |
| 0.200 | ScatRoPE-demod | 0.06387 | 0.10261 | 0.07322 | 0.01443 | 0.0000001 | 2.79x |

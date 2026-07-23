# Warped recency retrieval benchmark

Two identical content keys occur before a query; the correct key is the most recent occurrence. Models train on uniform positions and are evaluated on clean and smoothly warped coordinates.

| model | clean accuracy | warped accuracy | prediction agreement | clean loss | warped loss |
|---|---:|---:|---:|---:|---:|
| RoPE | 0.9505 ± 0.0106 | 0.9368 ± 0.0121 | 0.9863 ± 0.0097 | 1.2359 | 1.2299 |
| ScatRoPE-demod | 0.9993 ± 0.0009 | 0.9987 ± 0.0009 | 0.9993 ± 0.0009 | 1.3761 | 1.3690 |
| ScatRoPE-localized | 0.9974 ± 0.0018 | 0.9987 ± 0.0009 | 0.9987 ± 0.0009 | 1.3373 | 1.3137 |

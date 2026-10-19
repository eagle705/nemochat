# Historical midtraining variants

Versioned experiments remain adjacent to their owning implementation under `../training/offline_kd/`: `prefix/continuation*`, `prefix/dynamic/{attempt02,restart01,train1000}`, `prefix/filtered`, `ablation_lm_only/ordinary02`, and `continue2000`. These are historical controls/recovery attempts, not a series of automatic production launch stages. Explicit nested paths preserve local imports and version-specific guards; consult each README and immutable run receipt before reuse. Latest accepted KD1000 source family is `../training/offline_kd/prefix/dynamic/train1000/`.

`../evaluation/ablation_{auto,eval,gpu}` contains comparison/report tooling, not SFT. Midtraining publication contains KD-only exports. The original frozen bytes are preserved by the migration snapshot. Relocated versions need fresh review/source freezes and native qualification before execution.

# Supervised fine-tuning

SFT is a top-level stage, never part of KD/midtraining. Run from repository root.

- `data/`: pinned source discovery/acquisition, official selected-assistant preparation, tokenizer migration and `acceptance/` transport/provenance guards.
- `configs/`: native recipe, approved dataset settings, long-context plan, tokenizer migration contract and sealed-input contract.
- `training/`: native SFT and inert guarded launchers; v1 deadline preparation is historical, not a fresh ready-to-run authorization.
- `resume/runtime/`: actual full-state continuation implementation used for905→1248, not HF model-only reset.
- `evaluation/benchmark1248/`: small-subset diagnostic scoring/export code for latest accepted checkpoint.
- `publication/hf_sft1200/`: historical private publication templates/tooling. The repository name1200 refers to actual1248.
- `archive/`: failed/superseded runtime02–05, recovery02, snapshot/sealed-input experiments, first905 examples and initial preparation handoff. These are reference implementations, not competing current entrypoints. Dependencies between these versioned sources remain explicit; do not erase them or bypass guards.

Latest accepted native checkpoint is1248 (343 full-state updates after905); private revision `893c00ca610f25bde4c6b46050a224ecc9388385`. No new remote readback or GPU rerun occurs in this refactor. All six128-token/no-EOS examples retain failures;400-question diagnostics are not a quality win. Native/HF parity remains unrun; payload forensics remain waived.

Future inputs: `datasets/sft`; weights/state: `checkpoints/sft`; reports: `results/sft`. Existing payloads/receipts, including `results/midtraining/sft-continue1500-02/main/checkpoint/iter_0001248`, remain valid immutable legacy locations. Run-bound launchers retain old receipt defaults deliberately and fail closed when exact historical dependencies are absent. Do not run archived launchers against accepted outputs.

Portable help: `.venv/bin/python sft/data/nemotron_sft.py --help`.
See [source routing and operational limits](../docs/en/STAGE_LAYOUT.md).

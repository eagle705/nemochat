# Ordered native Bridge pilot entrypoints

Run from this checkout with its existing `.venv`, pinned NeMo image, Engine access and prepared pilot data. The wrappers currently depend on the local validation helpers under `results/`; they are not a standalone production trainer or a portable fresh-clone installation.

```bash
bash scripts/pretraining/1.wandb_preflight.sh
bash scripts/pretraining/2.megatron_bridge_pretrain.sh bridge-my-fresh-run 1800
bash scripts/pretraining/3.verify_run.sh bridge-my-fresh-run
```

1. Preflight parses allowlisted private `.env` assignments without shell sourcing, explicitly enables online tracking, and verifies a separate remote connectivity run.
2. Training runs the installed native Megatron Bridge `pretrain` entrypoint on eight GPUs with real indexed data, four WSD updates (W1/S2/D1), and native online W&B logging. Use a fresh run name; existing results/checkpoints must never be overwritten. Check GPU ownership first; do not overlap unrelated jobs.
3. Verification requires the exact training container to have exited successfully and all eight rank receipts. It exclusively publishes a fresh checkpoint inventory, or verifies an existing identical inventory without rewriting it, reads every payload twice, and checks remote W&B metrics. A mismatch fails rather than repairing the manifest. This is file/state-receipt verification, not an actual native checkpoint resume test.

Verified execution: `bridge-fresh-wsd4-01`; native W&B run `nvidia/nemochat/bridge-train-9271cf68694e`. Four updates/4,096 loss tokens passed. Fresh checkpoint SHA verification and separate metadata-indexed full DCP record CRC scan passed; native reload remains untested for this checkpoint. Timing was confounded by concurrent workloads and is not a throughput benchmark.

Detailed report: `results/pretrain_followup/bridge-fresh-wsd4-01/REPORT.md`. Parent verification evidence: `parent-remote-readback.json`, `parent-full-dcp-crc.json`, and `logs/pretrain_followup/parent-numbered-verify.log`.

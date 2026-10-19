# SFT905 full-state native continuation

All GPU launch paths require a new parent release bound to fresh sealed RAM READY. No GPU is launched by preparation, freeze, unit tests or `verify_cpu.py`. No checkpoint payload hashes/copies, fresh C weight import, recalibration or extra reload/eval gate.

The old C HF config is architecture metadata only. Native905 model/optimizer/main params/scheduler/Python+NumPy+Torch+CUDA+tracker RNG and TrainState must load through native Bridge; observed native restore methods must return successfully. Missing/partial loads fail. First global update906 is segment_update1. The native full horizon34135, one epoch4369280rows, batch128, seed42 remain unchanged; native consumed samples115840 resumes at that index in the same seeded permutation. The restored optimizer LR is used untouched; actual optimizer-boundary LRs and post-scheduler LRs are logged separately. No synthetic terminal minimum LR.

## Parent release and launch

After the new holder publishes accepted READY and parent has accepted its actual descriptor transport and identical-byte audits:

```
.venv/bin/python midtraining/sft_after_kd/continue1500/runtime/release.py \
 --ready /raid/yoon/nemochat/results/midtraining/sft-continue1500-snapshot-01/READY.sealed.json \
 --producer-freeze /ABSOLUTE/ACTUAL/PRODUCER/source-freeze.json \
 --parent-accept-fresh-transport-and-identical-data-audits
midtraining/sft_after_kd/continue1500/runtime/start.sh
```

`release.py` never starts GPU work. Use the producer's actual canonical freeze path. The parent release cannot bind the old holder. No polling/waiting/retry or automatic restart. Fresh service: `nemochat-sft-sealed03-continue1500-main.service`; exact Engine: `nemochat-sft-sealed03-continue1500-main`, sealed03 label preserved. Shared all8 lock + live idle check precede allocation. The supervisor checks its systemd PID/UID/argv; cleanup touches only its owned Engine.

At launch, cumulative cap=905+floor((14:50-now-300)/(14.5/0.85)), bounded by34135. 14:50KST train stop;600seconds reserved through15:00KST for native save/cleanup. At step start, a collective deadline check invokes native synchronous save of the current committed state then clean exit without another forward. Native end-of-step duration/cap paths also save full state. Final hard watchdog failure remains failure, never fabricated success. Runtime uses fresh HOME/compiler caches and /nct private8G exec tmpfs.

Final checkpoint path: `results/midtraining/sft-continue1500-01/main/checkpoint/iter_<cumulative-step>`. Accept requires eight contiguous905->final update logs, actual full-state restore provenance, all-rank save ACKs, exact clean Engine exit and actual checkpoint inventory. Optional final evaluation/on_train_end may not run on native cap SystemExit0 and are not acceptance gates. Coverage is segment [115840,final_step*128), never a fake fresh prefix.

CPU proof includes actual pinned ConfigContainer/provider construction, post-wrapper unchanged four-SDK guard + negative controls, actual native single sampler from115840 on all eight ranks, a native cap-decision control fixture with save I/O intercepted (NOT real full-state save), and real spawned Manager Queue/executable/shared-object at /nct. Actual native weights load and new learning/full-save remain UNRUN until parent release and GPU execution.

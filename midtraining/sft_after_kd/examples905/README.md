# Final SFT905 local export, actual examples, and frozen benchmark

Source: `results/midtraining/sft-after-kd-runtime05/main/checkpoint/iter_0000905`.
Local HF: `checkpoints/exports/nemochat_sft905_hf_bf16_01`.
Migrated SFT tokenizer revision: `fc067b34f6b514014d2f8bddba855d2e4bfdfba6`.

The original KD1000 exporter/schema helper are imported read-only. The task-local
adapter adds only the observed `chained_0.optimizer.*` and `chained_1.optimizer.*`
training-only namespaces, selects the exact SFT tokenizer, preserves native scan
clamp0, and records the SFT context limit16384. No C/BASE weight tensor is loaded.
619 native model records map exhaustively to447 HF runtime records /6289 serialized
records;9,790,967,416 parameters. Native/source and reloaded tensors are finite;
23 router correction buffers remain FP32. No source payload SHA/CRC audit was run.
The inherited converter's console message says `KD1000`; its actual DCP source
argument and final `export_result.json` correctly identify final SFT905.

Examples were generated once each on GPU0 through official Transformers5.12.1,
trust_remote_code=False, eager BF16, seed42, greedy max_new_tokens128, exact official
nonthinking chat template. All six reached the cap without official EOT. Outputs
were not retried or edited. They show instruction-following/language/repetition
failures; loading success is not quality success.

- `results/midtraining/sft905-examples-01/examples.json`: full generation IDs/text/runtime.
- `results/midtraining/sft905-examples-01/examples.en.md` and `.ko.md`: full readable outputs.
- `results/midtraining/sft905-examples-01/final-acceptance.json`: exact Engine readbacks/source lineage.

The later user authorization added one SFT905-only execution of the prior frozen
400-question/1600-choice benchmark and64-window/65536-target pretraining-heldout
suite. `benchmark/score_sft905.py` invokes the unchanged prior scoring logic; its
only adaptation is the disclosed full-vocabulary identity guard after proving
that only aliases18..29 differ and all1600 frozen prompt/choice encodings match.
Benchmark prompts stay RAW (no chat template), unlike the chat examples.
A/C/B reports were read unchanged, not rerun. D adds85,001,437 assistant-target
exposures and is not a controlled KD ablation.

- `results/midtraining/sft905-benchmark-gpu-01/D-full01/report.json`: all400 per-question scores and64 windows.
- `results/midtraining/sft905-benchmark-gpu-01/complete.json`: verified exact suite/model/Engine.
- `results/midtraining/sft905-benchmark-gpu-01/summary.json`: four-arm metrics.
- `results/midtraining/sft905-benchmark-gpu-01/FINAL_REPORT.md` and `.ko.md`: scoped interpretation and two PNGs.

No training, downloads, Hub publication, checkpoint forensic copies or benchmark
retries occurred. Native/HF logit parity remains unrun. Operational Transformers
compatibility is established; payload byte integrity remains explicitly waived.

# 한국어 요약

실제 최종 native SFT905를 local HF로 변환했고 migrated SFT tokenizer를 정확히
사용했습니다. 영어/한국어 프롬프트6개를 각각 한 번만 생성하여 전체 원문을
보존했습니다. 모두128-token 상한에 도달했고 반복/지시 불이행이 관측되었습니다.
후속 사용자 승인으로 기존과 동일한 RAW prompt benchmark를 D에만 실행했습니다.
기존 A/C/B는 재실행하거나 변경하지 않았습니다. D의 추가 SFT supervision 때문에
통제된 KD 비교가 아니며, heldout NLL/PPL은 SFT validation Loss가 아닙니다.
학습/다운로드/Hub write/재시도는 없었고 native/HF parity와 checkpoint byte integrity는
각각 미실행/명시적 waiver 상태입니다.

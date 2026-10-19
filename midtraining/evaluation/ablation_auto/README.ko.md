# 동일 suite 자동 ablation controller — Parent 검토 전 실행 금지

소유 범위는 `midtraining/evaluation/ablation_auto/`와 `results/midtraining/offline2b-ablation-auto-01/`뿐입니다. 기존 B learner, A evaluator, converter, root ledger를 수정하지 않습니다.

## 명령

Repository root `/raid/yoon/nemochat`에서:

```bash
.venv/bin/python -m unittest discover -s midtraining/evaluation/ablation_auto -v
bash midtraining/evaluation/ablation_auto/start.sh  # 실제 job을 시작하지 않는 preview
# Parent가 source-freeze.json과 실제 source를 검토한 뒤에만:
bash midtraining/evaluation/ablation_auto/start.sh --start
```

실제 unit은 `nemochat-offline2b-ablation-auto-01.service`, UID1001, Restart=no, 최대23h입니다. 전용 run.lock을 독점 생성하므로 실패 후 자동 재실행하지 않습니다. 새 사용자 승인 요청은 필요 없지만 Parent source 검토는 필수입니다. Builder는 production을 시작하지 않습니다.

## 자동 실행 순서

1. 기존 B PID4083833/UID1001/starttime47312487/전체 argv를 검사하고 pidfd를 열어 재검사한 뒤 실제 exit event를 기다립니다. Zombie/이미 종료된 경우에도 final proof를 확인해야 합니다. 실패·중단·다른 identity·최종1000 미완료는 failure.json이며 학습을 재시작하지 않습니다.
2. B Engine exit0/noOOM, 8rank 각각1000 optimizer calls/999positive+1zeroLR, finite positive Grad Norm, noSkipped/nonfinite, 원본 BASE fresh0 및1.0LM objective를 검증합니다. B/C의 실제 accepted physical mapping·input digest·순서·LR가 같아야 합니다. Final checkpoint1000의 model/optimizer/RNG metadata·8rank transaction·8distcp/.metadata를 확인합니다. 사용자가 면제한 checkpoint SHA/CRC/forensic payload 읽기는 수행하지 않습니다.
3. A가 이미 실패했다면 release를 차단합니다. A를 기다리지 않고 C CPU 평가를 시작하며, 새로운 B HF 경로 `checkpoints/exports/nemochat_mid_lmonly_hf_bf16_02`를 미래 실제 실행 시에만 생성합니다. 기존 converter를 schema-only → 실제 B model export → 공식 HF full reload/한영 finite forward/cache generation 순서로 실행합니다. `--source-step 1000 --stage midtraining_lm_only --model-prefix ''`를 사용하며 BASE weights를 B로 대체하지 않습니다. B HF가 검증되면 B CPU 평가를 시작합니다.
4. 기존 A Engine `e1404d98ba226a85000984e8a7dee2dc758a797c997fbb46ea9ec1aaa845bfa2`는 그대로 유지합니다. 필요 시 Engine blocking wait로 종료를 기다릴 뿐 새로 실행하거나 중단하지 않습니다. B/C는 A와 동일한 frozen evaluator, pinned CPU image, BF16/Transformers5.12.1 backend를 사용합니다. 각각4CPU/128GiB/최대6h/networknone/noGPU입니다. 별도 GPU variant는 없습니다.
5. 동일400문항/1600선택지,64heldout window/65536target token과 정확한 IDs/분모/aggregate 및 실제 per-question receipt를 검사한 뒤 paired compare.py 통계와 PNG를 생성합니다. 표시 순서는 **[1]BASE → [2]LM+KD → [3]LM-only**입니다. Benchmark Accuracy와 전체/source별 NLL/PPL, summary.json, FINAL_REPORT.md/.ko.md를 작성합니다. 실제 Engine exit와 PNG decode 및 소유 job cleanup 검증 후 complete.json을 마지막으로 원자적으로 발행합니다.

`complete.json`에는 모든3개 검증 report와 절대 PNG 경로가 있습니다. 일부만 완료되거나 실패하면 failure.json입니다. Parent가 이 정확한 unit의 pidfd terminal watcher를 설정하여 기존 Slack thread에 실제 PNG를 전달합니다. Controller 자체는 Slack posting/HF publishing을 수행하지 않습니다.

MMLU10subjects×20문항5shot, ARC100test0shot, HellaSwag100validation0shot은 비표준 diagnostic subset이며 전체 MMLU/공식 leaderboard가 아닙니다. Benchmark contamination은 확인되지 않았습니다. B1.0LM 대 C0.5LM+0.5KD는 실용 recipe 비교이고 coefficient-matched KD 인과 효과가 아닙니다. Bootstrap은 질문/window 표본 불확실성이며 training seed 불확실성이 아닙니다. 성능 하락도 그대로 보고합니다.

CPU test는 실제 pidfd exit/zombie/stale identity/timeout, 실제 C artifact schema, partial/missing rank/shard/잘못된 suite/backend/분모/확률 gate, A blocking wait control, 명시적 mock Engine을 통한 기존 foreground launcher 연결, durable failure/terminal ordering과 실제 PNG 생성을 검증합니다. CPU TEST FIXTURE 그림은 실제 모델 평가 결과가 아닙니다. Production B export와 전체3개 평가 완료는 Parent start 전에는 실행되지 않았습니다. 상세 source 및 운영 contract는 README.md를 확인하십시오.

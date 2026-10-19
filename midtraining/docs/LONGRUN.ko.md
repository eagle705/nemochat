# 신규 corpus native LM+KD 장시간 실행

English: [LONGRUN.md](LONGRUN.md).

## 구현 범위와 실행 상태

`longrun_native.py`, `launch_longrun.py`, `calibration.yaml`, `longrun24h.yaml`은 native 실행 경로와 차단 게이트다. **24시간 학습을 실행했다는 증거가 아니다.** 실제 pinned image config 구성, GPU calibration/control, 충분한 production corpus 확보, 독립 supervisor 시작 및 W&B remote readback은 부모가 수행한다. 이 구현 작업은 GPU/service/download를 시작하지 않으며 기존 pilot/control checkpoint를 변경하지 않는다. CPU fixture 테스트와 GPU 검증을 구분해야 한다.

- Calibration: BF16, EP8/TP1/PP1/CP1, sequence4096, GB128/MB1, 총16회 native optimizer 호출.
- Production 표시 이름: **nemochat baby midtraining super35 24h**. 실제 calibration 이전에는 iteration/WSD horizon이 null이다.
- 원본 `checkpoints/exports/nemochat_baby_base_hf_bf16_02`에서 model-only warmstart하고 optimizer/scheduler/RNG/cursor를 초기화한다. 학습된 운영 control student를 production 부모로 사용하지 않는다.
- Teacher는 Super3.5 VL **Base BF16**, revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`의 검토된 text view다. 기존 embedding-import/shape-guard/provider helper를 재사용한다.
- Native Bridge `pretrain`과 `forward_step_modelopt`가 Loss/backward/optimizer/scheduler/save를 소유한다. ModelOpt dynamic balancing은 `LM + scale*KD*stopgrad(LM/KD)`이며 raw0.5KL이 아니다. Native clip norm1, finite/positive Grad Norm, 실제 사용 LR을 기록한다. 큰 finite 수치를 임의 크기 제한으로 실패시키거나 LR을 매 step 덮어쓰지 않는다.

## 실제 데이터와 tokenizer 게이트

Producer `manifest.json`과 같은 디렉터리의 `success.json`을 함께 읽는다. `phase=prepared_verified`와 `success.result.manifest_sha256`의 실제 SHA256 일치를 요구한다. Source 순서, 60/10/10/20 fixed-sequence exposure, 길이4096, 서로 다른 train/validation native prefix, reader 검증 및 overlap/duplicate0을 검사한다. 자세한 producer schema는 [CORPUS.md](CORPUS.md)를 따른다.

컨테이너에도 저장소를 `/raid/yoon/nemochat`의 동일 절대 경로로 mount하므로 producer prefix를 바꾸지 않는다. **신규 train/validation 전체8개 .bin을 int32 streaming하여 ID18..29의 실제 등장 횟수를 전부 센다.** 동시에 vocabulary 범위, token 수, producer .bin SHA256과 .idx 크기/SHA256을 확인한다. Host audit는 `results/midtraining/longrun-corpus-audits/<run-id>/reserved-id-audit.json`, native rank별 fresh audit는 해당 실행 폴더에 저장한다. 실제 Base tokenizer asset SHA/repository/revision도 요구한다. ID18..29가 하나라도 있으면 명시적인 alias 의미 정책을 별도 검토하기 전까지 차단한다. **기존 ClimbMix ID부재 증거를 새 데이터에 재사용하지 않는다.** 기존 helper gate는 student/teacher embedding lineage 증거로만 남긴다.

실제 Bridge `GPTDatasetConfig` → native registry → `BlendedMegatronDatasetBuilder(GPTDataset,...)` 경로를 쓴다. Build 전 source별 weighted sample 수, 반올림 reserve, builder0.5% surplus 및 shifted-label 추가 token을 확인한다. 프로세스 내부 관찰 wrapper는 원래 native builder를 호출하고 종료 시 registry를 복원한다. Training document index가 두 epoch 이상이면 거부한다. 실제 첫 global batch와 전체 계획에 네 source가 모두 있어야 한다. 별도 blending/optimizer loop는 없다.

**10M pack은 calibration 용량이지 24시간 corpus가 아니다.** 16×128×4096은8,388,608 training positions다. 부족한 production source는 자동 반복하지 않고 Engine create 전에 차단한다. Calibration 측정 후 부모가 충분한 신규 recipe generation을 검토해야 한다. ClimbMix20%는 BASE가 이미 소비한 replay이며 새 고유 지식으로 주장하지 않는다.

Validation은 training과 분리된 heldout pool을 명시적으로 재사용한다. EvalGB8/MB1/1iteration이다. Native periodic+final sampler budget에 startup window 하나를 추가하고 양쪽 budget을 기록한다. Optimizer call1 직후 native in-loop heldout을 수행하므로 시작 Loss와 실제 LR 문맥을 관측한다. LR0 호출과 positive-LR learning update를 구분한다. 정상 horizon 완료에서는 native post-loop final validation만 사용하며 강제 final pass를 추가하지 않는다. Duration 조기 종료는 native in-loop validation을 요청한 뒤 native final save/exit로 간다. Eval 동안에만 native duration 제한을 잠시 해제하고 save/exit 판단 전에 복원한다. 외부 watchdog은 계속 유효하다. 두 경로 모두 실제 GPU 검증이 남아 있다.

## Calibration과 24시간 해석

CUDA synchronize로 native 전체 update 시간을 잰다. Rank8개의 최대 update 시간을 기록하고 rank별 peak allocated memory를 남긴다. Cold calls1..8을 제외하고 **unprofiled calls9..16 전체8회**의 mean/median/min/max/stdev를 사용한다. Diagnostic pilot 시간으로 대체하지 않는다. Teacher frozen/no-grad, finite Loss/positive Grad Norm, 사용 LR, 연속 optimizer 호출 및 LR0/positive-LR 수가 receipt에 들어간다.

Production horizon은 `floor(86400*0.90/mean_warm_step_seconds)`다. Save/eval/data overhead10%는 가정이지 장시간 실측이 아니다. WSD peak1e-5/min1e-6, warmupceil1%, 마지막 linear decayceil10%를 같은 실제 horizon에 고정한다. Runner-consumed YAML을 Engine argv에 직접 전달하며 native config, finalized config, teacher/KD config도 저장한다.

`exit_duration_in_mins=1440`은 **GlobalState.start_time부터 초기화 시간을 포함**한다. Learning-only24h가 아니다. Iteration budget이86400초 전에 끝나면 `iteration_budget_complete_short_of_wall_target`, `training_24h_accepted=false`다. 실제 경과 시간이 짧은 duration-complete 주장은 거부한다. W&B finished, config-only, Engine create/exit만으로 학습 완료를 주장할 수 없다.

## Save/reload, 보안, supervisor

Native synchronous full optimizer/RNG torch_dist save를250update마다 수행하고 recent3 및 final save를 설정한다. 비교 가능한 full checkpoint 약137GB를 기준으로 retained3+in-flight1+16GB 여유를 요구한다. 실제 retention 회전 검증은 별도다. 기존 output/checkpoint는 덮어쓰지 않고 새 run ID를 요구한다. 자동 retry/resume는 없다. **Full-state resume CLI는 아직 노출하지 않는다.** Native full state를 저장하지만 exact resume는 별도 검토 adapter가 필요하다.

정확한 Engine clean exit와8rank의 실제 update/final-heldout/full-state save 뒤에 별도 프로세스로 **teacher-free native reload/eval**을 실행한다. Reload에서 optimizer/save0 및 finite heldout이 확인되어야 `*_save_reload_accepted`다. Checkpoint integrity는 사용자 waiver인 `not_checked_explicit_user_waiver`를 유지하며 데이터 audit를 이유로 weight forensic을 재개하지 않는다.

Main은 이미 승인된 `nvidia/nemochat` online destination을 사용한다. 기존 secure parser로 read-only credential mount를 읽으며 API key를 argv/Engine Env/payload/config/log에 저장하지 않는다. Calibration/control은 offline이다. Run ID는 명시적이며 URL을 만들거나 remote readback을 수행했다고 주장하지 않는다. 부모 검증 전에는 `remote_tracking_verified=false`다.

기존 `results/pretraining/nemochat_baby/training/gpu-all8.lock`을 save/reload까지 보유하고 GPU0..7 idle/recovery 상태를 확인한다. Engine은 별도 cgroup이며 restart=no, bounded mounts/capabilities/memory다. Native24h+save grace30min, timeout kill grace 및 별도 reload30min은26h보다 작다. 부모 service는 RuntimeMaxSec93600, 독립 cgroup과 정확한 `ExecStopPost --cleanup-owned`를 설정해야 한다. Cleanup은 exact Engine ID/name/labels를 검증하고 그 컨테이너만 stop한 뒤 state를 다시 읽는다. Calibration은 native180min+save30min, reload30min, config-only600s다.

## 부모 실행 API

무부작용 확인:

```bash
.venv/bin/python midtraining/training/launch_longrun.py --help
.venv/bin/python midtraining/training/launch_longrun.py --preview
.venv/bin/python midtraining/training/longrun_native.py --preview
.venv/bin/python -m pytest tests/test_midtraining_longrun.py -q
```

부모가 실제 검토 후 `launch_longrun.SOURCES`와 정확히 같은 경로 집합의 SHA256 JSON map을 작성한다. Source가 바뀌면 review는 무효다. `NEMOCHAT_LONGRUN_SUPERVISED=1`을 가진 새 독립 부모 service에서:

```bash
.venv/bin/python midtraining/training/launch_longrun.py --config midtraining/configs/calibration.yaml \
  --run-id super35-calibration-01 --reviewed-sources /ABS/PARENT-REVIEW.json --execute
```

실제 config-only는 새 ID와 `--config-only`, native1min early-duration control은 또 다른 새 ID와 `--duration-control --execute`를 쓴다. 1분은 초기화 포함이므로 learning1min이 아니다. Timeout/nonzero는 실패다. Accepted 경로는 `results/midtraining/longrun/<run-id>/controller/accepted.json`이다.

Production은 `--config midtraining/configs/longrun24h.yaml`, `--manifest /ABS/PRODUCTION/manifest.json`, `--calibration /ABS/CALIBRATION/controller/accepted.json`, `--duration-control-receipt /ABS/CONTROL/controller/accepted.json`, `--calibration-compatibility /ABS/PARENT-COMPATIBILITY.json`, 새 `--run-id`, 검토 JSON과 `--execute`를 사용한다. 다른 manifest의 compatibility에는 실제 두 digest와 `status=parent_accepted_same_recipe_expansion`, `same_provider_tokenizer_sequence_batch_topology=true`, `heldout_exclusions_preserved=true`가 정확히 필요하다. 이는 expanded data extrapolation을 명시하는 것이지 sustained24h 증거가 아니다.

ExecStopPost: `.venv/bin/python midtraining/training/launch_longrun.py --run-id EXACT-RUN --cleanup-owned`. 부모는 MainPID/argv, exact Engine, 증가하는 native update 및 실제 remote run을 readback한 뒤에만 시작/학습 중이라고 보고해야 한다.

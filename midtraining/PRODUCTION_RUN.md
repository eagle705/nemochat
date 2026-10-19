# Production CPU → native main handoff

[한국어](PRODUCTION_RUN.ko.md)

Generation02 omits explicit `User=` in `systemd-run --user`. Generation01 failed before Python execution with `216/GROUP` because the user manager could not set supplementary groups; its source snapshot/journal are preserved. The real `/proc` UID, MainPID and complete argv remain mandatory even when the unit's `User` property is empty. No corpus writer or native GPU job was restarted for this controller-only correction.

`run_production.py` is an executable contingent controller, not training evidence.
Default invocation, `--preview`, and `--help` only print information: no output
files, Engine requests, services, GPU work, downloads, or tracking registration.
Only the parent reviews/freezes sources and starts the already-authorized service.
Do not edit any reviewed source while the controller is armed.

## Fixed scope and acceptance

- Existing CPU producer: `9e0861343af8efa265c622623ff78714109132a1cd48cc59c4a2731ec456bab5`,
  name `nemochat-midtraining-production-prepare-01`. Its complete approved identity/payload
  receipt is SHA256-pinned in the controller. The controller never starts, retries,
  stops, or replaces this prerequisite.
- One blocking Engine `POST /containers/<exact-id>/wait?condition=not-running`,
  bounded by the remaining 12 hours from **October 17, 2026, 11:31:59 UTC** plus
  300 seconds grace. Exact ID/name/task-owned `nemochat.*` labels and `Config.Image`
  / `Config.Cmd` are checked before and after. Inherited NVIDIA labels are permitted.
  Require wait status zero, final exit zero, not running, and `OOMKilled=false`.
- Production manifest:
  `datasets/midtraining/super35-kd-01/production-24h-v1/data/manifest.json`.
  Native companion-success SHA validation is mandatory, not artifact polling.
- Preserve the entire prior heldout reservation (source/content hash/group ID),
  including unused reserved documents, and all four frozen validation `.bin/.idx`
  digests/sizes. Original seed manifest SHA is
  `52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308`.
- Run native per-source capacity at **3200 updates, GB128, sequence4096**, original
  weights `[0.6, 0.1, 0.1, 0.2]`, no cycling. Fresh native reserved-ID audit scans
  all eight train/validation `.bin` files and verifies their digests/lengths and
  every `.idx`; IDs 18–29 must have zero counts. Compare complete tokenizer identity.
- Explicit parent-approved same-recipe expansion policy: compare accepted calibration
  consumed configuration (bound to its frozen config SHA) with production BASE,
  teacher, precision/parallel topology, sequence/batch, optimizer/DDP/KD, RNG,
  source weights and checkpoint policy. Only stage/name, dataset locator, logger,
  duration/watchdog, WSD horizons, validation cadence and save destination vary.
  The exact five-field compatibility dictionary is derived only after these checks.
  This is a **projection from the same recipe**, not a new-manifest calibration or
  proof of sustained 24-hour training. Native calibration measured mean warm update
  time 24.29540627438837 seconds; the native horizon derivation assumes 10% overhead.
- Main is `super35-longrun-01`, online `nvidia/nemochat`, using the **original BASE
  model-only reset**, never calibration-trained weights. Existing explicit student
  checkpoint byte-integrity waiver remains; no checkpoint hashes are added.
  Keys remain private to existing native/launcher code, never argv/config/log entries.
- Supervisor bound **140400 seconds (39 hours)** covers the 12-hour CPU prerequisite
  and a complete **93600-second (26-hour)** main subprocess budget plus orchestration
  room. It is not a 39-hour training request. Insufficient remaining main budget blocks
  release. Native training target remains 86400 seconds with its existing watchdog.

## Two independent source reviews

The controller startup review is a plain `{relative_path: sha256}` mapping for the
exact union of existing launcher's 14 `SOURCES`, producer's six `REVIEW_FILES`, and
`midtraining/run_production.py` (overlaps deduplicated). It is **not** passed to the
native launcher. The launcher still receives its unchanged 14-file review:
`results/midtraining/super35-kd-01/longrun-source-review-02.json`.
Producer's separate existing six-file review also remains unchanged.

The parent creates the startup review **after** reviewing the finished four-file
handoff. Example below is parent-only; it refuses to overwrite an earlier freeze:

```bash
cd /raid/yoon/nemochat
.venv/bin/python - <<'PY'
import hashlib, json, pathlib, subprocess
root = pathlib.Path.cwd()
p = json.loads(subprocess.check_output([
    str(root/'.venv/bin/python'), str(root/'midtraining/run_production.py'), '--preview'
], text=True))
review = root/'results/midtraining/super35-kd-01/production-controller-source-review-02.json'
with review.open('x') as stream:
    json.dump({name: hashlib.sha256((root/name).read_bytes()).hexdigest()
               for name in p['source_paths']}, stream, indent=2)
    stream.write('\n')
print(review)
PY
```

No startup checksum artifact is generated by this handoff itself. Sources are
checked before the CPU wait, after it, before data/main release and immediately
before launching. Frozen prerequisite receipts are compared again before release.
Missing, extra, duplicate-key, or changed source entries fail closed and are durably
recorded after supervisor ownership/fresh-output checks.

## Exact parent launch

First run the focused checks, inspect `--preview`, and confirm no existing queued
production service or `super35-longrun-01` output/checkpoint/audit directory. Never
remove historical outputs to force a retry. The parent must independently review
this command before running it:

```bash
cd /raid/yoon/nemochat
.venv/bin/python -m pytest -q tests/test_midtraining_production_pipeline.py \
  tests/test_midtraining_longrun.py tests/test_midtraining_production_corpus.py
.venv/bin/python midtraining/run_production.py --preview
systemd-run --user --unit=nemochat-midtraining-production-02.service \
  --property=Type=exec \
  --property=WorkingDirectory=/raid/yoon/nemochat \
  --property=RuntimeMaxSec=140400 --property=KillMode=control-group \
  --property=TimeoutStopSec=300 --property=Restart=no \
  '--property=ExecStopPost=/raid/yoon/nemochat/.venv/bin/python /raid/yoon/nemochat/midtraining/run_production.py --cleanup-owned --reviewed-sources /raid/yoon/nemochat/results/midtraining/super35-kd-01/production-controller-source-review-02.json' \
  --property=StandardOutput=append:/raid/yoon/nemochat/results/midtraining/longrun/super35-production-02-supervisor.log \
  --property=StandardError=append:/raid/yoon/nemochat/results/midtraining/longrun/super35-production-02-supervisor.log \
  --setenv=NEMOCHAT_PRODUCTION_SUPERVISED=1 --setenv=PYTHONUNBUFFERED=1 \
  --setenv=PYTHONDONTWRITEBYTECODE=1 \
  /raid/yoon/nemochat/.venv/bin/python /raid/yoon/nemochat/midtraining/run_production.py \
  --execute --reviewed-sources /raid/yoon/nemochat/results/midtraining/super35-kd-01/production-controller-source-review-02.json
```

`--preview` includes the complete machine-readable `supervisor_argv` and `main_argv`.
`supervisor_command(review)` constructs the same command using the caller's actual
UID. The explicit example uses the approved producer/user UID 1001. The service must
report the controller's exact MainPID, UID, and full interpreter/script/argument
vector; an environment flag alone is insufficient. Do not add interpreter `-u`:
use the buffering environment already supplied. This is independent user-systemd
ownership; it neither restarts the producer nor changes any prerequisite service.

After all gates pass, the separate main command is precisely:

```bash
NEMOCHAT_LONGRUN_SUPERVISED=1 /raid/yoon/nemochat/.venv/bin/python \
  /raid/yoon/nemochat/midtraining/launch_longrun.py \
  --config /raid/yoon/nemochat/midtraining/longrun24h.yaml \
  --run-id super35-longrun-01 \
  --manifest /raid/yoon/nemochat/datasets/midtraining/super35-kd-01/production-24h-v1/data/manifest.json \
  --calibration /raid/yoon/nemochat/results/midtraining/longrun/super35-newdata-calibration-01/controller/accepted.json \
  --duration-control-receipt /raid/yoon/nemochat/results/midtraining/longrun/super35-newdata-duration-01/controller/accepted.json \
  --calibration-compatibility /raid/yoon/nemochat/results/midtraining/longrun/super35-production-02/controller/calibration-compatibility.json \
  --reviewed-sources /raid/yoon/nemochat/results/midtraining/super35-kd-01/longrun-source-review-02.json \
  --execute
```

Do not run this directly while the contingent controller is armed. The controller
passes the environment internally and owns the finite child-process wait.

## Readback, stop and honest completion

Immediately after parent start, read back the exact unit and `/proc/<MainPID>/cmdline`:

```bash
systemctl --user show nemochat-midtraining-production-02.service \
  --property=MainPID,User,ActiveState,SubState,ExecStart,ExecStopPost,RuntimeMaxUSec,KillMode
.venv/bin/python -c 'import json,pathlib; print(json.dumps(json.loads(pathlib.Path("results/midtraining/longrun/super35-production-02/controller/state.json").read_text()),indent=2))'
```

Durable state and fsynced phase history live under
`results/midtraining/longrun/super35-production-02/controller/`:
`WAITING_CORPUS` → `VALIDATING_DATA` → `NATIVE_MAIN_CONTROLLER_RUNNING` →
`FINISHED`, or `FAILED`. A running controller is **not** evidence of learning;
inspect all-rank native update receipts and independently read back the exact remote
W&B run/name/update counters after main starts. The controller does not register
auxiliary runs or claim remote tracking verification.

SIGTERM/INT records failure; service `KillMode=control-group` propagates stop to the
launcher, and `ExecStopPost --cleanup-owned` delegates only to the existing exact-ID,
name and task-label checked cleanup for this main's config/train/reload containers.
Without a recorded main release cleanup does nothing. It never stops the CPU
producer, prerequisite services, foreign jobs, or resets GPUs. No automatic retry,
resume, SFT or hardware mutation exists.

`FINISHED` requires the separate launcher's actual
`production_save_reload_accepted` result with matching manifest and all eight ranks.
Its `training_24h_accepted` boolean is preserved: a 3200-iteration finish short of
native 86400 seconds remains **false**, not relabeled a 24-hour success. CPU synthetic
ordering tests (including a real tiny child process) are not GPU, capacity-materialization,
quality, or sustained-duration evidence. Actual production capacity is known only
after the real producer and downstream audits succeed.

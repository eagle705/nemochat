# Dashboard execution verification

## Delivered scope

Read-only Python standard-library monitoring server and local Korean HTML/CSS/JavaScript frontend under `dashboard/`. The server reads explicitly allowlisted training artifacts and samples host CPU/RAM/GPU telemetry, independently of browser clients. Default five-second collection, cached SSE snapshots, polling fallback, bounded histories/reads/SQLite queries, loopback and host/origin checks. Configuration can be reloaded with `--reload-config`.

Stage profile selectors cover pretrain / mid / SFT / long-context / RLVR / MOPD. Only pretraining currently has a configured native capture. Educational Lens/bridge artifacts are available through the **unscoped/common import** selector rather than relabeled as native trainer instrumentation. Recommendations separate observed scope, candidate hypotheses and correctness + unprofiled full-update A/B validation. No automatic changes to training or profiling.

## Parent execution, 2026-10-13 KST

```bash
cd /raid/yoon/nemochat
.venv/bin/python -m pytest -q dashboard/tests
python3 -m compileall -q dashboard
node --check dashboard/static/app.js
git diff --check
.venv/bin/python dashboard/server.py --root . --config dashboard/config.json --port 8787 --interval 5 --reload-config
```

- Dashboard suite: **49 passed** in the parent rerun. Tests cover real loopback HTTP/SSE, shared collector cache, client bounds, secure static routes, config reload, artifact counter semantics, SQLite read-only/cache/privacy, six-stage profiles, DOM renderer/transport and live-browser regressions.
- Parent HTTP read-back: eight configured runs, eight GPUs. Actual Bridge pilot totals: **4,096 processed / 4,096 loss-bearing / 4,128 source-prefix reads**. Rank losses/counters are not blindly multiplied. Failed producer and successful recovery are separate runs.
- Parameter total **13,772,670,192** is from the explicitly configured earlier native A census, labeled as prior model evidence rather than a fresh numel measurement in the selected run. Active parameters remain unknown when not supplied by a trustworthy record.
- Update-duration accumulation is labeled separately from container elapsed time. The Bridge pilot's recorded container duration is **427.567018 seconds**. Overlapping/resumed runs are not summed as a project-wide total.
- Two consecutive live SSE snapshots have different generated timestamps; exact timestamps and collection costs are in `results/dashboard-live-final.json` (final server) and `results/dashboard-live-verification.json` (earlier live read-back). Observed collection costs are monitor CPU/I/O costs, **not measured training slowdown**.
- Native Nsight SQLite: **631,489 whole-capture kernels / 268 preserved names**. Separate producer-selected updates retain **7,025 / 6,834** correlated counts. Startup/warmup, rank-zero coverage and instrumentation limits remain visible. Selected correlation and producer provenance are not recomputed/authenticated by the viewer.
- Separate sparse-MoE Lens control: **21 spans / 21 bridge operations**. No native Lens integration or source attribution claim.

## Real browser verification

The parent opened the running server in Chrome, inspected the visible screen, selected runs/profile stages and Nsight tabs, verified unavailable RLVR capture state and pause/resume, and checked responsive layout. Desktop 1440×1080 and mobile 390×844 had no document-wide horizontal overflow. Captured-window error/unhandled-rejection checks during interactions returned no errors. A browser-helper evaluation that intentionally waited 6.5 seconds hit its own five-second IPC timeout; HTTP/SSE read-back was independently repeated successfully rather than treating that timeout as app evidence.

The first actual screenshot revealed missing Korean glyphs on the minimal Linux browser host. Added an OFL-licensed self-hosted, renamed Noto Sans KR subset, a strictly whitelisted font route and regression coverage. Font readiness and legible glyphs were verified in actual screenshots. Upstream revision/license/hash: `static/FONT-PROVENANCE.json`, `static/FONT-OFL.txt`. No runtime font CDN requests or system package changes.

Nested Nsight JSON initially obscured kernels. Replaced it with scalar scope facts, selected-update tables and collapsible phase details; kernel tables have bounded internal scrolling. DOM regression checks exercise the structured view.

Real captures (ignored runtime evidence):
- `results/dashboard-overview.png`
- `results/dashboard-profiling.png`
- `results/dashboard-optimizations.png`
- `results/dashboard-mobile.png`

## Access and remaining boundaries

Local server: `http://127.0.0.1:8787` on **dgx-b300**. Remote workstation access requires SSH forwarding:

```bash
ssh -N -L 8787:127.0.0.1:8787 yoon@dgx-b300
```

Use the actual hostname/IP accepted by your workstation if `dgx-b300` does not resolve. Then open the same loopback URL on the workstation. This service has no public-facing authentication/TLS and is not exposed externally.

No trainer/optimizer/checkpoint/default dependency changes; no GPU training/capture/collector startup, remote W&B fetch, credential reads, commit or publication. A fresh native capture for another stage still needs its own authorized diagnostic workflow and explicit config entry. Broad root GPU regression was not rerun for this isolated dashboard work. This is not whole-node bottleneck proof or a zero-overhead guarantee. The running server is a background process, not an installed systemd/autostart service.

## 한국어 요약

- 실제 서버 접속, 서로 다른 두 SSE 갱신, 8개 실행·8개 GPU, Bridge 토큰 4096/4096/4128을 직접 확인했습니다.
- 대시보드 테스트 49개 통과, JS/Python 문법 및 공백 검사 통과. 실제 데스크톱·모바일 화면과 선택/일시정지 동작을 확인했습니다.
- 단계별 프로파일은 6단계 선택이 가능하지만 현재 native capture는 사전학습만 있습니다. 다른 단계는 미수집, 교육용 Lens 자료는 공통 가져오기에서 분리합니다.
- 모델 크기는 이전의 명시적 native census 근거이며 현재 실행에서 다시 세었다고 표시하지 않습니다. 업데이트 누적 시간과 컨테이너 전체 시간을 분리합니다.
- 실제 스크린샷과 live 검증 receipt는 위 `results/` 경로에 저장했습니다. 새 학습 실행/캡처는 허용 목록에 추가해야 합니다.

# NemoChat project skills / 프로젝트 스킬

These versioned procedures belong to the project, not a personal Hermes profile. Canonical `SKILL.md` files are English; this index provides Korean guidance. They encode reproducible engineering procedures, not historical test counts or a second implementation of the code.

| Skill | 적용 시점 |
|---|---|
| [Experiment validation](nemochat-experiment-validation/SKILL.md) | YAML 실험·성능 A/B·GPU 성능 주장 검증 |
| [Checkpoint resume](nemochat-checkpoint-resume/SKILL.md) | checkpoint 보존·전송·정확한 재개 상태 검증 |
| [Observability validation](nemochat-observability-validation/SKILL.md) | Lens trace·Jaeger·rollout 병목·bridge source evidence 검증 |

## Cross-agent use / 다른 에이전트에서 사용

`AGENTS.md` directs matching tasks to these files. An agent must actually read the applicable `SKILL.md`; placing it here alone does not guarantee automatic discovery in every harness. Use native file/terminal tools instead of assuming Hermes-specific APIs exist. Missing tools, dependencies and unverifiable results must be reported explicitly.

Hermes can be explicitly told to read a root skill file. If automatic catalog discovery is desired, its documented `skills.external_dirs` setting can include the **absolute path of this checkout's `skills/` directory**. No personal configuration or trust settings are changed by adding these files. Alternatively, install a reviewed copy into the target harness's supported skill directory. Do not create multiple divergent copies of the same workflow; choose a source of truth.

프로젝트에는 개인 memory·세션·API key를 복사하지 않는다. 스킬은 Git으로 공유 가능한 절차 문서이고, 실제 데이터·checkpoint·실험 산출물·로그는 `datasets/`, `checkpoints/`, `results/`, `logs/`에서 계속 Git 제외한다. 스킬을 읽었다고 commit·push·외부 upload·유료 GPU 실행 권한이 생기는 것은 아니다.

## Maintenance / 유지보수

- Add a skill only for a recurring project-specific workflow with concrete checks.
- Verify commands with actual CLI help/execution and keep source/docs authoritative.
- Keep triggers concise; move large reference material into the owning skill folder when needed.
- Preserve CPU/offline versus native GPU/distributed/quality boundaries.
- Use repo-relative paths. The initial examples target the tested Linux layout; other OS/interpreter layouts need explicit adaptation, not assumed compatibility.
- Review newly shared skills as executable instructions and honor upstream licenses if importing third-party content. These initial workflows were authored for NemoChat, not copied wholesale from the private skill catalog.

Validation: `.venv/bin/python -m pytest -q tests/test_project_skills.py`.

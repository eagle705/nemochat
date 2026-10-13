"""Explicit, optional stage tracking. No SDK import when disabled/nonzero rank."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import uuid

STAGES = ("data", "pretrain", "midtrain", "longcontext", "sft", "rlvr", "mopd")
RUN_TYPES = ("stage", "evaluation", "checkpoint-merge", "schedule-plan")
TRACKING_OPTIONS = frozenset({"run_id", "entity", "project", "group", "parent_id",
                              "checkpoint_id", "directory"})


def sanitize(value):
    """Filter credential keys, not tokenizer/token-budget configuration."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            secret = (normalized in {"token", "auth", "authorization", "password", "secret", "credentials"}
                      or normalized.endswith(("_api_key", "_password", "_secret", "_access_token", "_auth_token"))
                      or normalized in {"api_key", "hf_token", "wandb_api_key", "access_token", "auth_token", "bearer_token", "refresh_token", "client_secret"})
            result[str(key)] = "[REDACTED]" if secret else sanitize(item)
        return result
    if isinstance(value, (tuple, list)):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@contextmanager
def preserve_rng():
    """Preserve Python, imported NumPy legacy and torch CPU RNG only."""
    python_state = random.getstate()
    torch = sys.modules.get("torch")
    numpy = sys.modules.get("numpy")
    torch_state = torch.get_rng_state().clone() if torch else None
    numpy_state = numpy.random.get_state() if numpy else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


def flat_metrics(metrics, prefix=""):
    """Flatten scalar metrics for charts; keep large raw result arrays in local reports."""
    result = {}
    for key, value in metrics.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            result.update(flat_metrics(value, name + "/"))
        elif value is None or isinstance(value, (str, int, float, bool)):
            result[name] = value
    return result


def hf_reference(repository, commit):
    """Immutable metadata reference only; no implicit weight upload/download."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        raise ValueError("HF metadata references require an immutable 40-character commit")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?", repository):
        raise ValueError("Invalid HF repository ID")
    return {"repository": repository, "commit": commit,
            "url": f"https://huggingface.co/{repository}/tree/{commit}",
            "weights_uploaded": False, "reference_only": True}


def add_tracking_args(parser):
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"),
                        default=os.environ.get("NEMOCHAT_WANDB_MODE", "disabled"))
    parser.add_argument("--wandb-config", type=Path, help="JSON tracking defaults; CLI flags override")
    for name in ("id", "entity", "project", "group", "parent-id", "checkpoint-id"):
        parser.add_argument("--wandb-" + name)
    parser.add_argument("--wandb-dir", type=Path)
    parser.add_argument("--wandb-resume", action="store_true", help="Online resume=must; offline requires explicit ID")


class StageRun:
    """Small init/config/log/finish wrapper; exceptions remain failures."""
    def __init__(self, stage, *, mode="disabled", run_type="stage", run_id=None,
                 entity=None, project="nemochat", group=None, parent_id=None,
                 checkpoint_id=None, directory="outputs/wandb", resume=False, config=None):
        if stage not in STAGES or run_type not in RUN_TYPES:
            raise ValueError("Unknown stage/run type")
        if mode not in ("disabled", "offline", "online"):
            raise ValueError("Unknown tracking mode")
        if resume and not run_id:
            raise ValueError("Resume tracking requires explicit --wandb-id (including offline)")
        if run_id and not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Unsafe W&B run ID")
        self.mode = mode
        self.id = run_id or uuid.uuid4().hex[:12]
        self.directory = Path(directory)
        self.run = None
        self.started = None
        self.metadata = sanitize(dict(stage=stage, run_type=run_type, run_id=self.id, entity=entity,
                             project=project, group=group, parent_run_id=parent_id,
                             checkpoint_id=checkpoint_id, mode=mode))
        self.config = sanitize(config or {})
        self.resume = resume
        self.active = mode != "disabled" and int(os.environ.get("RANK", "0")) == 0
        self.receipt = None
        self._failed = False

    def __enter__(self):
        return self.init()

    def init(self):
        if self.run:
            raise RuntimeError("Tracking run is already initialized")
        if not self.active:
            return self
        if self.mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise ValueError("Explicit online mode requires WANDB_API_KEY; no interactive login")
        self._validate_continuity()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.receipt = self.directory / f"tracking-{self.id}-{uuid.uuid4().hex[:8]}.json"
            self._receipt("initializing")
            with preserve_rng():
                import wandb
                self.run = wandb.init(mode=self.mode, id=self.id,
                    entity=self.metadata["entity"], project=self.metadata["project"],
                    group=self.metadata["group"], job_type=self.metadata["run_type"],
                    dir=str(self.directory), config={**self.config, "tracking": self.metadata},
                    resume=("must" if self.resume else "never") if self.mode == "online" else None,
                    settings=wandb.Settings(disable_git=True, disable_code=True,
                                            x_disable_stats=True))
            if self.run is None:
                raise RuntimeError("W&B initialization returned no run")
            self.started = time.monotonic()
            self._receipt("running")
        except BaseException as exc:
            self._record_failure(exc)
            self._failed_finish(exc)
            raise
        return self

    def _validate_continuity(self):
        # Only available local receipts are evidence; do not infer remote history.
        if self.resume:
            for path in self.directory.glob(f"tracking-{self.id}-*.json"):
                if not re.fullmatch(rf"tracking-{re.escape(self.id)}-[0-9a-f]{{8}}\.json", path.name):
                    continue  # A different run ID may share this ID's filename prefix.
                prior = json.loads(path.read_text())
                fields = ("run_id", "stage", "run_type", "entity", "project")
                if not isinstance(prior, dict) or any(
                    prior.get(key) != self.metadata[key] for key in fields
                ):
                    raise ValueError("Tracking resume continuity conflicts with local receipt; "
                                     "use the original stage/type/entity/project or a new --wandb-id")

    def _record_failure(self, original):
        self._failed = True
        try:
            self._receipt("failed")
        except BaseException:
            original.add_note("Tracking failed-receipt write also failed; receipt may be stale.")

    def _failed_finish(self, original):
        try:
            self.finish(failed=True)
        except BaseException:
            original.add_note("Tracking failed-finish cleanup also failed.")

    def _receipt(self, status):
        if self.receipt:
            payload = {**self.metadata, "status": status,
                       "sdk_run_dir": self.run.dir if self.run else None}
            self.receipt.write_text(json.dumps(payload, indent=2) + "\n")

    @contextmanager
    def _operation(self):
        try:
            with preserve_rng():
                yield
        except BaseException as exc:
            self._record_failure(exc)
            raise

    def log(self, metrics, step=None):
        if self.run:
            with self._operation():
                clean = sanitize(metrics)
                json.dumps(clean, allow_nan=False)
                self.run.log(flat_metrics(clean), step=step)

    def update_config(self, config):
        if self.run:
            with self._operation():
                clean = sanitize(config)
                json.dumps(clean, allow_nan=False)
                self.run.config.update(clean, allow_val_change=True)

    def artifact_metadata(self, name, *, kind, metadata):
        """Metadata-only artifact: never upload weights or download HF references."""
        if self.run:
            with self._operation():
                import wandb
                clean = sanitize(metadata)
                json.dumps(clean, allow_nan=False)
                artifact = wandb.Artifact(name, type=kind, metadata=clean)
                self.run.log_artifact(artifact)

    def finish(self, failed=False):
        failed = failed or self._failed
        self._failed = bool(failed)
        if self.run:
            try:
                with preserve_rng():
                    if self.started is not None and not failed:
                        self.run.summary["wall_seconds_this_invocation"] = time.monotonic() - self.started
                    self.run.finish(exit_code=1 if failed else 0)
                self._receipt("failed" if failed else "finished")
            except BaseException as exc:
                self._record_failure(exc)
                raise
            finally:
                self.run = None

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self._record_failure(exc)
            self._failed_finish(exc)
        else:
            self.finish()
        return False


def from_args(args, stage, *, run_type="stage", config=None):
    options = {}
    if args.wandb_config:
        options = json.loads(args.wandb_config.read_text())
        if not isinstance(options, dict):
            raise ValueError("Tracking config must be a JSON object")
        if options.keys() - TRACKING_OPTIONS:
            # Never echo arbitrary supplied keys or values (either can contain secrets).
            raise ValueError("Unknown/reserved tracking config option. Allowed keys: "
                             + ", ".join(sorted(TRACKING_OPTIONS))
                             + ". Use --wandb-mode/--wandb-resume for activation/resume; "
                             "keep credentials in environment variables, not JSON.")
        for key, value in options.items():
            nullable = key not in {"project", "directory"}
            if not ((nullable and value is None) or (isinstance(value, str) and value.strip())):
                raise ValueError(f"Tracking option {key} must be a nonempty string"
                                 + (" or null" if nullable else ""))
    mapping = {"id": "run_id", "dir": "directory", "parent_id": "parent_id"}
    for name in ("id", "entity", "project", "group", "parent_id", "checkpoint_id", "dir"):
        value = getattr(args, "wandb_" + name)
        if value is None:
            value = os.environ.get("NEMOCHAT_WANDB_" + name.upper())
        if value is not None:
            options[mapping.get(name, name)] = value
    checkpoint_resume = bool(getattr(args, "resume", None)) and args.wandb_mode != "disabled"
    return StageRun(stage, mode=args.wandb_mode, run_type=run_type,
                    resume=args.wandb_resume or checkpoint_resume, config=config, **options)

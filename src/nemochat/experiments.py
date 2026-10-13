"""Strict, bounded offline dense-control experiment contracts and execution."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from types import MappingProxyType
from typing import Mapping

MAX_CONFIG_BYTES = 32_768
MAX_SECONDS = 3600
EXIT_CODES = {"success": 0, "failed": 1, "invalid": 2, "blocked": 3, "timeout": 4}
EVIDENCE = "custom_offline_dense_control; native hybrid NOT validated"
FIELDS = {
    "experiment": {"name", "seed"},
    "workload": {"device", "dtype", "batch_size", "sequence_length", "hidden_size", "ffn_size", "layers", "vocab_size", "learning_rate", "activation_checkpointing"},
    "measurement": {"warmup_steps", "measured_steps", "repeats", "profile"},
    "resources": {"max_seconds", "max_gpu_count"},
}


class ContractError(ValueError):
    """An input is outside the experiment contract."""


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError(f"{label} requires exactly these keys: {', '.join(sorted(expected))}")


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ContractError(f"{label} must be an integer in [{low}, {high}]")


def validate_config(config):
    """Validate all fields without importing torch or checking hardware."""
    _keys(config, {"schema_version", *FIELDS}, "root")
    _integer(config["schema_version"], 1, 1, "schema_version")
    for section, fields in FIELDS.items():
        _keys(config[section], fields, section)
    e, w, m, r = (config[k] for k in FIELDS)
    if not isinstance(e["name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", e["name"]):
        raise ContractError("experiment.name must be 1-64 ASCII letters/digits/_.-, starting with a letter/digit")
    _integer(e["seed"], 0, 2**32 - 1, "experiment.seed")
    if w["device"] not in ("cpu", "cuda") or w["dtype"] not in ("float32", "bfloat16"):
        raise ContractError("device must be cpu/cuda and dtype float32/bfloat16")
    if w["device"] == "cpu" and w["dtype"] != "float32":
        raise ContractError("CPU bfloat16 is not supported by this contract")
    for key, upper in {"batch_size": 64, "sequence_length": 4096, "hidden_size": 4096, "ffn_size": 16384, "layers": 24, "vocab_size": 65536}.items():
        _integer(w[key], 2 if key == "vocab_size" else 1, upper, f"workload.{key}")
    if w["hidden_size"] % 4:
        raise ContractError("hidden_size must be divisible by 4 (bounded-control contract; model uses one attention head)")
    lr = w["learning_rate"]
    if type(lr) not in (float, int) or not 0 < lr <= 1 or not math.isfinite(lr):
        raise ContractError("learning_rate must be a finite number in (0, 1]")
    for section, key in ((w, "activation_checkpointing"), (m, "profile")):
        if type(section[key]) is not bool:
            raise ContractError(f"{key} must be a boolean")
    for key, low, upper in (("warmup_steps", 0, 100), ("measured_steps", 1, 1000), ("repeats", 1, 20)):
        _integer(m[key], low, upper, f"measurement.{key}")
    if (m["warmup_steps"] + m["measured_steps"]) * m["repeats"] > 2000:
        raise ContractError("total warmup+measured steps across repeats must be <= 2000")
    _integer(r["max_seconds"], 1, MAX_SECONDS, "resources.max_seconds")
    _integer(r["max_gpu_count"], 0, 1, "resources.max_gpu_count")
    if (w["device"] == "cuda") != (r["max_gpu_count"] == 1):
        raise ContractError("cuda requires GPU budget 1; cpu requires GPU budget 0")
    b, s, h, f, l, v = (w[k] for k in ("batch_size", "sequence_length", "hidden_size", "ffn_size", "layers", "vocab_size"))
    if l * (4 * h * h + 2 * h * f) + 2 * v * h > 100_000_000:
        raise ContractError("estimated matrix parameters exceed 100,000,000")
    # Fourfold single-head attention element bound is a conservative safety guard.
    if b * s * v > 16_777_216 or b * s * max(h, f) * l > 16_777_216 or b * 4 * s * s * l > 16_777_216:
        raise ContractError("logit, activation or conservative attention safety element budget exceeds 16,777,216")
    return config


def load_strict_json(path, *, max_bytes):
    """Bound local benchmark inputs; reject ambiguous or non-finite JSON."""
    def unique_mapping(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ContractError(f"non-finite JSON number: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ContractError("non-finite JSON number")
        return number

    with Path(path).open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ContractError(f"JSON input exceeds {max_bytes} bytes")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_mapping,
                          parse_constant=reject_constant, parse_float=finite_float)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ContractError(f"invalid JSON: {exc}") from exc


def _freeze(value):
    return MappingProxyType({k: _freeze(v) if isinstance(v, dict) else v for k, v in value.items()})


def _thaw(value):
    return {k: _thaw(v) if isinstance(v, Mapping) else v for k, v in value.items()}


@dataclass(frozen=True)
class ResolvedConfig:
    config: Mapping
    source_sha256: str

    def to_dict(self):
        return _thaw(self.config)


def load_config(path):
    """Read once, reject YAML references/tags/duplicates, freeze validated values."""
    import yaml
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ContractError(f"YAML input exceeds {MAX_CONFIG_BYTES} bytes")
    class StrictLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ContractError("YAML mapping keys must be unique strings")
            result[key] = loader.construct_object(value_node)
        return result

    StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        text = raw.decode("utf-8")
        for token in yaml.scan(text):
            if isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken, yaml.tokens.TagToken)):
                raise ContractError("YAML aliases, anchors and explicit tags are prohibited")
        depth = 0
        for event in yaml.parse(text):
            if isinstance(event, (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent)):
                depth += 1
                if depth > 8:
                    raise ContractError("YAML nesting exceeds 8 levels")
            elif isinstance(event, (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent)):
                depth -= 1
        config = yaml.load(text, Loader=StrictLoader)
    except (yaml.YAMLError, UnicodeError, RecursionError, ValueError) as exc:
        raise ContractError(f"invalid YAML: {exc}") from exc
    return ResolvedConfig(_freeze(validate_config(config)), hashlib.sha256(raw).hexdigest())


def benchmark_argv(output, repository):
    output = Path(output).resolve()
    return [sys.executable, str(Path(repository).resolve() / "scripts/benchmark_training.py"), "--config", str(output / "resolved_config.json"), "--output", str(output / "measurement")]


def _terminate_group(process):
    """Kill the entire session, including descendants of an exited group leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute_process(argv, *, cwd, env, timeout, stdout, stderr):
    """Return (returncode, timed_out); never invoke a shell."""
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, shell=False, start_new_session=True)
    try:
        return process.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        _terminate_group(process)
        return process.returncode, True
    except BaseException:
        _terminate_group(process)
        raise


def _preflight(config):
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed"
    if config["workload"]["device"] == "cuda":
        if not torch.cuda.is_available():
            return "CUDA is not available"
        if config["workload"]["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            return "CUDA bfloat16 is not supported"
    return None


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_experiment(resolved, output, repository):
    """Exclusively reserve a fresh output and persist honest execution evidence."""
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    # Logs are separate from result bundles; unique names avoid basename collisions.
    logs = Path(repository).resolve() / "logs" / "experiments" / uuid.uuid4().hex
    logs.mkdir(parents=True, exist_ok=False)
    config = resolved.to_dict()
    _write_json(output / "resolved_config.json", config)
    (output / "resolved_config.json").chmod(0o444)
    argv = benchmark_argv(output, repository)
    started = time.monotonic()
    status, reason, returncode = "failed", None, None
    try:
        reason = _preflight(config)
        if reason:
            status = "blocked"
        else:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "" if config["workload"]["device"] == "cpu" else env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled"})
            remaining = config["resources"]["max_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                status, reason = "timeout", "resource deadline exhausted during preflight"
            else:
                with (logs / "stdout.log").open("x") as stdout, (logs / "stderr.log").open("x") as stderr:
                    returncode, timed_out = execute_process(argv, cwd=repository, env=env, timeout=remaining, stdout=stdout, stderr=stderr)
                status = "timeout" if timed_out or returncode == 4 else "success" if returncode == 0 else "blocked" if returncode == 3 else "failed"
                if status != "success":
                    reason = "resource deadline exceeded" if timed_out else "benchmark cooperative time budget exceeded" if returncode == 4 else f"benchmark exited {returncode}"
                    # Typed worker failures are emitted on stdout, not necessarily stderr.
                    if not timed_out:
                        try:
                            summary = load_strict_json(logs / "stdout.log", max_bytes=MAX_CONFIG_BYTES)
                        except (ValueError, OSError):
                            summary = None
                        if isinstance(summary, dict):
                            kind = summary.get("failure_kind")
                            if returncode == 3 and kind == "out-of-memory":
                                reason = "benchmark out-of-memory"
                            elif returncode == 4 and kind == "time-budget-exceeded":
                                reason = "benchmark time-budget-exceeded (cooperative deadline)"
                    with (logs / "stderr.log").open("rb") as stream:
                        stream.seek(max(0, (logs / "stderr.log").stat().st_size - 65536))
                        error = stream.read().decode("utf-8", errors="replace").lower()
                    if status != "timeout" and ("out of memory" in error or "outofmemoryerror" in error):
                        status, reason = "blocked", "OOM detected in benchmark stderr"
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
    result = {"schema_version": 1, "status": status, "exit_code": EXIT_CODES[status], "benchmark_returncode": returncode, "reason": reason, "source_sha256": resolved.source_sha256, "config": config, "argv": argv, "elapsed_seconds": time.monotonic() - started, "log_directory": str(logs), "evidence_label": EVIDENCE}
    _write_json(output / "result.json", result)
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(f"# Offline experiment: {config['experiment']['name']}\n\n- Status: **{status}**\n- Exit code: {result['exit_code']}\n- Source SHA256: `{resolved.source_sha256}`\n- Evidence: {EVIDENCE}\n- Reason: {reason or 'benchmark completed'}\n\nMeasurements, if produced, are in `measurement/`; inspect their scope before making performance claims. No native hybrid, distributed training, model quality or speedup validation is implied.\n")
    return result

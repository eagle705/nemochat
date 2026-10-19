"""Bounded, local, read-only profile artifact adapters; no runtime instrumentation."""
from pathlib import Path
import stat
import sqlite3
import time
from collections import OrderedDict
from contextlib import closing
from copy import deepcopy
from datetime import datetime
import json
import math
import os
import re
import base64
import binascii


def _otlp_id(value, digits):
    # Standard OTLP JSON uses hex; protobuf MessageToDict uses base64 bytes.
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{" + str(digits) + "}", value):
        return value if int(value, 16) else None
    if isinstance(value, str) and len(value) <= 24:
        try:
            raw = base64.b64decode(value, validate=True)
            if len(raw) == digits // 2 and any(raw) and base64.b64encode(raw).decode() == value:
                return raw.hex()
        except (ValueError, binascii.Error):
            pass
    return None

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_JSON_RECORDS = 500
MAX_LINE_BYTES = 64 * 1024
ATTRIBUTES = frozenset({
    "bridge.id", "bridge.role", "bridge.explanation", "bridge.evidence",
    "source.id", "source.file_sha256", "source.snippet_sha256", "source.start_line", "source.end_line",
    "nv.dl.rank", "nv.dl.world_size", "nv.dl.step", "nv.dl.microbatch", "nv.dl.layer",
    "process.pid", "service.name", "nemo.run.id", "nv.dl.run.uuid",
    "dl.iteration", "dl.microbatch_id", "dl.data_parallel.rank", "dl.tensor_parallel.rank",
    "dl.pipeline_parallel.rank", "dl.data_parallel.size", "dl.tensor_parallel.size",
    "dl.pipeline_parallel.size", "dl.batch_size", "dl.loss", "dl.grad_norm",
    "dl.learning_rate", "dl.sequence_length", "dl.throughput_tflops", "dl.throughput_tokens_per_sec",
    "megatron.iteration", "megatron.loss", "megatron.grad_norm",
    "megatron.global_batch_size", "megatron.num_microbatches",
})


def _read_bytes(path):
    # Descriptor-relative traversal also closes the check/open symlink race.
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            next_directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
                raise ValueError("JSON artifact exceeds size limit or is not regular")
            value = stream.read(MAX_JSON_BYTES + 1)
            if len(value) > MAX_JSON_BYTES:
                raise ValueError("JSON artifact exceeds size limit")
            return value
    finally:
        os.close(directory)


def _jsonl(path, warnings):
    lines = _read_bytes(path).splitlines()
    if len(lines) > MAX_JSON_RECORDS:
        warnings.append("Record limit reached; only the bounded prefix is displayed")
    records = []
    for line in lines[:MAX_JSON_RECORDS]:
        try:
            if len(line) > MAX_LINE_BYTES:
                raise ValueError("Line limit")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Not an object")
            records.append(record)
        except (ValueError, UnicodeError, RecursionError):
            if "Skipped malformed or oversized JSONL records" not in warnings:
                warnings.append("Skipped malformed or oversized JSONL records")
    return records


def _scalar(value):
    if isinstance(value, str):
        return value if len(value) <= 512 else None
    if type(value) is int:
        return value if -(2 ** 63) < value < 2 ** 63 else None
    if type(value) is float:
        return value if math.isfinite(value) else None
    return value if isinstance(value, bool) else None


def _id(value, digits):
    if isinstance(value, str) and re.fullmatch(r"(?:0x)?[0-9a-fA-F]{" + str(digits) + "}", value):
        return value
    return None


def _lens_documents(path, warnings):
    raw = _read_bytes(path)
    try:
        document = json.loads(raw)
        if isinstance(document, dict) and "resourceSpans" in document:
            return [document]
    except (ValueError, UnicodeError, RecursionError):
        pass
    return _jsonl(path, warnings)


def _unix_ns(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,20}", value):
        value = int(value)
    if type(value) is not int or not 0 <= value < 2 ** 64:
        raise ValueError("Invalid nanoseconds")
    return value


def _any_value(value):
    if not isinstance(value, dict) or len(value) != 1:
        return None
    key, scalar = next(iter(value.items()))
    if key == "intValue":
        if isinstance(scalar, str) and re.fullmatch(r"-?[0-9]{1,19}", scalar):
            scalar = int(scalar)
        return _scalar(scalar) if type(scalar) is int else None
    if key == "doubleValue":
        return _scalar(scalar) if type(scalar) in (int, float) else None
    if key == "boolValue":
        return scalar if type(scalar) is bool else None
    if key == "stringValue":
        return _scalar(scalar) if isinstance(scalar, str) else None
    return None


def _lens_attributes(attributes, otlp, warnings):
    safe = {}
    if otlp:
        if not isinstance(attributes, list):
            warnings.append("Skipped malformed OTLP attributes")
            return safe
        pairs = [(a.get("key"), _any_value(a.get("value"))) for a in attributes[:128]
                 if isinstance(a, dict)]
        if len(attributes) > 128:
            warnings.append("Attribute limit reached")
    else:
        pairs = list(attributes.items()) if isinstance(attributes, dict) else []
    for key, value in pairs:
        if not isinstance(key, str) or key not in ATTRIBUTES:
            continue
        value = _scalar(value)
        numeric = key.startswith(("dl.", "megatron.")) or key in {
            "nv.dl.rank", "nv.dl.world_size", "nv.dl.step", "nv.dl.microbatch", "nv.dl.layer", "process.pid",
            "source.start_line", "source.end_line"}
        measurements = {"dl.loss", "dl.grad_norm", "dl.learning_rate", "dl.throughput_tflops",
                        "dl.throughput_tokens_per_sec", "megatron.loss", "megatron.grad_norm"}
        if numeric and (type(value) not in (int, float) if key in measurements else type(value) is not int):
            value = None
        if not numeric and not isinstance(value, str):
            value = None
        if value is not None:
            safe[key] = value
        else:
            warnings.append("Skipped malformed or oversized semantic attribute")
    return safe


def _lens_records(documents, warnings):
    count = 0
    for document in documents:
        if "resourceSpans" not in document:
            rows = [(document, {}, False)]
        else:
            rows = []
            resources = document.get("resourceSpans")
            if not isinstance(resources, list):
                warnings.append("Skipped malformed OTLP resourceSpans")
                continue
            for resource in resources[:MAX_JSON_RECORDS]:
                if not isinstance(resource, dict):
                    warnings.append("Skipped malformed OTLP resource")
                    continue
                res = resource.get("resource", {})
                attrs = _lens_attributes(res.get("attributes", []) if isinstance(res, dict) else [], True, warnings)
                scopes = resource.get("scopeSpans")
                if not isinstance(scopes, list):
                    warnings.append("Skipped malformed OTLP scopeSpans")
                    continue
                for scope in scopes[:MAX_JSON_RECORDS]:
                    spans = scope.get("spans") if isinstance(scope, dict) else None
                    if not isinstance(spans, list):
                        warnings.append("Skipped malformed OTLP spans")
                        continue
                    for span in spans[:MAX_JSON_RECORDS + 1]:
                        rows.append((span, attrs, True))
                        if len(rows) > MAX_JSON_RECORDS:
                            break
                    if len(rows) > MAX_JSON_RECORDS:
                        break
                if len(rows) > MAX_JSON_RECORDS:
                    break
        for row in rows:
            if count >= MAX_JSON_RECORDS:
                warnings.append("Record limit reached; only the bounded prefix is displayed")
                return
            count += 1
            yield row


def _lens(path):
    warnings = ["Lens spans measure host operation time, not CUDA kernel timing"]
    spans = []
    for record, resource_attrs, otlp in _lens_records(_lens_documents(path, warnings), warnings):
        if not isinstance(record, dict):
            warnings.append("Skipped malformed span")
            continue
        name = _scalar(record.get("name"))
        if not isinstance(name, str):
            warnings.append("Skipped span without a bounded name")
            continue
        duration = None
        try:
            if otlp:
                start = _unix_ns(record["startTimeUnixNano"])
                end = _unix_ns(record["endTimeUnixNano"])
                if end < start:
                    raise ValueError("Negative duration")
                duration = (end - start) / 1e6
            else:
                start = datetime.fromisoformat(record["start_time"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(record["end_time"].replace("Z", "+00:00"))
                if start.tzinfo is None or end.tzinfo is None or end < start:
                    raise ValueError("Invalid timestamps")
                duration = (end - start).total_seconds() * 1000
        except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
            warnings.append("Span duration unavailable: invalid or missing timezone-aware timestamps")
        if otlp:
            context = {"trace_id": record.get("traceId"), "span_id": record.get("spanId")}
            parent = record.get("parentSpanId")
            attrs = dict(resource_attrs)
        else:
            context = record.get("context") if isinstance(record.get("context"), dict) else {}
            parent = record.get("parent_id")
            resource = record.get("resource")
            attrs = _lens_attributes(resource.get("attributes") if isinstance(resource, dict) else {}, False, warnings)
        resource_identity = {key: attrs[key] for key in ("nv.dl.run.uuid", "nemo.run.id", "nv.dl.rank", "service.name") if key in attrs}
        attrs.update(_lens_attributes(record.get("attributes", [] if otlp else {}), otlp, warnings))
        # Job identity comes from explicit resource attributes, never operation names.
        for key in ("nv.dl.run.uuid", "nemo.run.id", "service.name"):
            attrs.pop(key, None)
        attrs.update(resource_identity)
        if "bridge.id" in attrs:
            warnings.append("Bridge Lens spans are an educational sparse-MoE control, independent of native training")
        else:
            warnings.append("Imported span producer and native training coverage are unverified; names do not establish integration")
        decode_id = _otlp_id if otlp else _id
        if any(decode_id(context.get(key), digits) is None for key, digits in (("trace_id", 32), ("span_id", 16))) or (parent not in (None, "") and decode_id(parent, 16) is None):
            warnings.append("Span identity unavailable: invalid or missing trace/span/parent ID")
        spans.append({"name": name, "duration_ms": duration,
                      "trace_id": decode_id(context.get("trace_id"), 32), "span_id": decode_id(context.get("span_id"), 16),
                      "parent_span_id": decode_id(parent, 16), "attributes": attrs})
    return spans, list(dict.fromkeys(warnings))


def _relative_identifier(value):
    return (isinstance(value, str) and 0 < len(value) <= 512 and not Path(value).is_absolute()
            and ".." not in Path(value).parts and "\\" not in value and "\x00" not in value)


def _bridge(path, root=None):
    warnings = ["Educational sparse-MoE control, independent of native training",
                "Host operation durations are not CUDA timing; source identifiers are unverified here",
                "NVTX emission is not proof of capture or kernel/source correlation"]
    operations = []
    for record in _jsonl(path, warnings):
        if not isinstance(record.get("role"), str) or _scalar(record.get("role")) is None:
            warnings.append("Skipped operation without a bounded role")
            continue
        operation = {}
        for key in ("bridge_id", "role", "title", "explanation", "evidence", "status", "nvtx_emitted", "lens_enabled"):
            value = _scalar(record.get(key))
            if value is not None:
                operation[key] = value
        start, end = record.get("host_start_monotonic_ns"), record.get("host_end_monotonic_ns")
        duration = None
        if type(start) is int and type(end) is int and 0 <= start <= end and end < 2 ** 63:
            duration = (end - start) / 1e6
        operation["duration_ms"] = duration
        source = record.get("source")
        safe_source = {}
        if isinstance(source, dict):
            for key in ("source_id", "file_sha256", "snippet_sha256"):
                value = _id(source.get(key), 64)
                if value is not None:
                    safe_source[key] = value
            if _relative_identifier(source.get("path")):
                safe_source["path"] = source["path"]
            for key in ("start_line", "end_line"):
                if type(source.get(key)) is int and 0 < source[key] < 2 ** 31:
                    safe_source[key] = source[key]
        operation["source"] = safe_source
        if root is not None and isinstance(source, dict):
            try:
                from .source_evidence import verified_bridge_source
            except ImportError:
                try:
                    from dashboard.source_evidence import verified_bridge_source
                except ImportError:
                    from source_evidence import verified_bridge_source
            operation['source_evidence'] = verified_bridge_source(Path(root), path, record)
        operations.append(operation)
    return operations, list(dict.fromkeys(warnings))


def _numbers(record, keys):
    return {key: record[key] for key in keys if type(record.get(key)) in (int, float)
            and 0 <= record[key] < 2 ** 63 and math.isfinite(record[key])}


def _summary(root, path, nsight):
    record = json.loads(_read_bytes(path))
    if not isinstance(record, dict) or record.get("status") != "coverage-verified":
        raise ValueError("Summary producer coverage is not verified")
    counts = record.get("capture_counts")
    if not isinstance(counts, dict) or type(counts.get("CUPTI_ACTIVITY_KIND_KERNEL")) is not int:
        raise ValueError("Summary capture count missing")
    if not nsight["available"] or counts["CUPTI_ACTIVITY_KIND_KERNEL"] != nsight["summary"].get("kernel_count"):
        raise ValueError("Summary whole-capture count cannot be independently checked against SQLite")
    selected = {"source": str(path.relative_to(root)),
                "verification": "whole_capture_kernel_count_checked; selected_correlation_not_recomputed",
                "scope": "producer-selected ranges only; separate from whole-capture kernel aggregates",
                "trace_ranks": [], "whole_node_trace_coverage": record.get("whole_node_trace_coverage")
                    if type(record.get("whole_node_trace_coverage")) is bool else None,
                "updates": []}
    ranks = record.get("trace_ranks")
    if isinstance(ranks, list):
        selected["trace_ranks"] = [rank for rank in ranks[:128] if type(rank) is int and 0 <= rank < 100000]
    selected.update(_numbers(record, ("selected_update_ranges", "trace_loss_tokens", "all_rank_run_loss_tokens",
                                     "rank_zero_measured_loss_bearing_tokens", "all_rank_run_measured_loss_bearing_tokens")))
    numeric = ("rank", "update", "cpu_nvtx_ms", "associated_correlated_kernel_count",
               "parent_thread_correlated_kernel_count", "worker_thread_correlated_kernel_count",
               "summed_kernel_ms_not_wall", "lr", "loss_bearing_tokens")
    updates = record.get("updates")
    if not isinstance(updates, list) or len(updates) > 64:
        raise ValueError("Summary update list missing or exceeds limit")
    for update in updates:
        if not isinstance(update, dict):
            raise ValueError("Malformed summary update")
        safe = _numbers(update, numeric)
        phases = update.get("phases")
        if isinstance(phases, list):
            safe["phases"] = []
            for phase in phases[:32]:
                if isinstance(phase, dict) and isinstance(_scalar(phase.get("name")), str):
                    safe["phases"].append({"name": phase["name"], **_numbers(phase, numeric)})
        devices = update.get("captured_process_activity_by_device")
        if isinstance(devices, list):
            safe["captured_process_activity_by_device"] = [
                _numbers(device, ("device", "clipped_activity_union_ms")) for device in devices[:128]
                if isinstance(device, dict)]
        selected["updates"].append(safe)
    return selected

_CACHE = OrderedDict()


def _fingerprint(path):
    values = []
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"), Path(str(path) + "-journal")):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            values.append(None)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Unsafe SQLite sidecar")
        values.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(values)


def _cached_sqlite(path):
    before = _fingerprint(path)
    key = (str(path), before)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return deepcopy(_CACHE[key])
    result = _sqlite(path)
    if before == _fingerprint(path):
        _CACHE[key] = deepcopy(result)
        while len(_CACHE) > 8:
            _CACHE.popitem(last=False)
    else:
        result[2].append("SQLite changed during collection; result is a read transaction snapshot, not cached")
    return result

MAX_SQLITE_BYTES = 2 * 1024 ** 3
MAX_KERNEL_ROWS = 1_000_000
MAX_KERNEL_NAMES = 512
MAX_COVERAGE = 128
SQLITE_TIMEOUT_SECONDS = 2.0


def _sqlite(path):
    if path.stat().st_size > MAX_SQLITE_BYTES:
        raise ValueError("SQLite file exceeds size limit")
    warnings = [
        "Summed kernel durations may overlap; not wall time or whole-device utilization",
        "Raw kernel names alone do not establish source attribution",
        "Only recorded process/device coverage is known; missing ranks and trace completeness are unknown",
    ]
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.1)) as db:
        deadline = time.monotonic() + SQLITE_TIMEOUT_SECONDS
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("BEGIN")
        columns = {row[1] for row in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        strings = {row[1] for row in db.execute("PRAGMA table_info(StringIds)")}
        if not {"start", "end", "demangledName"} <= columns or not {"id", "value"} <= strings:
            raise ValueError("Unsupported Nsight kernel/StringIds schema")
        process = "globalPid" if "globalPid" in columns else "NULL"
        device = "deviceId" if "deviceId" in columns else "NULL"
        base = (f"WITH bounded AS (SELECT start, end, demangledName, {process} AS globalPid, "
                f"{device} AS deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT {MAX_KERNEL_ROWS + 1}) ")
        scanned, valid, duration = db.execute(base +
            "SELECT count(*), sum(CASE WHEN end >= start THEN 1 ELSE 0 END), "
            "sum(CASE WHEN end >= start THEN end-start ELSE 0 END) FROM bounded").fetchone()
        if scanned > MAX_KERNEL_ROWS:
            raise ValueError("Kernel record limit exceeded; aggregation unavailable")
        valid, duration = valid or 0, duration or 0
        if valid != scanned:
            warnings.append("Invalid negative-duration kernel records excluded")
        grouped = db.execute(base.rstrip() +
            ", grouped AS (SELECT demangledName, count(*) AS calls, sum(end-start) AS ns "
            "FROM bounded WHERE end >= start GROUP BY demangledName) "
            "SELECT s.value, sum(g.calls), sum(g.ns) FROM grouped g "
            "LEFT JOIN StringIds s ON s.id=g.demangledName "
            f"GROUP BY s.value ORDER BY sum(g.ns) DESC, s.value LIMIT {MAX_KERNEL_NAMES + 1}").fetchall()
        if len(grouped) > MAX_KERNEL_NAMES:
            warnings.append("Kernel name display limit reached; percentages use all valid captured kernels")
        kernels = []
        for name, count, ns in grouped[:MAX_KERNEL_NAMES]:
            if name is not None and (not isinstance(name, str) or len(name) > 8192):
                raise ValueError("Kernel name exceeds display limit")
            if name is None:
                warnings.append("Some raw kernel names are missing from StringIds")
            kernels.append({"name": name, "count": count, "total_ms": ns / 1e6,
                            "average_us": ns / count / 1e3, "percent": ns / duration * 100 if duration else None})
        coverage = db.execute(base + f"SELECT {process}, {device}, count(*) FROM bounded "
            f"WHERE end >= start GROUP BY {process}, {device} ORDER BY {process}, {device} "
            f"LIMIT {MAX_COVERAGE + 1}").fetchall()
        if len(coverage) > MAX_COVERAGE:
            warnings.append("Process/device coverage display limit reached")
        summary = {"scope": "whole capture; all recorded processes/devices; includes startup and warmup",
                   "kernel_count": valid, "records_scanned": scanned,
                   "summed_kernel_ms_not_wall": duration / 1e6,
                   "coverage": [{"global_pid": pid, "device_id": dev, "kernel_count": count}
                                for pid, dev, count in coverage[:MAX_COVERAGE]]}
    return kernels, summary, warnings


def _path(root, value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Artifact path must be a repository-relative file")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Artifact path escapes repository or is not a file")
    path = root
    for part in relative.parts:
        path = path / part
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ValueError("Symlink artifact paths are not allowed")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Artifact is not a regular file")
    return path


def _collect_group(root: Path, paths: dict) -> dict:
    result = {
        "lens": {"available": False, "source": None, "spans": [], "warnings": []},
        "nsight": {"available": False, "source": None, "kernels": [], "summary": {}, "warnings": []},
        "bridge": {"available": False, "source": None, "operations": [], "warnings": []},
    }
    for name, key in (("lens", "lens_spans"), ("nsight", "nsight_sqlite"), ("bridge", "bridge_operations")):
        value = paths.get(key)
        if value is None:
            continue
        try:
            path = _path(Path(root).resolve(), value)
            if name == "nsight":
                kernels, summary, warnings = _cached_sqlite(path)
                result[name].update(available=True, source=value, kernels=kernels, summary=summary, warnings=warnings)
            elif name == "lens":
                spans, warnings = _lens(path)
                result[name].update(available=bool(spans), source=value, spans=spans, warnings=warnings)
            elif name == "bridge":
                operations, warnings = _bridge(path, root)
                result[name].update(available=bool(operations), source=value, operations=operations, warnings=warnings)
        except (OSError, ValueError, sqlite3.Error):
            result[name]["warnings"].append("Configured artifact unavailable: missing, unsafe, malformed, over limit, or SQLite query timed out")
    summary_path = paths.get("nsight_summary")
    if summary_path is not None:
        try:
            root = Path(root).resolve()
            selected = _summary(root, _path(root, summary_path), result["nsight"])
            result["nsight"]["summary"]["selected_ranges"] = selected
            result["nsight"]["warnings"].extend([
                "Imported summary whole-capture count checked; selected-range correlation is producer-reported, not recomputed",
                "Selected activity union is captured-process/device timing, not utilization; CPU NVTX time is host time",
                "Summary source paths/hashes are not followed; this check does not authenticate producer provenance",
            ])
        except (OSError, ValueError, UnicodeError, RecursionError):
            result["nsight"]["warnings"].append("Configured summary unavailable, unsafe, malformed, or capture count mismatched")
    if paths.get('nsight_analysis') is not None:
        try:
            from .inspector import load_inspector
        except ImportError:
            try:
                from dashboard.inspector import load_inspector
            except ImportError:
                from inspector import load_inspector
        result['nsight']['inspector'] = load_inspector(root, paths)
    return result


STAGES = ("pretrain", "mid", "sft", "long", "rlvr", "mopd")


def _optimizations(stage, group):
    """Cheap evidence cards, not causal bottleneck or achieved speedup claims."""
    nsight = group["nsight"]
    cards = []
    def card(title, evidence, hypothesis, experiment, confidence, source, severity="info"):
        cards.append({"stage": stage, "severity": severity, "title": title, "evidence": evidence,
                      "hypothesis": hypothesis, "next_experiment": experiment,
                      "confidence": confidence, "source": source})
    inspector = nsight.get('inspector', {})
    if inspector.get('available'):
        for candidate in inspector.get('candidates', [])[:4]:
            card(candidate['title'], candidate['evidence'], candidate['interpretation'],
                 candidate['action'], candidate['confidence'], inspector.get('source', nsight['source']))
        return cards[:4]
    if nsight["available"]:
        summary = nsight["summary"]
        selected = summary.get("selected_ranges", {})
        card("프로파일 범위 확인", f"전체 캡처 커널 {summary.get('kernel_count')}개, "
             f"기록된 프로세스/장치 조합 {len(summary.get('coverage', []))}개. 초기화·워밍업 포함. "
             f"계측 캡처; 보고된 trace rank {selected.get('trace_ranks') or '미확인'}, "
             f"선택 업데이트 {len(selected.get('updates', []))}개는 전체 집계와 별개입니다.",
             "현재 캡처만으로 전체 노드 병목이나 GPU 사용률을 판단할 수 없습니다.",
             "같은 입력의 선택 업데이트를 필요한 rank에서 짧게 캡처하세요. 후보 변경 하나만 정확도·메모리·비계측 전체 업데이트 A/B로 검증하세요.",
             "observed", nsight["source"])
        kernels = nsight["kernels"]
        if kernels:
            top = kernels[0]
            card("누적 커널 시간 상위 항목 조사 후보",
                 f"원래 이름: {top['name']}; 호출 {top['count']}회; 합산 {top['total_ms']:.3f} ms. "
                 "전체 캡처 집계이며 겹치는 시간은 벽시계 시간이 아닙니다.",
                 "반복 호출·동기화·연산 비용을 조사할 후보입니다. 이름이나 시간 비중만으로 소스 또는 병목을 확정하지 않습니다.",
                 "선택 업데이트의 CUDA launch correlation·stream·NVTX 범위를 확인하세요. 호출 비용이 검증되면 fusion/batching을 정확도·메모리·비계측 전체 업데이트 A/B로 검증하세요.",
                 "candidate", nsight["source"])
        updates = selected.get("updates", [])
        phases = [phase for update in updates for phase in update.get("phases", []) if "cpu_nvtx_ms" in phase]
        if phases:
            longest = max(phases, key=lambda phase: phase["cpu_nvtx_ms"])
            card("선택 업데이트 호스트 구간 조사 후보",
                 f"생산자가 보고한 선택 범위 {len(updates)}개; 호스트 구간 {longest['name']}: "
                 f"{longest['cpu_nvtx_ms']:.3f} ms. 선택 상관관계는 재계산하지 않았습니다.",
                 "긴 호스트 구간에는 대기·동기화·계측 비용이 포함될 수 있으며 CUDA 실행 시간과 같지 않습니다.",
                 "같은 선택 구간의 API launch와 GPU 활동을 대조하고 입력·연산·통신을 분리하세요. 후보 변경 하나만 정확도·메모리·비계측 전체 업데이트 A/B로 검증하세요.",
                 "candidate", selected["source"])
    lens = group["lens"]
    durations = [span for span in lens["spans"] if span["duration_ms"] is not None]
    if durations and len(cards) < 4:
        longest = max(durations, key=lambda span: span["duration_ms"])
        card("Lens 호스트 구간 확인", f"표시된 호스트 span {len(lens['spans'])}개; "
             f"최장 {longest['name']}: {longest['duration_ms']:.3f} ms.",
             "bridge.id로 식별된 span만 별도 교육용 sparse-MoE 제어 실험입니다. 나머지 span의 native 학습 범위는 미확인입니다. 호스트 시간 합계는 CUDA 시간이나 학습 병목이 아닙니다.",
             "실제 단계의 span·CUDA launch 증거를 먼저 확보하세요. telemetry on/off의 작은 반복 대조에서 정확도·메모리·비계측 전체 업데이트 A/B를 확인하고 호스트 계측 비용을 분리하세요.",
             "observed", lens["source"])
    return cards[:4]


def collect_profiling(root: Path, config: dict) -> dict:
    """Collect explicit per-stage artifacts; legacy native capture is pretrain only.

    Legacy Lens/bridge control evidence remains top-level, not silently assigned
    to any training stage. Explicit stage paths are the caller's attribution.
    """
    paths = config.get("profiling", {}) if isinstance(config, dict) else {}
    paths = paths if isinstance(paths, dict) else {}
    stage_paths = paths.get("stages", {})
    stage_paths = dict(stage_paths) if isinstance(stage_paths, dict) else {}
    if "long" not in stage_paths and "long_context" in stage_paths:
        stage_paths["long"] = stage_paths["long_context"]
    legacy = _collect_group(root, paths)
    stages = {}
    for stage in STAGES:
        configured = stage_paths.get(stage)
        if isinstance(configured, dict):
            group = _collect_group(root, configured)
        elif stage == "pretrain":
            group = _collect_group(root, {})
            group["nsight"] = deepcopy(legacy["nsight"])
        else:
            group = _collect_group(root, {})
        group["optimizations"] = _optimizations(stage, group)
        stages[stage] = group
    default = paths.get("default_stage", "pretrain")
    if default == "long_context":
        default = "long"
    if default not in STAGES:
        default = "pretrain"
    if default != "pretrain" or isinstance(stage_paths.get(default), dict):
        result = deepcopy(stages[default])
    else:
        result = legacy
    result["stages"] = stages
    result["optimizations"] = deepcopy(stages[default]["optimizations"])
    return result

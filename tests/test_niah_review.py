"""Offline regressions for NIAH review blockers (no model-quality claims)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_long_context import TinyTokenizer
from nemochat.long_context.niah import generate_fixtures, run_predictions

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("niah_review_cli", ROOT / "scripts/niah.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize("damage", ["tokens", "hash", "duplicate"])
def test_complete_validation_before_client(damage):
    tokenizer = TinyTokenizer()
    rows = generate_fixtures(tokenizer, [512], [.5], samples=2)
    if damage == "tokens":
        rows[1]["prompt_tokens"] += 1
    elif damage == "hash":
        rows[1]["prompt_token_sha256"] = "bad"
    else:
        rows[1]["id"] = rows[0]["id"]
    create = Mock(return_value={"choices": [{"text": "x"}], "usage": {"prompt_tokens": rows[0]["prompt_tokens"]}})
    with pytest.raises(ValueError):
        run_predictions(rows, tokenizer, "hosted-client", "model", client=SimpleNamespace(completions=SimpleNamespace(create=create)))
    create.assert_not_called()


@pytest.mark.parametrize("command", ["generate", "run", "score", "plot"])
@pytest.mark.parametrize("broken", [False, True])
def test_existing_destination_rejected_before_io(tmp_path, monkeypatch, command, broken):
    dest = tmp_path / ("figure.metadata.json" if command == "plot" else "output")
    if broken:
        dest.symlink_to(tmp_path / "missing")
    else:
        dest.write_text("EXISTING EVIDENCE")
    load = Mock(side_effect=AssertionError("tokenizer IO"))
    calls = Mock(side_effect=AssertionError("completion IO"))
    monkeypatch.setattr(cli, "load_tokenizer", load)
    monkeypatch.setattr(cli, "run_predictions", calls)
    argv = [command]
    if command in ("generate", "run"):
        argv += ["--tokenizer", "unloaded"]
    if command != "generate":
        source = tmp_path / "input.jsonl"
        source.write_text(json.dumps(generate_fixtures(TinyTokenizer(), [512], [.5], samples=1)[0]) + "\n")
        argv += ["--input", str(source)]
    if command == "run":
        argv += ["--endpoint", "https://unused.test", "--model", "unused"]
    argv += ["--output-prefix", str(tmp_path / "figure")] if command == "plot" else ["--output", str(dest)]
    with pytest.raises(SystemExit):
        cli.main(argv)
    load.assert_not_called()
    calls.assert_not_called()
    assert dest.is_symlink() if broken else dest.read_text() == "EXISTING EVIDENCE"
    if command == "plot":
        assert not (tmp_path / "figure.png").exists()
        assert not (tmp_path / "figure.svg").exists()


@pytest.mark.parametrize("collision", ["input", "csv", "ancestor", "config", "haystack", "tokenizer", "plot"])
def test_collisions_before_io(tmp_path, monkeypatch, collision):
    dest = tmp_path / "out"
    load = Mock(side_effect=AssertionError("tokenizer IO"))
    read = Mock(side_effect=AssertionError("input IO"))
    monkeypatch.setattr(cli, "load_tokenizer", load)
    monkeypatch.setattr(cli, "read_jsonl", read)
    if collision in ("config", "haystack", "tokenizer"):
        argv = ["generate", "--tokenizer", str(dest) if collision == "tokenizer" else "unloaded", "--output", str(dest)]
        if collision != "tokenizer":
            argv += ["--" + collision, str(dest)]
    elif collision == "plot":
        argv = ["plot", "--input", str(dest) + ".svg", "--output-prefix", str(dest)]
    else:
        argv = ["score", "--input", str(dest) if collision == "input" else str(tmp_path / "input"), "--output", str(dest)]
        if collision != "input":
            argv += ["--csv", str(dest if collision == "csv" else dest / "child")]
    with pytest.raises(SystemExit):
        cli.main(argv)
    load.assert_not_called()
    read.assert_not_called()
    assert not dest.exists()


def test_invalid_parent_before_run_io(tmp_path, monkeypatch):
    parent = tmp_path / "not-directory"
    parent.write_text("SENTINEL")
    load = Mock(side_effect=AssertionError("tokenizer IO"))
    calls = Mock(side_effect=AssertionError("client IO"))
    monkeypatch.setattr(cli, "load_tokenizer", load)
    monkeypatch.setattr(cli, "run_predictions", calls)
    with pytest.raises(SystemExit):
        cli.main(["run", "--tokenizer", "unloaded", "--input", str(tmp_path / "input"),
                  "--output", str(parent / "out"), "--endpoint", "https://unused.test", "--model", "m"])
    load.assert_not_called()
    calls.assert_not_called()
    assert parent.read_text() == "SENTINEL"


@pytest.mark.parametrize("suffix", [".png", ".svg", ".metadata.json", ".csv"])
def test_publication_race_no_clobber(tmp_path, suffix):
    from nemochat.long_context.io import FreshOutputs
    first, raced = tmp_path / "first.jsonl", tmp_path / ("second" + suffix)
    with pytest.raises(FileExistsError):
        with FreshOutputs([first, raced]) as outputs:
            outputs.write(first, "our first")
            outputs.write(raced, "our second")
            raced.write_text("RACING EVIDENCE")
    assert raced.read_text() == "RACING EVIDENCE"
    assert first.read_text() == "our first"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([
        first.name, raced.name, f".{first.name}.niah.lock", f".{raced.name}.niah.lock",
    ])


def test_concurrent_reservation_no_clobber(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from nemochat.long_context.io import FreshOutputs
    gate = Barrier(2)
    dest = tmp_path / "shared.jsonl"
    def writer(value):
        gate.wait(timeout=5)
        try:
            with FreshOutputs([dest]) as outputs:
                outputs.write(dest, value)
            return value
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(writer, ["first", "second"]))
    winners = [value for value in results if value is not None]
    assert len(winners) == 1
    assert dest.read_text() == winners[0]
    assert set(tmp_path.iterdir()) == {dest, dest.with_name(f".{dest.name}.niah.lock")}


@pytest.mark.parametrize("suffix", [".png", ".svg", ".metadata.json"])
def test_direct_plot_rejects_all_destinations(tmp_path, suffix):
    from nemochat.long_context.plot import plot_predictions
    prefix = tmp_path / "figure"
    existing = Path(str(prefix) + suffix)
    existing.write_text("SENTINEL")
    with pytest.raises(FileExistsError):
        plot_predictions([], prefix)
    assert existing.read_text() == "SENTINEL"
    assert list(tmp_path.iterdir()) == [existing]


def test_config_overrides_explicit_flags(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"context_lengths": [512], "samples": 2}))
    monkeypatch.setattr(cli, "load_tokenizer", lambda args: TinyTokenizer())
    dest = tmp_path / "fixture.jsonl"
    cli.main(["generate", "--tokenizer", "unloaded", "--config", str(config),
              "--context-lengths", "128", "--samples", "1", "--depths", ".5", "--output", str(dest)])
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    assert len(rows) == 2 and all(row["context_length"] == 512 for row in rows)
    with pytest.raises(SystemExit) as exc:
        cli.main(["generate", "--help"])
    assert exc.value.code == 0
    assert "config values override explicit CLI flags" in " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("broken", [False, True])
def test_existing_csv_rejects_entire_score_bundle(tmp_path, broken):
    source, output, csv = (tmp_path / name for name in ("input", "score", "score.csv"))
    source.write_text(json.dumps(generate_fixtures(TinyTokenizer(), [512], [.5], samples=1)[0]))
    if broken:
        csv.symlink_to(tmp_path / "missing")
    else:
        csv.write_text("CSV EVIDENCE")
    with pytest.raises(SystemExit):
        cli.main(["score", "--input", str(source), "--output", str(output), "--csv", str(csv)])
    assert not output.exists()
    assert csv.is_symlink() if broken else csv.read_text() == "CSV EVIDENCE"


@pytest.mark.parametrize("command", ["generate", "run", "score"])
def test_cli_publication_race_preserves_competitor(tmp_path, monkeypatch, command):
    dest, source = tmp_path / "out", tmp_path / "input"
    rows = generate_fixtures(TinyTokenizer(), [512], [.5], samples=1)
    source.write_text(json.dumps(rows[0]))
    monkeypatch.setattr(cli, "load_tokenizer", lambda args: TinyTokenizer())
    def race(*args, **kwargs):
        dest.write_text("COMPETING EVIDENCE")
        return [{**row, "status": "ok"} for row in rows]
    target = {"generate": "generate_fixtures", "run": "run_predictions", "score": "score_predictions"}[command]
    monkeypatch.setattr(cli, target, race)
    argv = [command, "--output", str(dest)]
    if command in ("generate", "run"):
        argv += ["--tokenizer", "unloaded"]
    if command != "generate":
        argv += ["--input", str(source)]
    if command == "run":
        argv += ["--endpoint", "https://unused.test", "--model", "m"]
    with pytest.raises(SystemExit):
        cli.main(argv)
    assert dest.read_text() == "COMPETING EVIDENCE"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".out.niah.lock", "input", "out"]


def test_symlink_parent_alias_collision(tmp_path):
    from nemochat.long_context.io import FreshOutputs
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="collide"):
        with FreshOutputs([real / "out", alias / "out"]):
            pytest.fail("collision not rejected")
    assert list(real.iterdir()) == []


@pytest.mark.parametrize("target_kind", ["published", "reservation"])
def test_cleanup_never_unlinks_replaced_public_paths(tmp_path, monkeypatch, target_kind):
    import os
    from nemochat.long_context.io import FreshOutputs
    first, second = tmp_path / "first", tmp_path / "second"
    outputs = FreshOutputs([first, second])
    outputs.__enter__()
    outputs.write(first, "ours")
    outputs.write(second, "ours")
    second.write_text("SECOND COMPETITOR")
    target = first if target_kind == "published" else next(iter(outputs.locks))
    trigger_fd = outputs.staged[first][0] if target_kind == "published" else outputs.locks[target]
    original_close = os.close
    replaced = []

    def replace():
        replacement = tmp_path / "replacement"
        replacement.write_text("REPLACEMENT EVIDENCE")
        os.replace(replacement, target)
        replaced.append(target)

    # Reproduce replacement immediately after the old successful ownership check.
    old_check = getattr(FreshOutputs, "_same_inode", None)
    def checked_then_replaced(path, fd):
        owned = old_check(path, fd) if old_check is not None else False
        if path == target and owned:
            replace()
        return owned
    monkeypatch.setattr(FreshOutputs, "_same_inode", staticmethod(checked_then_replaced), raising=False)

    # Under retention there is no ownership check: replace during FD cleanup.
    def close(fd):
        if fd == trigger_fd and not replaced:
            replace()
        original_close(fd)
    monkeypatch.setattr(os, "close", close)
    with pytest.raises(FileExistsError):
        outputs.__exit__(None, None, None)
    assert replaced == [target]
    assert target.read_text() == "REPLACEMENT EVIDENCE"
    assert second.read_text() == "SECOND COMPETITOR"
    for fd in [fd for fd, _ in outputs.staged.values()] + list(outputs.locks.values()):
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("failure_stage", ["body", "publication", "reservation"])
def test_cleanup_permission_error_preserves_primary_and_closes_all(tmp_path, monkeypatch, failure_stage):
    import os
    import tempfile
    from nemochat.long_context.io import FreshOutputs
    first, second = tmp_path / "first", tmp_path / "second"
    outputs = FreshOutputs([first, second])
    primary = ValueError("PRIVATE PRIMARY MESSAGE")
    original_unlink, original_link = Path.unlink, os.link
    attempted, descriptors, locks = [], [], []

    def deny_first_stage(path, *args, **kwargs):
        if path in [temporary for _, temporary in outputs.staged.values()]:
            attempted.append(path)
            if path == next(iter(outputs.staged.values()))[1]:
                raise PermissionError("PRIVATE CLEANUP MESSAGE")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", deny_first_stage)

    def fail_publication(source, dest, *args, **kwargs):
        if Path(dest) == second:
            second.write_text("COMPETITOR")
            raise primary
        return original_link(source, dest, *args, **kwargs)
    if failure_stage == "publication":
        monkeypatch.setattr(os, "link", fail_publication)
    if failure_stage == "reservation":
        original_mkstemp = tempfile.mkstemp
        def fail_second_stage(*args, **kwargs):
            if outputs.staged:
                descriptors.extend(fd for fd, _ in outputs.staged.values())
                descriptors.extend(outputs.locks.values())
                locks.extend(outputs.locks)
                raise primary
            return original_mkstemp(*args, **kwargs)
        monkeypatch.setattr(tempfile, "mkstemp", fail_second_stage)

    with pytest.raises(ValueError) as caught:
        with outputs:
            descriptors.extend(fd for fd, _ in outputs.staged.values())
            descriptors.extend(outputs.locks.values())
            locks.extend(outputs.locks)
            outputs.write(first, "ours")
            outputs.write(second, "ours")
            if failure_stage == "body":
                second.write_text("COMPETITOR")
                raise primary
    assert caught.value is primary
    assert len(attempted) == len(outputs.staged)
    assert all(lock.exists() for lock in locks)
    if failure_stage != "reservation":
        assert second.read_text() == "COMPETITOR"
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert getattr(primary, "__notes__", []) == ["NIAH cleanup staging-unlink: PermissionError"]


@pytest.mark.parametrize("close_stage", ["staging", "reservation"])
@pytest.mark.parametrize("body_failure", [False, True])
def test_close_failure_attempts_every_cleanup_with_safe_diagnostics(tmp_path, monkeypatch, close_stage, body_failure):
    import os
    from nemochat.long_context.io import FreshOutputs
    first, second = tmp_path / "first", tmp_path / "second"
    outputs = FreshOutputs([first, second])
    outputs.__enter__()
    outputs.write(first, "first")
    outputs.write(second, "second")
    staged_fds = [fd for fd, _ in outputs.staged.values()]
    lock_fds = list(outputs.locks.values())
    target = staged_fds[0] if close_stage == "staging" else lock_fds[0]
    original_close = os.close
    attempted = []
    def close(fd):
        attempted.append(fd)
        original_close(fd)
        if fd == target:
            raise PermissionError("PRIVATE CLOSE MESSAGE")
    monkeypatch.setattr(os, "close", close)
    primary = ValueError("PRIVATE BODY MESSAGE") if body_failure else None
    with pytest.raises(ValueError if body_failure else RuntimeError) as caught:
        if primary is not None:
            try:
                raise primary
            except ValueError as error:
                outputs.__exit__(type(error), error, error.__traceback__)
                raise
        else:
            outputs.__exit__(None, None, None)
    if body_failure:
        assert caught.value is primary
    else:
        assert str(caught.value) == "NIAH cleanup failed"
        assert first.read_text() == "first"
        assert second.read_text() == "second"
    assert caught.value.__notes__ == [f"NIAH cleanup {close_stage}-close: PermissionError"]
    assert attempted == staged_fds + lock_fds
    for fd in attempted:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert all(not temporary.exists() for _, temporary in outputs.staged.values())
    assert all(lock.exists() for lock in outputs.locks)


def test_failed_attempt_tombstone_requires_new_destination(tmp_path):
    from nemochat.long_context.io import FreshOutputs
    dest = tmp_path / "attempt"
    with pytest.raises(ValueError, match="body"):
        with FreshOutputs([dest]):
            raise ValueError("body")
    assert not dest.exists()
    assert dest.with_name(".attempt.niah.lock").exists()
    with pytest.raises(FileExistsError):
        with FreshOutputs([dest]):
            pytest.fail("failed attempt was reused")
    fresh = tmp_path / "new-attempt"
    with FreshOutputs([fresh]) as outputs:
        outputs.write(fresh, "new")
    assert fresh.read_text() == "new"

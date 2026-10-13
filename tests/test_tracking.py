"""Tracking failures and RNG invariance; fake SDK tests are explicitly fixtures."""
import argparse
import json
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nemochat.tracking import STAGES, StageRun, add_tracking_args, from_args, sanitize


@pytest.fixture
def sdk(monkeypatch):
    run = Mock()
    run.dir = "fixture-offline-dir"
    run.summary = {}
    run.config = Mock()
    wandb = SimpleNamespace(init=Mock(return_value=run), Settings=Mock(), Artifact=Mock())
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    return wandb, run


@pytest.mark.parametrize("stage", STAGES)
def test_registration(stage, sdk, tmp_path):
    wandb, run = sdk
    with StageRun(stage, mode="offline", directory=tmp_path, group="pipeline", parent_id="parent", checkpoint_id="sha") as tracking:
        tracking.log({"metric": 1}, step=3)
        tracking.artifact_metadata("weights", kind="checkpoint-metadata", metadata={"weights_uploaded": False})
    assert wandb.init.call_args.kwargs["config"]["tracking"]["stage"] == stage
    assert wandb.init.call_args.kwargs["config"]["tracking"]["parent_run_id"] == "parent"
    run.finish.assert_called_once_with(exit_code=0)
    assert json.loads(next(tmp_path.glob("tracking-*.json")).read_text())["status"] == "finished"


def test_disabled_and_nonzero_rank_do_not_import(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", None)
    for mode, rank in (("disabled", "0"), ("offline", "1")):
        monkeypatch.setenv("RANK", rank)
        with StageRun("pretrain", mode=mode, directory=tmp_path) as tracking:
            tracking.log({"x": 1})
    assert list(tmp_path.iterdir()) == []


def test_secret_keys_not_tokenizer():
    filtered = sanitize({"token": "private", "hf_token": "private", "nested": {"api_key": "private"},
                         "tokenizer": "gpt2", "tokenizer_revision": "sha", "token_quota": 99,
                         "loss_tokens_seen": 10})
    assert filtered["token"] == filtered["hf_token"] == "[REDACTED]"
    assert filtered["nested"]["api_key"] == "[REDACTED]"
    assert filtered["tokenizer"] == "gpt2" and filtered["token_quota"] == 99


@pytest.mark.parametrize("failure", ["body", "log", "finish", "init", "none"])
def test_failures_are_not_success(sdk, tmp_path, failure):
    wandb, run = sdk
    if failure in ("init", "none"):
        wandb.init.side_effect = RuntimeError("SDK failed") if failure == "init" else None
        if failure == "none":
            wandb.init.return_value = None
    elif failure in ("log", "finish"):
        getattr(run, failure).side_effect = RuntimeError("SDK failed")
    with pytest.raises(RuntimeError):
        with StageRun("data", mode="offline", directory=tmp_path) as tracking:
            if failure == "body":
                raise RuntimeError("work failed")
            tracking.log({"x": 1})
    assert json.loads(next(tmp_path.glob("tracking-*.json")).read_text())["status"] == "failed"
    if failure in ("body", "log"):
        run.finish.assert_called_once_with(exit_code=1)


def test_rng_preserved_for_all_sdk_operations(sdk, tmp_path):
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    wandb, run = sdk
    def consume(*args, **kwargs):
        random.random(); torch.rand(2); numpy.random.rand(2)
    wandb.init.side_effect = lambda **kwargs: (consume(), run)[1]
    for fn in (run.log, run.finish, run.config.update, run.log_artifact):
        fn.side_effect = consume
    before = random.getstate(), torch.get_rng_state().clone(), numpy.random.get_state()
    with StageRun("pretrain", mode="offline", directory=tmp_path) as tracking:
        tracking.log({"lm_ce": 1})
        tracking.update_config({"tokenizer": "fixture"})
        tracking.artifact_metadata("test", kind="checkpoint-metadata", metadata={})
    assert random.getstate() == before[0]
    assert torch.equal(torch.get_rng_state(), before[1])
    assert numpy.array_equal(numpy.random.get_state()[1], before[2][1])


def test_resume_modes_and_cli_precedence(sdk, tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    add_tracking_args(parser)
    config = tmp_path / "config.json"
    config.write_text('{"project":"from-json","group":"json-group"}')
    args = parser.parse_args(["--wandb-config", str(config), "--wandb-project", "cli-project"])
    tracking = from_args(args, "pretrain")
    assert tracking.mode == "disabled" and tracking.metadata["project"] == "cli-project"
    assert tracking.metadata["group"] == "json-group"
    with pytest.raises(ValueError, match="explicit"):
        StageRun("pretrain", mode="offline", resume=True)
    with StageRun("pretrain", mode="offline", resume=True, run_id="same", directory=tmp_path):
        pass
    assert sdk[0].init.call_args.kwargs["resume"] is None
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no interactive login"):
        with StageRun("data", mode="online", directory=tmp_path):
            pass
    monkeypatch.setenv("WANDB_API_KEY", "fixture-never-sent")
    with StageRun("pretrain", mode="online", resume=True, run_id="same", directory=tmp_path):
        pass
    assert sdk[0].init.call_args.kwargs["resume"] == "must"


def test_caught_failure_stays_failed(sdk, tmp_path):
    sdk[1].log.side_effect = RuntimeError("fixture failure")
    with StageRun("data", mode="offline", directory=tmp_path) as tracking:
        with pytest.raises(RuntimeError):
            tracking.log({"x": 1})
    sdk[1].finish.assert_called_once_with(exit_code=1)
    assert json.loads(next(tmp_path.glob("tracking-*.json")).read_text())["status"] == "failed"


def test_immutable_hf_reference_and_nonfinite(sdk, tmp_path):
    from nemochat.tracking import hf_reference
    ref = hf_reference("sshleifer/tiny-gpt2", "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be")
    assert ref["weights_uploaded"] is False
    with pytest.raises(ValueError):
        hf_reference("sshleifer/tiny-gpt2", "main")
    with pytest.raises(ValueError):
        with StageRun("data", mode="offline", directory=tmp_path) as tracking:
            tracking.log({"loss": float("nan")})
    sdk[1].finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize("key", ["mode", "stage", "run_type", "resume", "config", "api_key", "unknown", "private-key-value"])
def test_tracking_options_reject_unknown_and_reserved(tmp_path, key):
    parser = argparse.ArgumentParser()
    add_tracking_args(parser)
    path = tmp_path / "options.json"
    path.write_text(json.dumps({key: "private-value"}))
    with pytest.raises(ValueError, match="Allowed") as error:
        from_args(parser.parse_args(["--wandb-config", str(path)]), "data")
    assert "private-value" not in str(error.value)
    assert "private-key-value" not in str(error.value)


@pytest.mark.parametrize("options", [{"directory": None}, {"project": 4}, {"run_id": []}, {"group": {}}])
def test_tracking_options_reject_invalid_types(tmp_path, options):
    parser = argparse.ArgumentParser()
    add_tracking_args(parser)
    path = tmp_path / "options.json"
    path.write_text(json.dumps(options))
    with pytest.raises(ValueError, match="Tracking option"):
        from_args(parser.parse_args(["--wandb-config", str(path)]), "data")


@pytest.mark.parametrize("failure", ["body", "log", "config", "artifact", "init"])
def test_original_exception_survives_cleanup_failures(sdk, tmp_path, monkeypatch, failure):
    wandb, run = sdk
    tracking = StageRun("data", mode="offline", directory=tmp_path)
    original = RuntimeError("original failure")
    if failure == "init":
        wandb.init.side_effect = original
    elif failure != "body":
        {"log": run.log, "config": run.config.update,
         "artifact": run.log_artifact}[failure].side_effect = original
    run.finish.side_effect = OSError("finish failure")
    receipt = tracking._receipt
    def fail_receipt(status):
        if status == "failed":
            raise OSError("receipt failure")
        receipt(status)
    monkeypatch.setattr(tracking, "_receipt", fail_receipt)
    with pytest.raises(RuntimeError) as error:
        with tracking:
            if failure == "body":
                raise original
            {"log": lambda: tracking.log({"x": 1}),
             "config": lambda: tracking.update_config({"x": 1}),
             "artifact": lambda: tracking.artifact_metadata("x", kind="metadata", metadata={})}[failure]()
    assert error.value is original
    assert tracking._failed
    assert json.loads(tracking.receipt.read_text())["status"] != "finished"
    if failure != "init":
        run.finish.assert_called_once_with(exit_code=1)


def test_post_sdk_init_failure_attempts_failed_finish(sdk, tmp_path, monkeypatch):
    tracking = StageRun("data", mode="offline", directory=tmp_path)
    original = OSError("running receipt failed")
    receipt = tracking._receipt
    def fail_receipt(status):
        if status in ("running", "failed"):
            raise original if status == "running" else OSError("failed receipt failed")
        receipt(status)
    monkeypatch.setattr(tracking, "_receipt", fail_receipt)
    sdk[1].finish.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(OSError) as error:
        tracking.init()
    assert error.value is original
    sdk[1].finish.assert_called_once_with(exit_code=1)
    assert tracking._failed


def test_finish_failure_preserved_when_receipt_also_fails(sdk, tmp_path, monkeypatch):
    tracking = StageRun("data", mode="offline", directory=tmp_path).init()
    original = RuntimeError("SDK finish failed")
    sdk[1].finish.side_effect = original
    monkeypatch.setattr(tracking, "_receipt", Mock(side_effect=OSError("receipt failed")))
    with pytest.raises(RuntimeError) as error:
        tracking.finish()
    assert error.value is original
    assert tracking._failed


def test_final_receipt_failure_is_not_finished(sdk, tmp_path, monkeypatch):
    tracking = StageRun("data", mode="offline", directory=tmp_path).init()
    receipt = tracking._receipt
    original = OSError("finished receipt failed")
    def fail_receipt(status):
        if status == "finished":
            raise original
        receipt(status)
    monkeypatch.setattr(tracking, "_receipt", fail_receipt)
    with pytest.raises(OSError) as error:
        tracking.finish()
    assert error.value is original
    assert tracking._failed
    assert json.loads(tracking.receipt.read_text())["status"] == "failed"


def test_identity_metadata_uses_recursive_key_sanitizer(sdk, tmp_path):
    with StageRun("data", mode="offline", directory=tmp_path,
                  checkpoint_id={"api_key": "private", "tokenizer": "fixture"}) as tracking:
        assert tracking.metadata["checkpoint_id"]["api_key"] == "[REDACTED]"
    assert "private" not in tracking.receipt.read_text()
    assert sdk[0].init.call_args.kwargs["config"]["tracking"] == tracking.metadata


@pytest.mark.parametrize("changes", [{"stage": "pretrain"}, {"run_type": "evaluation"},
                                    {"project": "other"}, {"entity": "other"}])
def test_resume_rejects_conflicting_local_stage_identity(sdk, tmp_path, changes):
    with StageRun("data", mode="offline", run_id="same", directory=tmp_path):
        pass
    with pytest.raises(ValueError, match="continuity"):
        options = {"stage": "data", **changes}
        StageRun(mode="offline", run_id="same", resume=True, directory=tmp_path, **options).init()
    assert sdk[0].init.call_count == 1


def test_resume_does_not_confuse_run_id_prefixes(sdk, tmp_path):
    with StageRun("data", mode="offline", run_id="same-prefix", directory=tmp_path):
        pass
    with StageRun("pretrain", mode="offline", run_id="same", resume=True, directory=tmp_path):
        pass
    assert sdk[0].init.call_count == 2


def test_real_offline_sdk(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with StageRun("data", mode="offline", directory=tmp_path, run_id="offline-unit") as tracking:
        tracking.log({"data/accepted_tokens": 12, "data/consumed_training_tokens": 0})
        tracking.artifact_metadata("offline-metadata", kind="dataset-metadata", metadata={"fixture": True})
    receipt = json.loads(next(tmp_path.glob("tracking-*.json")).read_text())
    assert receipt["status"] == "finished"
    assert "offline-run" in receipt["sdk_run_dir"]
    assert list(tmp_path.rglob("*.wandb"))


def test_real_offline_resume_segments_and_body_failure(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with StageRun("pretrain", mode="offline", directory=tmp_path, run_id="offline-resume") as first:
        first.log({"update": 1}, step=1)
    with StageRun("pretrain", mode="offline", directory=tmp_path,
                  run_id="offline-resume", resume=True) as resumed:
        resumed.log({"update": 2}, step=2)
    assert first.receipt != resumed.receipt
    receipts = [json.loads(path.read_text()) for path in tmp_path.glob("tracking-*.json")]
    assert len(receipts) == 2
    assert all(item["status"] == "finished" and item["run_id"] == "offline-resume" for item in receipts)
    assert len({item["sdk_run_dir"] for item in receipts}) == 2
    assert len(list(tmp_path.rglob("*.wandb"))) == 2
    original = RuntimeError("offline work failed")
    with pytest.raises(RuntimeError) as error:
        with StageRun("data", mode="offline", directory=tmp_path, run_id="offline-failure") as failed:
            raise original
    assert error.value is original
    assert json.loads(failed.receipt.read_text())["status"] == "failed"

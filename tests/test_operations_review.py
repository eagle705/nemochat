"""Regression checks for independent operations review findings."""
import pytest
from nemochat.planning import plan_training
from nemochat.artifacts.hf_checkpoint import safe_name, pack


def config():
    return dict(data_parallel=1, micro_batch_size=1, gradient_accumulation_steps=1,
                nodes=1, gpus_per_node=1, price_per_hour=1, price_unit="node",
                stages=[dict(name="test", sequence_length=1, target_tokens=1,
                             tokens_per_update=1, tokens_per_second=1,
                             measurement_provenance="arithmetic fixture")])


def test_sunk_overruns_are_not_fit():
    c = config()
    c.update(elapsed_seconds=100, spent_cost=100, time_limit_seconds=10, budget_limit=10)
    c["stages"][0]["completed_tokens"] = 1
    r = plan_training(c)["scenarios"]["base"]
    assert not r["fits_limits"]
    assert r["time_slack_seconds"] == -90
    assert r["budget_slack"] == -90


@pytest.mark.parametrize("events,expected", [(False,1000),(True,1100)])
def test_storage_includes_initial_footprint(events, expected):
    c = config()
    c["storage"] = dict(checkpoint_bytes=100, retain_checkpoints=2, existing_checkpoints=10)
    if events:
        c["stages"][0].update(timing_scope="core_compute", utilization=1,
                               checkpoint_every_updates=1, checkpoint_seconds=1)
    r = plan_training(c)["scenarios"]["base"]["storage"]
    assert r["peak_bytes_upper_bound"] == expected


def test_integer_update_rounding_above_float_precision():
    c = config()
    c["stages"][0]["target_tokens"] = 9007199254740993
    r = plan_training(c)["scenarios"]["base"]["stages"][0]
    assert r["updates"] == 9007199254740993
    assert r["rounded_update_tokens"] == 9007199254740993


@pytest.mark.parametrize("name", [".git/config", "sub/.git/config", ".cache/huggingface/x", "a/.cache/huggingface/x"])
def test_reserved_hub_names(name, tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        safe_name(name)
    source = tmp_path / "source"
    p = source / name
    p.parent.mkdir(parents=True)
    p.write_text("not a credential")
    with pytest.raises(ValueError, match="reserved"):
        pack(source, tmp_path / "bundle", kind="hf_model", stage="pretrain", run="fixture", checkpoint="test")
    assert not (tmp_path / "bundle").exists()

"""WSD policy and local random tiny-HF fixtures (not pretrained-model quality)."""

import json
from pathlib import Path
import sys

import pytest

from nemochat.training.wsd import WSDConfig, learning_rate_for_update

# Direct scripts are not an installed Python package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_cpu import main, save_checkpoint
from scripts.verify_cpu_resume import assert_same


def test_boundaries_and_floor():
    config = WSDConfig(2, 3, 2, 7, 0.1)
    results = [learning_rate_for_update(config, 1.0, step) for step in range(1, 8)]
    assert [lr for lr, _ in results] == pytest.approx([0.5, 1, 1, 1, 1, 0.55, 0.1])
    assert [phase for _, phase in results] == ["warmup"] * 2 + ["stable"] * 3 + ["decay"] * 2


def test_zero_length_optional_phases():
    assert learning_rate_for_update(WSDConfig(0, 2, 0, 2), 0.2, 1) == (0.2, "stable")
    assert learning_rate_for_update(WSDConfig(0, 1, 1, 2), 0.2, 2) == (0.0, "decay")


@pytest.mark.parametrize("counts", [(-1, 2, 1, 2), (1, 0, 1, 2), (1, 2, 1, 5),
                                   (True, 2, 1, 4), (1.0, 2, 1, 4)])
def test_invalid_counts(counts):
    with pytest.raises(ValueError):
        WSDConfig(*counts)


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_floor(ratio):
    with pytest.raises(ValueError):
        WSDConfig(1, 2, 1, 4, ratio)


@pytest.mark.parametrize("lr", [0, -1, float("nan"), float("inf")])
def test_invalid_peak(lr):
    with pytest.raises(ValueError):
        learning_rate_for_update(WSDConfig(1, 2, 1, 4), lr, 1)


@pytest.mark.parametrize("step", [0, -1, 5, True, 1.5])
def test_invalid_update(step):
    with pytest.raises(ValueError):
        learning_rate_for_update(WSDConfig(1, 2, 1, 4), 0.1, step)


@pytest.mark.parametrize("options", [
    ["--learning-rate", "nan"], ["--learning-rate", "inf"],
    ["--lr-schedule", "wsd"], ["--warmup-steps", "2"],
    ["--checkpoint-every", "2"], ["--checkpoint-dir", "/tmp/unused-wsd-fixture"],
    ["--checkpoint-every", "0", "--checkpoint-dir", "/tmp/unused-wsd-fixture"],
])
def test_cli_rejects_invalid_before_model_load(tmp_path, options):
    with pytest.raises(SystemExit) as exc:
        main(["--texts", str(tmp_path / "absent"), "--output", str(tmp_path / "out"), *options])
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.fixture
def local_hf_control(tmp_path):
    """Actual local HF model, but random fixture weights and tiny fixture tokenizer."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(7)
    source = tmp_path / "source"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
                                    "one": 3, "two": 4, "three": 5}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                          bos_token="[BOS]", eos_token="[EOS]")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_positions=8, n_embd=8,
                                     n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2))
    model.save_pretrained(source)
    hf_tokenizer.save_pretrained(source)
    texts = tmp_path / "texts.jsonl"
    texts.write_text('{"text": "one two three"}\n{"text": "three two one two"}\n')
    return ["--model", str(source), "--revision", "local-fixture", "--texts", str(texts),
            "--context-length", "8"]


def compare_checkpoints(left, right):
    import torch
    from safetensors.torch import load_file

    assert_same(load_file(str(left / "model.safetensors")),
                load_file(str(right / "model.safetensors")), "model")
    assert_same(torch.load(left / "optimizer_rng.pt", weights_only=True),
                torch.load(right / "optimizer_rng.pt", weights_only=True), "state")
    a, b = [json.loads((path / "run.json").read_text()) for path in (left, right)]
    for field in ("contract", "step", "cursor", "loss_tokens_seen"):
        assert_same(a[field], b[field], field)


@pytest.mark.parametrize("split", [1, 2, 3, 4])
def test_exact_wsd_resume_across_boundaries(local_hf_control, tmp_path, split):
    args = local_hf_control + ["--lr-schedule", "wsd", "--warmup-steps", "1",
                              "--stable-steps", "2", "--decay-steps", "2",
                              "--schedule-total-steps", "5", "--min-lr-ratio", "0.1"]
    first, resumed, continuous = [tmp_path / name for name in ("first", "resumed", "continuous")]
    main(args + ["--steps", str(split), "--output", str(first)])
    main(args + ["--steps", str(5 - split), "--resume", str(first), "--output", str(resumed)])
    main(args + ["--steps", "5", "--output", str(continuous)])
    compare_checkpoints(resumed, continuous)
    before, after, whole = [json.loads((path / "run.json").read_text())["history_this_invocation"]
                            for path in (first, resumed, continuous)]
    assert before + after == whole


def test_constant_contract_and_periodic_resume(local_hf_control, tmp_path):
    first, resumed, continuous, snapshots = [tmp_path / name for name in
                                             ("first", "resumed", "continuous", "snapshots")]
    main(local_hf_control + ["--steps", "2", "--output", str(first)])
    contract = json.loads((first / "run.json").read_text())["contract"]
    assert set(contract) == {"texts_sha256", "context_length", "learning_rate", "model", "revision"}
    main(local_hf_control + ["--steps", "3", "--output", str(continuous),
                            "--checkpoint-every", "2", "--checkpoint-dir", str(snapshots)])
    compare_checkpoints(first, snapshots / "step-00000002")
    main(local_hf_control + ["--steps", "1", "--resume", str(snapshots / "step-00000002"),
                            "--output", str(resumed)])
    compare_checkpoints(resumed, continuous)
    # Existing snapshot paths must fail before another update or final directory write.
    with pytest.raises(SystemExit):
        main(local_hf_control + ["--steps", "3", "--output", str(tmp_path / "collision"),
                                "--checkpoint-every", "2", "--checkpoint-dir", str(snapshots)])
    assert not (tmp_path / "collision").exists()


def test_wsd_contract_mismatch_and_horizon(local_hf_control, tmp_path):
    args = local_hf_control + ["--lr-schedule", "wsd", "--warmup-steps", "1",
                              "--stable-steps", "2", "--decay-steps", "1",
                              "--schedule-total-steps", "4"]
    first = tmp_path / "first"
    main(args + ["--steps", "2", "--output", str(first)])
    with pytest.raises(ValueError, match="config differs"):
        main(args + ["--min-lr-ratio", "0.2", "--resume", str(first), "--steps", "1",
                     "--output", str(tmp_path / "changed")])
    with pytest.raises(ValueError, match="config differs"):
        main(local_hf_control + ["--resume", str(first), "--steps", "1",
                                "--output", str(tmp_path / "constant")])
    with pytest.raises(SystemExit):
        main(args + ["--resume", str(first), "--steps", "3", "--output", str(tmp_path / "overrun")])


def test_tokenizer_id_swap_rejected_before_loading(local_hf_control, tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    first = tmp_path / "first"
    main(local_hf_control + ["--steps", "1", "--output", str(first)])
    path = first / "tokenizer.json"
    content = json.loads(path.read_text())
    vocab = content["model"]["vocab"]
    vocab["one"], vocab["two"] = vocab["two"], vocab["one"]
    path.write_text(json.dumps(content))

    def must_not_load(*args, **kwargs):
        pytest.fail("Checkpoint loader called before integrity verification")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", must_not_load)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", must_not_load)
    with pytest.raises(ValueError, match="SHA256"):
        main(local_hf_control + ["--steps", "1", "--resume", str(first),
                                "--output", str(tmp_path / "resumed")])


def test_save_preserves_rng_and_rejects_overwrite(tmp_path):
    import torch

    class ConsumesRNG:
        def save_pretrained(self, path, **kwargs):
            torch.rand(3)

    torch.manual_seed(5)
    rng = torch.get_rng_state().clone()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    path = tmp_path / "save"
    save_checkpoint(path, ConsumesRNG(), ConsumesRNG(), optimizer, {}, 0, 0, 0, None, [])
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(torch.load(path / "optimizer_rng.pt", weights_only=True)["torch_rng"], rng)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, ConsumesRNG(), ConsumesRNG(), optimizer, {}, 0, 0, 0, None, [])


@pytest.mark.parametrize("name", ["config.json", "generation_config.json", "model.safetensors",
                                  "optimizer_rng.pt", "tokenizer_config.json", "special_tokens_map.json"])
def test_artifact_tamper_before_any_load(local_hf_control, tmp_path, monkeypatch, name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    first = tmp_path / "first"
    main(local_hf_control + ["--steps", "1", "--output", str(first)])
    path = first / name
    path.write_bytes(path.read_bytes() + b" ")

    def must_not_load(*args, **kwargs):
        pytest.fail("Load attempted before verification")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", must_not_load)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", must_not_load)
    monkeypatch.setattr(torch, "load", must_not_load)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        main(local_hf_control + ["--steps", "1", "--resume", str(first),
                                "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["missing-inventory", "empty-inventory", "omitted-entry",
                                    "missing-file", "unexpected-file", "traversal", "symlink",
                                    "missing-both", "invalid-digest", "incomplete-required"])
def test_inventory_fail_closed(local_hf_control, tmp_path, change):
    first = tmp_path / "first"
    main(local_hf_control + ["--steps", "1", "--output", str(first)])
    manifest = json.loads((first / "run.json").read_text())
    inventory = manifest["artifacts_sha256"]
    if change == "missing-inventory":
        del manifest["artifacts_sha256"]
    elif change == "empty-inventory":
        manifest["artifacts_sha256"] = {}
    elif change == "omitted-entry":
        del inventory["tokenizer.json"]
    elif change == "missing-file":
        (first / "tokenizer.json").unlink()
    elif change == "unexpected-file":
        (first / "added_tokens.json").write_text("{}")
    elif change == "traversal":
        inventory["../outside"] = "0" * 64
    elif change == "symlink":
        (first / "tokenizer.json").unlink()
        (first / "tokenizer.json").symlink_to(tmp_path / "source" / "tokenizer.json")
    elif change == "missing-both":
        del manifest["checkpoint_format"]
        del manifest["artifacts_sha256"]
    elif change == "invalid-digest":
        inventory["tokenizer.json"] = "wrong"
    elif change == "incomplete-required":
        del inventory["config.json"]
        (first / "config.json").unlink()
    (first / "run.json").write_text(json.dumps(manifest))
    # Opt-in must never bypass a damaged new-format inventory.
    options = [] if change == "missing-both" else ["--allow-unverified-legacy-resume"]
    with pytest.raises(ValueError):
        main(local_hf_control + ["--steps", "1", "--resume", str(first),
                                "--output", str(tmp_path / "out"), *options])
    assert not (tmp_path / "out").exists()


def test_legacy_requires_explicit_opt_in(local_hf_control, tmp_path, capsys):
    first, resumed, continuous = [tmp_path / name for name in ("first", "resumed", "continuous")]
    main(local_hf_control + ["--steps", "2", "--output", str(first)])
    manifest = json.loads((first / "run.json").read_text())
    del manifest["checkpoint_format"]
    del manifest["artifacts_sha256"]
    for row in manifest["history_this_invocation"]:
        del row["learning_rate"]
        del row["lr_phase"]
    (first / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="allow-unverified-legacy-resume"):
        main(local_hf_control + ["--steps", "2", "--resume", str(first), "--output", str(resumed)])
    main(local_hf_control + ["--steps", "2", "--resume", str(first), "--output", str(resumed),
                            "--allow-unverified-legacy-resume"])
    assert "WARNING: unverified legacy resume" in capsys.readouterr().err
    main(local_hf_control + ["--steps", "4", "--output", str(continuous)])
    compare_checkpoints(resumed, continuous)
    assert json.loads((resumed / "run.json").read_text())["artifacts_sha256"]


def test_sharded_inventory_and_index_paths(local_hf_control, tmp_path):
    from transformers import AutoModelForCausalLM
    from scripts.train_cpu import artifact_sha256, checkpoint_artifacts, verify_checkpoint_inventory

    first = tmp_path / "first"
    main(local_hf_control + ["--steps", "1", "--output", str(first)])
    model = AutoModelForCausalLM.from_pretrained(first)
    (first / "model.safetensors").unlink()
    model.save_pretrained(first, max_shard_size="1KB")
    manifest = json.loads((first / "run.json").read_text())
    manifest["artifacts_sha256"] = {name: artifact_sha256(path)
                                    for name, path in checkpoint_artifacts(first).items()}
    verify_checkpoint_inventory(first, manifest)
    assert len([name for name in manifest["artifacts_sha256"] if name.endswith(".safetensors")]) > 1
    (first / "run.json").write_text(json.dumps(manifest))
    main(local_hf_control + ["--steps", "1", "--resume", str(first), "--output", str(tmp_path / "resumed")])
    index_path = first / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"][next(iter(index["weight_map"]))] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))
    manifest["artifacts_sha256"][index_path.name] = artifact_sha256(index_path)
    with pytest.raises(ValueError, match="shard index"):
        verify_checkpoint_inventory(first, manifest)

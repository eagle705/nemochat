"""CPU controls for selected-record DCP export; no full-model integrity audit."""
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner


@pytest.mark.filterwarnings("ignore:torch.distributed is disabled")
def test_exports_model_only_dcp_preserving_tensors_and_extra_state(tmp_path):
    from nemochat.hub.base_checkpoint import export_model_dcp

    source = tmp_path / "source"
    original = {
        "embedding.word_embeddings.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "decoder.layers.0.mlp.experts.weight": torch.arange(32, dtype=torch.bfloat16).reshape(2, 4, 4),
        "decoder.layers.0.router.expert_bias": torch.ones(2, dtype=torch.float32),
        "decoder.layers.0._extra_state/shard_0_1": b"native-extra-state",
        "output_layer.weight": torch.ones(3, 4, dtype=torch.bfloat16),
        "chained_0.optimizer.state": torch.ones(100),
        "rng_state/shard_0": b"rng-not-model",
        "common_state/shard_0_1": {"step": 123},
    }
    dcp.save(original, checkpoint_id=source, planner=DefaultSavePlanner(flatten_state_dict=False))
    output = tmp_path / "export"
    receipt = export_model_dcp(source, output, shard_limit=5120)
    reader = dcp.FileSystemReader(output)
    metadata = reader.read_metadata()
    expected = {k: v for k, v in original.items() if k.startswith(("embedding.", "decoder.", "output_layer."))}
    assert set(metadata.state_dict_metadata) == set(expected)
    loaded = {k: torch.empty_like(v) if isinstance(v, torch.Tensor) else None for k, v in expected.items()}
    dcp.load(loaded, storage_reader=reader, planner=DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False))
    for k, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, loaded[k])
        else:
            assert value == loaded[k]
    assert receipt["checkpoint_integrity"] == "not_checked_explicit_user_waiver"
    assert receipt["tensor_numel"] == 58
    assert receipt["model_keys"] == 5
    assert receipt["excluded_keys"] == 3
    assert sum(p.stat().st_size for p in output.glob("*.distcp")) == receipt["encoded_model_bytes"]
    with pytest.raises(FileExistsError):
        export_model_dcp(source, output)


@pytest.mark.filterwarnings("ignore:torch.distributed is disabled")
def test_export_has_empty_megatron_common_state_not_training_state(tmp_path):
    from nemochat.hub.base_checkpoint import export_model_dcp

    source = tmp_path / "source"
    dcp.save({"embedding.weight": torch.ones(1), "common_state/shard_0_1": {"optimizer": "excluded"}},
             checkpoint_id=source, planner=DefaultSavePlanner(flatten_state_dict=False))
    (source / "metadata.json").write_text('{"sharded_backend": "torch_dist", "sharded_backend_version": 1}')
    export_model_dcp(source, tmp_path / "export")
    assert torch.load(tmp_path / "export" / "common.pt", weights_only=True) == {}
    assert (tmp_path / "export" / "metadata.json").read_text() == (source / "metadata.json").read_text()


@pytest.mark.filterwarnings("ignore:torch.distributed is disabled")
def test_export_rejects_storage_symlink_ancestor_before_creating_output(tmp_path):
    """Trusted metadata is still not permission to follow storage-directory links."""
    import dataclasses
    import pickle
    from nemochat.hub.base_checkpoint import export_model_dcp

    source = tmp_path / "source"
    dcp.save({"embedding.weight": torch.ones(1)}, checkpoint_id=source,
             planner=DefaultSavePlanner(flatten_state_dict=False))
    metadata = dcp.FileSystemReader(source).read_metadata()
    outside = tmp_path / "outside"
    outside.mkdir()
    for storage in metadata.storage_data.values():
        (source / storage.relative_path).rename(outside / storage.relative_path)
    (source / "linked").symlink_to(outside, target_is_directory=True)
    metadata.storage_data = {
        key: dataclasses.replace(value, relative_path="linked/" + value.relative_path)
        for key, value in metadata.storage_data.items()}
    with (source / ".metadata").open("wb") as handle:
        pickle.dump(metadata, handle)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="unsafe source storage path"):
        export_model_dcp(source, output)
    assert not output.exists()


@pytest.mark.parametrize("private,files", [(False, []), (True, ["README.md"])])
def test_refuses_existing_public_or_occupied_repo(private, files):
    from types import SimpleNamespace
    from nemochat.hub.base_checkpoint import initialize_private_repo

    class Api:
        def whoami(self):
            return {"name": "eagle705"}
        def repo_info(self, **kwargs):
            assert kwargs["repo_id"] == "eagle705/nemochat-10BA1B-BF16-base"
            return SimpleNamespace(private=private, siblings=[SimpleNamespace(rfilename=f) for f in files], sha="a" * 40)
        def create_repo(self, **kwargs):
            pytest.fail("must not change an existing repository")

    with pytest.raises(ValueError):
        initialize_private_repo(Api())


def test_remote_readback_requires_private_exact_file_sizes(tmp_path):
    from types import SimpleNamespace
    from nemochat.hub.base_checkpoint import verify_remote

    (tmp_path / "README.md").write_text("base-card")
    (tmp_path / "model.distcp").write_bytes(b"tiny-synthetic")
    class Api:
        def repo_info(self, **kwargs):
            assert kwargs["revision"] == "a" * 40
            return SimpleNamespace(private=True, sha="a" * 40, siblings=[
                SimpleNamespace(rfilename=".gitattributes", size=10),
                SimpleNamespace(rfilename="README.md", size=9),
                SimpleNamespace(rfilename="model.distcp", size=14),
            ])
    assert verify_remote(Api(), tmp_path, "a" * 40)["private"] is True
    (tmp_path / "model.distcp").write_bytes(b"wrong-size")
    with pytest.raises(ValueError, match="files or sizes"):
        verify_remote(Api(), tmp_path, "a" * 40)

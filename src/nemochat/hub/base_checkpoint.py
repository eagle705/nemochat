"""Model-only native DCP publication under an explicit checkpoint-byte waiver.

This is storage-range repacking, NOT Transformers conversion, model loading,
checkpoint integrity verification, or an optimizer-resumable training bundle.
Only the already serialized model records are read, once, to create the requested
export. Tensor/global chunk metadata and native extra-state records are retained.
"""
from __future__ import annotations

import copy
import dataclasses
import pickle
from contextlib import ExitStack
from pathlib import Path

MODEL_PREFIXES = ("embedding.", "decoder.", "output_layer.")
REPO_ID = "eagle705/nemochat-10BA1B-BF16-base"
WAIVER = "not_checked_explicit_user_waiver"


def initialize_private_repo(api):
    """Create only the authorized repo; never change visibility or overwrite data."""
    from huggingface_hub.errors import RepositoryNotFoundError

    if api.whoami().get("name") != "eagle705":
        raise ValueError("authenticated identity does not own the authorized namespace")
    try:
        info = api.repo_info(repo_id=REPO_ID, repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=False)
        info = api.repo_info(repo_id=REPO_ID, repo_type="model")
    if info.private is not True:
        raise ValueError("target is not private; visibility is never silently changed")
    if {s.rfilename for s in info.siblings} - {".gitattributes"}:
        raise ValueError("target is occupied; refusing overwrite")
    return info.sha


def verify_remote(api, folder: Path, revision: str) -> dict:
    """Read immutable remote privacy/file sizes; deliberately do not reread weights."""
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("full immutable Hub commit required")
    expected = {p.relative_to(folder).as_posix(): p.stat().st_size
                for p in folder.rglob("*") if p.is_file()}
    info = api.repo_info(repo_id=REPO_ID, repo_type="model", revision=revision, files_metadata=True)
    if info.private is not True or info.sha != revision:
        raise ValueError("remote privacy or immutable revision mismatch")
    actual = {s.rfilename: s.size for s in info.siblings if s.rfilename != ".gitattributes"}
    if actual != expected:
        raise ValueError("remote files or sizes differ from the authorized payload")
    return {"repo_id": REPO_ID, "revision": revision, "private": True,
            "payload_files": expected, "payload_file_count": len(expected),
            "payload_bytes": sum(expected.values()), "checkpoint_integrity": WAIVER,
            "verification": "immutable_revision_private_exact_file_names_and_sizes",
            "independent_weight_download_or_hash": False}



def export_model_dcp(source: Path, output: Path, shard_limit: int = 4_000_000_000) -> dict:
    """Repack selected encoded model records without deserializing weight tensors.

    Trusted native metadata is pickled. Never use with untrusted checkpoints.
    Files are fresh, shard size is a soft limit (a single record is indivisible).
    No native SHA/CRC, independent weight reread, or optimizer copy is performed.
    """
    from torch.distributed.checkpoint import FileSystemReader

    source, output = Path(source), Path(output)
    if shard_limit <= 0:
        raise ValueError("positive shard limit required")
    if output.exists():
        raise FileExistsError(output)
    metadata = FileSystemReader(source).read_metadata()
    selected = {k: v for k, v in metadata.state_dict_metadata.items() if k.startswith(MODEL_PREFIXES)}
    if not selected or not any(k.startswith("embedding.") for k in selected):
        raise ValueError("expected native model key roots absent")
    records = sorted(((index, info) for index, info in metadata.storage_data.items() if index.fqn in selected),
                     key=lambda item: (item[1].relative_path, item[1].offset))
    for _, info in records:
        rel = Path(info.relative_path)
        if (rel.is_absolute() or ".." in rel.parts or
                any((source / parent).is_symlink() for parent in (rel, *rel.parents))):
            raise ValueError("unsafe source storage path")
    output.mkdir(parents=True, exist_ok=False)
    new_storage = {}
    files = []
    offset, shard_number, writer = 0, 0, None
    with ExitStack() as stack:
        readers = {}
        for index, info in records:
            if writer is None or (offset and offset + info.length > shard_limit):
                if writer is not None:
                    writer.close()
                filename = f"model_{shard_number:05d}.distcp"
                writer = stack.enter_context((output / filename).open("xb"))
                files.append(filename)
                shard_number += 1
                offset = 0
            if info.relative_path not in readers:
                readers[info.relative_path] = stack.enter_context((source / info.relative_path).open("rb"))
            reader = readers[info.relative_path]
            reader.seek(info.offset)
            remaining = info.length
            while remaining:
                data = reader.read(min(8 * 1024 * 1024, remaining))
                if not data:
                    raise EOFError("native storage record ended early; export incomplete")
                writer.write(data)
                remaining -= len(data)
            new_storage[index] = dataclasses.replace(info, relative_path=filename, offset=offset)
            offset += info.length
    exported = copy.copy(metadata)
    exported.state_dict_metadata = selected
    exported.storage_data = new_storage
    exported.storage_meta = None
    exported.planner_data = None
    if hasattr(exported, "all_local_plans"):
        exported.all_local_plans = None
    with (output / ".metadata").open("xb") as stream:
        pickle.dump(exported, stream, protocol=4)
    # Megatron checkpoint format/version marker, not training state.
    if (source / "metadata.json").is_file():
        (output / "metadata.json").write_bytes((source / "metadata.json").read_bytes())
    # Empty legacy common state is accepted by Megatron load_common_state_dict.
    # It is newly generated model-only metadata, not copied training state.
    import torch
    torch.save({}, output / "common.pt")
    tensors = [v for v in selected.values() if hasattr(v, "size")]
    return {
        "format": "megatron_native_model_only_torch_dist",
        "checkpoint_integrity": WAIVER,
        "model_keys": len(selected),
        "excluded_keys": len(metadata.state_dict_metadata) - len(selected),
        "tensor_numel": sum(v.size.numel() for v in tensors),
        "tensor_dtypes": {str(dtype): sum(v.size.numel() for v in tensors if v.properties.dtype == dtype)
                          for dtype in sorted({v.properties.dtype for v in tensors}, key=str)},
        "encoded_model_bytes": sum(info.length for _, info in records),
        "storage_records": len(records),
        "storage_files": files,
        "weight_tensor_deserialization": False,
        "optimizer_included": False,
        "transformers_compatible": False,
    }

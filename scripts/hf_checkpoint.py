#!/usr/bin/env python3
"""Checkpoint packaging and immutable Hub transfer CLI (offline by default)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.artifacts.hf_checkpoint import (KINDS, pack, verify, restore_local,
                                            upload, download, read_json, prefix_for)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="Offline packaging: no credentials or network")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=sorted(KINDS), required=True)
    for name in ("stage", "run", "checkpoint"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--components", type=Path, help="JSON role-to-file-list declaration (training only)")
    p = sub.add_parser("verify")
    p.add_argument("--source", required=True)
    p.add_argument("--kind", choices=sorted(KINDS), required=True)
    p = sub.add_parser("restore-local")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=sorted(KINDS), required=True)
    p.add_argument("--trusted-inputs", action="store_true")
    p = sub.add_parser("upload")
    p.add_argument("--source", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--dry-run", action="store_true", help="Only local inventory verification; no Hub import or credentials")
    p.add_argument("--rights-reviewed", action="store_true")
    p.add_argument("--repo-policy", type=Path)
    p = sub.add_parser("download")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--revision", required=True, help="Full immutable commit SHA; not main/tag")
    p.add_argument("--path", required=True, help="Exact stages/... checkpoint prefix")
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=sorted(KINDS), required=True)
    p.add_argument("--trusted-inputs", action="store_true")
    p.add_argument("--manifest-sha256", help="Expected manifest digest from saved upload receipt")
    p.add_argument("--rights-reviewed", action="store_true")
    p.add_argument("--repo-policy", type=Path)
    a = parser.parse_args()
    try:
        if a.command == "pack":
            result = pack(a.source, a.output, kind=a.kind, stage=a.stage, run=a.run,
                          checkpoint=a.checkpoint, components=read_json(a.components) if a.components else None)
        elif a.command == "verify":
            result = verify(a.source, a.kind)
        elif a.command == "restore-local":
            result = {"restored": restore_local(a.source, a.output, expected_kind=a.kind, trusted_inputs=a.trusted_inputs)}
        elif a.command == "upload" and a.dry_run:
            m = verify(a.source)
            result = {"dry_run": True, "remote_upload_exercised": False, "path": prefix_for(m), "files": len(m["files"])}
        else:
            policy = read_json(a.repo_policy) if a.repo_policy else None
            kwargs = {"rights_reviewed": a.rights_reviewed, "repo_policy": policy}
            if a.command == "upload":
                result = upload(a.source, a.repo_id, **kwargs)
            else:
                result = download(a.repo_id, a.revision, a.path, a.output, expected_kind=a.kind,
                                  trusted_inputs=a.trusted_inputs, expected_manifest_hash=a.manifest_sha256, **kwargs)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception:
        # SDK exceptions may contain auth headers/URLs. Do not echo them or
        # enable debug logging around credentials. Programmatic callers can
        # inspect failures within their own secure environment.
        print("Checkpoint operation failed; no verified success receipt issued.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

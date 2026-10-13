"""CPU control, verified snippet inspection, and version-aware Nsight launch."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .bridge import Bridge
from .source import extract_verified

BASE = Path(__file__).resolve().parents[1]


def probe(command, env=None):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20, env=env)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": type(exc).__name__}


def plugin_environment():
    env = os.environ.copy()
    own = str(BASE / "nsight")
    inherited = env.get("NSYS_PLUGIN_SEARCH_DIRS", "")
    env["NSYS_PLUGIN_SEARCH_DIRS"] = own + (os.pathsep + inherited if inherited else "")
    return env


def doctor():
    report = {"nsys_path": shutil.which("nsys"), "native_ui_panel": "pending",
              "plugin_kind": "configuration_only_explicit_import"}
    for package in ("torch", "nemo-lens", "opentelemetry-sdk"):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = "not installed"
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        report["cuda_available"] = False
    nsys = report["nsys_path"]
    if nsys:
        env = plugin_environment()
        report["version"] = probe([nsys, "--version"], env)
        report["profile_help"] = probe([nsys, "profile", "--help"], env)
        report["plugins_help"] = probe([nsys, "plugins", "--help"], env)
        report["plugins_list"] = probe([nsys, "plugins", "list"], env)
        text = report["profile_help"].get("stdout", "") + report["profile_help"].get("stderr", "")
        report["backtrace_help_mentions"] = [line for line in text.splitlines()
                                             if "backtrace" in line.lower() or "sampling" in line.lower()]
    else:
        report["capture_blocker"] = "nsys absent; plugin discovery/help cannot be validated"
    return report


def launch(args):
    report = doctor()
    if not report["nsys_path"]:
        raise RuntimeError("nsys absent: install compatible Nsight Systems and run doctor")
    help_result = report["profile_help"]
    help_text = help_result.get("stdout", "") + help_result.get("stderr", "")
    listing = report["plugins_list"]
    if (help_result.get("returncode") != 0 or "--enable" not in help_text
            or listing.get("returncode") != 0
            or "nemo_lens_nsight_bridge" not in listing.get("stdout", "")
            or "(error)" in listing.get("stdout", "")):
        raise RuntimeError("Installed Nsight help/plugin validation failed; inspect doctor output")
    if not report["cuda_available"]:
        raise RuntimeError("CUDA capture requires working GPU; CPU validation is separate")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Separate prefix avoids Nsight sidecars inside the bridge's fresh output directory.
    prefix = output.parent / (output.name + "-nsys")
    if any(output.parent.glob(prefix.name + "*")):
        raise FileExistsError("Nsight output prefix already exists")
    command = [report["nsys_path"], "profile", "--enable=nemo_lens_nsight_bridge",
               "--trace=cuda,nvtx", "--output=" + str(prefix), sys.executable,
               "-m", "lens_nsight_bridge", "run", "--device=cuda", "--lens",
               "--output=" + str(output)]
    print(json.dumps({"command": command, "warning": "Diagnostic capture; no performance claim"}), flush=True)
    return subprocess.run(command, env=plugin_environment(), cwd=BASE).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    run = sub.add_parser("run", help="Educational control; fresh local output only")
    run.add_argument("--output", required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--lens", action="store_true", help="Real pinned Lens local spans; never OTLP")
    run.add_argument("--nvtx", action="store_true", help="Require working CUDA NVTX (no silent fallback)")
    snippet = sub.add_parser("snippet", help="Extract real hash-verified captured source")
    snippet.add_argument("--output", required=True)
    snippet.add_argument("--bridge-id", required=True)
    snippet.add_argument("--current-root", help="Also verify live checkout against captured hash")
    capture = sub.add_parser("launch", help="Version/plugin checked Nsight CUDA educational control")
    capture.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(), indent=2))
        elif args.command == "launch":
            sys.exit(launch(args))
        elif args.command == "snippet":
            records = [json.loads(line) for line in (Path(args.output) / "operations.jsonl").read_text().splitlines()]
            matches = [record for record in records if record["bridge_id"] == args.bridge_id]
            if len(matches) != 1:
                raise ValueError("Bridge ID must identify exactly one explicit operation")
            ref = matches[0]["source"]
            text = extract_verified(args.output, ref, args.current_root)
            print(f"# verified snapshot: {ref['path']}:{ref['start_line']}-{ref['end_line']} sha256={ref['file_sha256']}")
            print(text, end="")
        else:
            from .control import run as run_control  # explicit application instrumentation import
            enabled = os.environ.get("NEMO_LENS_NSIGHT_BRIDGE") == "1"
            nvtx = args.nvtx or (enabled and os.environ.get("NEMO_LENS_NSIGHT_NVTX") == "1")
            with Bridge(args.output, BASE, nvtx=nvtx, lens=args.lens or enabled) as bridge:
                result = run_control(bridge, args.device)
            result.update({"nvtx_requested": nvtx, "lens_requested": args.lens or enabled})
            with (Path(args.output) / "result.json").open("x") as fh:
                json.dump(result, fh, indent=2)
            print(json.dumps(result, indent=2))
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"bridge error: {exc}\n")

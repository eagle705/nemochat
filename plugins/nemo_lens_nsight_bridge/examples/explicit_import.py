"""Copy this pattern into user code; no sitecustomize or global monkeypatch."""
import argparse
import os
from pathlib import Path

from lens_nsight_bridge import Bridge
from lens_nsight_bridge.control import run

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
if os.environ.get("NEMO_LENS_NSIGHT_BRIDGE") != "1":
    raise RuntimeError("Explicit demo expects enabled configuration plugin/environment")
with Bridge(args.output, Path(__file__).resolve().parents[1], lens=True,
            nvtx=os.environ.get("NEMO_LENS_NSIGHT_NVTX") == "1") as bridge:
    print(run(bridge, device="cuda"))

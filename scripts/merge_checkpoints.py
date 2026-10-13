#!/usr/bin/env python3
"""Launch pinned official Megatron WSM; see docs/en/WSM_IMPLEMENTATION.md."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.checkpoints.wsm import main

if __name__ == "__main__":
    main()

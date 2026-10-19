"""Real-process relative script identity regression, not GPU evidence."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest


def test_owned_relative_script_matches_absolute_path_but_not_other_directory(tmp_path):
    spec = importlib.util.spec_from_file_location('relative_identity_control', 'sft/training/nemotron_48h_pipeline.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    script = tmp_path / 'owned_worker.py'
    script.write_text("import time\nprint('READY', flush=True)\ntime.sleep(10)\n")
    child = subprocess.Popen([sys.executable, script.name], cwd=tmp_path,
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'READY'
        identity = controller.process_identity(child.pid, str(script))
        assert identity['pid'] == child.pid
        with pytest.raises(ValueError, match='argv'):
            controller.process_identity(child.pid, str(tmp_path / 'other' / script.name))
    finally:
        child.terminate()
        child.wait(timeout=5)

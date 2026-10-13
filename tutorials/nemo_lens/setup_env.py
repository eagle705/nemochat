"""Create only this tutorial's venv; borrow project torch read-only via .pth."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
project_python = root.parents[1] / ".venv/bin/python"
if not project_python.exists():
    raise SystemExit("Install the project's CPU .venv first; see root README")
if (root / ".venv").exists():
    raise SystemExit("Tutorial .venv already exists; refusing to overwrite")
subprocess.run(["uv", "venv", "--python", str(project_python), str(root / ".venv")], check=True)
python = root / ".venv/bin/python"
subprocess.run(["uv", "pip", "install", "--python", str(python), "-r", str(root / "requirements.lock")], check=True)
project_site = subprocess.check_output([str(project_python), "-c", "import site; print(site.getsitepackages()[0])"], text=True).strip()
tutorial_site = subprocess.check_output([str(python), "-c", "import site; print(site.getsitepackages()[0])"], text=True).strip()
Path(tutorial_site, "project_torch.pth").write_text(project_site + "\n")
subprocess.run([str(python), "-c", "import torch,nemo.lens; print(torch.__version__,nemo.lens.__version__)"], check=True)

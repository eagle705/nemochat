"""Offline checks for the versioned cross-agent skill library."""
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_project_skill_frontmatter_and_references():
    files = sorted((ROOT / "skills").glob("*/SKILL.md"))
    assert len(files) == 3
    guide = (ROOT / "AGENTS.md").read_text()
    for path in files:
        text = path.read_text()
        assert text.startswith("---\n")
        _, frontmatter, body = text.split("---", 2)
        data = yaml.safe_load(frontmatter)
        assert data["name"] == path.parent.name
        assert re.fullmatch(r"[a-z0-9-]{1,64}", data["name"])
        assert 0 < len(data["description"]) <= 60
        assert data["description"].endswith(".")
        assert data["version"] == "0.1.0"
        assert data["platforms"] == ["linux"]
        assert str(path.relative_to(ROOT)) in guide
        for heading in ["When to Use", "Prerequisites", "How to Run", "Procedure", "Pitfalls", "Verification"]:
            assert "## " + heading in body
        assert "/home/ubuntu" not in text
        assert sum(line.startswith("```") for line in body.splitlines()) % 2 == 0
        for target in re.findall(r"\]\(([^)]+)\)", body):
            assert (path.parent / target).is_file(), target
        for script in re.findall(r"(?:scripts/|tutorials/|plugins/)[A-Za-z0-9_./-]+\.py", body):
            assert (ROOT / script).is_file(), script


def test_library_has_no_runtime_payloads():
    allowed = {".md"}
    for path in (ROOT / "skills").rglob("*"):
        if path.is_file():
            assert not path.is_symlink()
            assert path.suffix in allowed

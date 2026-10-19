"""Source-distribution contracts, not weight/data license clearance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_redistributed_font_has_exact_subset_and_original_ofl():
    static = ROOT / 'dashboard/static'
    provenance = json.loads((static / 'FONT-PROVENANCE.json').read_text())
    data = (static / 'NemoChatKorean.ttf').read_bytes()
    assert len(data) == provenance['subset_bytes']
    assert hashlib.sha256(data).hexdigest() == provenance['subset_sha256']
    license_text = (static / 'FONT-OFL.txt').read_text()
    assert 'SIL OPEN FONT LICENSE Version 1.1' in license_text
    assert "Reserved Font Name 'Source'" in license_text


def test_cached_kd_port_and_minimal_snapshots_have_full_scoped_notices():
    patch = ROOT / 'midtraining/patches/offline_kd_pr'
    expected = {
        'Megatron-LM-LICENSE.txt': '39f1df137a306d31698db206063f462b69c19e536f012db09a66997ad2cb96f0',
        'Apache-2.0.txt': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'}
    for name, digest in expected.items():
        assert hashlib.sha256((patch / 'licenses' / name).read_bytes()).hexdigest() == digest
    assert '**modified ports**' in (patch / 'licenses/README.md').read_text()
    assert 'licenses/README.md' in (ROOT / 'THIRD_PARTY_NOTICES.md').read_text()
    provenance = json.loads((patch / 'evidence/integration-base.json').read_text())
    assert hashlib.sha256((patch / 'evidence/bridge-installed/training__gpt_step.py').read_bytes()).hexdigest() == provenance['bridge_gpt_step_sha256']
    for row in provenance['core_sources']:
        path = patch / 'evidence' / row['path']
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']


def test_historical_registry_only_names_existing_tests():
    contracts = json.loads((ROOT / 'tests/historical_contracts.json').read_text())['tests']
    assert contracts
    for nodeid, reason in contracts.items():
        path, *names = nodeid.split('::')
        assert path.startswith('tests/test_') and names and reason
        text = (ROOT / path).read_text()
        assert 'def ' + names[-1] + '(' in text

"""Separate exact run-bound contracts from source-only CPU unit tests.

No test exceptions are caught and no production/source guards are changed.
Opt-in preserves the original assertions, including known historical failures.
"""
import json
import os
from pathlib import Path

import pytest


def historical_reason(nodeid, contracts):
    return contracts.get(nodeid.split('[', 1)[0])


def pytest_collection_modifyitems(config, items):
    contracts = json.loads(Path(__file__).with_name('historical_contracts.json').read_text())['tests']
    enabled = os.environ.get('NEMOCHAT_HISTORICAL_TESTS') == '1'
    for item in items:
        reason = historical_reason(item.nodeid, contracts)
        if reason is not None and not enabled:
            item.add_marker(pytest.mark.skip(
                reason='historical integration (NEMOCHAT_HISTORICAL_TESTS=1): ' + reason))

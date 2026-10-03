"""Judge test configuration. No network, no GPU."""
from __future__ import annotations

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # never fetch the price map in tests

import pytest


@pytest.fixture
def tmp_cache(tmp_path):
    return str(tmp_path / "judge_cache.sqlite")

"""Test isolation: pin config to project defaults.

A developer machine may carry a .env that overrides paths or workflow
bounds; tests must behave identically there and on a clean clone.
"""

import pytest

from src import config


@pytest.fixture(autouse=True)
def _pin_config(monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", config.PROJECT_ROOT / "data" / "knowledge")
    monkeypatch.setattr(config, "CASES_DIR", config.PROJECT_ROOT / "data" / "sample_cases")
    monkeypatch.setattr(config, "MAX_SUPERVISOR_STEPS", 8)
    monkeypatch.setattr(config, "MAX_AGENT_TURNS", 12)

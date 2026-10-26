"""Shared fixtures for mesas test suite."""

import os

import pytest


# Ensure tests can find data files regardless of where pytest is invoked from
@pytest.fixture(autouse=True)
def _chdir_to_repo_root(monkeypatch):
    """Change to the repo root so relative paths in tests and data files work."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(repo_root)

"""
Shared pytest fixtures and path setup for the QA Agent test suite.

The project imports ``agents`` and ``agents_core`` as top-level packages,
so we prepend the project root to ``sys.path`` before any test file
gets imported. We also seed ``os.environ`` with safe defaults — this
must happen at *module load time* (i.e. before any test module is
imported) because ``agents_core.models`` instantiates ``ChatOpenAI``
at import time and validates its constructor arguments.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap — must come first.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Environment defaults — applied at import time so that
# ``from agents_core import models`` succeeds during test collection.
# Real keys (if any) win: we only fill in missing entries.
# ---------------------------------------------------------------------------
_ENV_DEFAULTS = {
    "llm_base_url": "https://example.invalid/v1",
    "api_key": "test-key",
    "api_model": "test-model",
    "openrouter_base_url": "https://example.invalid/v1",
    "openrouter_api": "test-key",
    "embedding_model": "text-embedding-3-small",
    "telegram_bot_api": "test-bot",
    "telegram_user_id": "0",
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-test",
    "LANGFUSE_BASE_URL": "http://localhost:3000",
    "rec_limit": "50",
}
for _key, _value in _ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)


# ---------------------------------------------------------------------------
# Sandbox for the data/ tree
# ---------------------------------------------------------------------------
@pytest.fixture
def data_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Provide a clean ``data/`` directory and chdir into it.

    The agents' tools accept absolute paths but normalise them to be
    *under* ``data/``. Creating a fresh directory per test keeps the
    tests hermetic and avoids accidental file lookups in the repo's
    real ``data/inputs`` or ``data/outputs`` folders.
    """
    sandbox = tmp_path / "data"
    (sandbox / "inputs").mkdir(parents=True)
    (sandbox / "outputs" / "tests").mkdir(parents=True)
    (sandbox / "outputs" / "docs").mkdir(parents=True)

    # ``agents_core.tools`` uses ``os.path.isfile`` on the *relative* path
    # after stripping ``/app/`` — so we chdir into the sandbox root for
    # the duration of the test.
    monkeypatch.chdir(tmp_path)
    yield sandbox

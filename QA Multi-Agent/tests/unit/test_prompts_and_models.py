"""
Unit tests for the static, non-LLM pieces of the project:

* ``agents_core.prompts`` — every prompt file must be loadable and
  non-empty, and each one must mention the agent it drives;
* ``agents_core.models`` — module imports without raising and exposes
  the four expected model slots;
* ``agents_core.logger`` — the logger is configured with a file
  handler and at INFO level.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agents_core import prompts
from agents_core import models
from agents_core import logger as core_logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# prompts.py
# ---------------------------------------------------------------------------
# Each agent's prompt must contain at least one of these role-marker
# strings. The prompts are written in a free-form, conversational style
# and not all of them repeat the agent's exact name verbatim.
EXPECTED_ROLE_MARKERS = {
    "supervisor_prompt": ("supervisor", "orchestrat", "router", "team lead"),
    "test_explorer_prompt": (
        "test automation",
        "auto test",
        "playwright",
        "test explorer",
        "test_explorer",
    ),
    "reviewer_prompt": (
        "reviewer",
        "review",
        "verdict",
        "codex",
    ),
    "writer_prompt": (
        "writer",
        "write",
        "documentation",
        "readme",
        "report",
    ),
}


@pytest.mark.parametrize("attr", list(EXPECTED_ROLE_MARKERS))
def test_prompt_attribute_exists(attr: str):
    """Every expected prompt slot is populated and contains some content."""
    text = getattr(prompts, attr)
    assert isinstance(text, str)
    assert len(text) > 200, f"{attr} looks suspiciously short"


@pytest.mark.parametrize("attr", list(EXPECTED_ROLE_MARKERS))
def test_prompt_content_mentions_the_agent_role(attr: str):
    """Each prompt must self-identify the role it drives."""
    text = getattr(prompts, attr).lower()
    markers = EXPECTED_ROLE_MARKERS[attr]
    assert any(m in text for m in markers), (
        f"{attr} does not mention any of {markers!r}"
    )


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------
def test_models_module_exposes_expected_slots():
    """All four model bindings must be importable as attributes."""
    for slot in ("supervisor_model", "test_explorer_model", "reviewer_model", "writer_model"):
        assert hasattr(models, slot), f"models.{slot} missing"


def test_models_module_uses_a_shared_sample(monkeypatch: pytest.MonkeyPatch):
    """The model_sample is shared by all four slots — they should be the same object."""
    # ``model_sample`` is created at import time; we just assert that the
    # four agent slots are bound to the same underlying object.
    assert (
        models.supervisor_model
        is models.test_explorer_model
        is models.reviewer_model
        is models.writer_model
    )


# ---------------------------------------------------------------------------
# logger.py
# ---------------------------------------------------------------------------
def test_logger_is_configured_with_file_handler():
    """The agent logger writes to ``logs/agent.log`` at INFO level."""
    assert core_logger.logger.name == "agent"
    assert core_logger.logger.level == logging.INFO
    assert any(
        h.__class__.__name__ == "TimedRotatingFileHandler"
        for h in core_logger.logger.handlers
    )


def test_logger_creates_logs_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The logger module creates ``logs/`` on import; it must be a directory."""
    # We can't easily undo the side-effect of the import-time mkdir, so
    # we just assert the directory exists in the current working dir.
    assert (Path.cwd() / "logs").is_dir() or (PROJECT_ROOT / "logs").is_dir()

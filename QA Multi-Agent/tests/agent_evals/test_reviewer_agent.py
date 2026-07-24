"""
Unit tests for ``agents.agents.reviewer_agent``.

The reviewer is the auditor — it statically compares a generated
``*.spec.ts`` with its test case and returns a verdict in one of
three forms. The supervisor parses the verdict back out, so the
output structure is part of the public contract:

* a ``## Verdict`` section with ``**Decision:**`` and an emoji
  (✅/❌/⚠️) is mandatory;
* the report names at least one of the section headers
  (``Critical Issues`` or ``Recommendations``).
"""
from __future__ import annotations

import re

import pytest

from agents import agents
from tests.agent_evals._helpers import _basic_config, install_fake_agent, invoke


_READY_REPORT = (
    "## Verdict\n"
    "**Decision:** ✅ Ready to merge\n"
    "\n"
    "### Critical Issues (Blockers)\n"
    "(none)\n"
    "\n"
    "### Recommendations (Non-Blockers)\n"
    "1. Consider lifting the `submitButton` locator to a top-level "
    "constant.\n"
)

_FIXES_REPORT = (
    "## Verdict\n"
    "**Decision:** ❌ Fixes required\n"
    "\n"
    "### Critical Issues (Blockers)\n"
    "1. `line 14` — locator uses `locator('.btn')`; the codex requires "
    "`getByTestId` for the submit button.\n"
    "2. `line 22` — `toHaveText('hello world')` is hardcoded; replace "
    "with a regex matcher.\n"
    "\n"
    "### Recommendations (Non-Blockers)\n"
    "1. Wrap the test in `test.describe` with the canonical name.\n"
)

_NOTES_REPORT = (
    "## Verdict\n"
    "**Decision:** ⚠️ Has notes\n"
    "\n"
    "### Critical Issues (Blockers)\n"
    "(none)\n"
    "\n"
    "### Recommendations (Non-Blockers)\n"
    "1. Step 3 of the test case is ambiguous — clarify with the case "
    "author whether the modal must be visible before the click.\n"
)


def _assert_verdict_contract(result: str) -> None:
    """Shared structural assertions for every review verdict."""
    assert "<think>" not in result
    # Mandatory structural elements.
    assert re.search(r"##\s+verdict", result, flags=re.IGNORECASE), (
        f"missing '## Verdict' header in: {result!r}"
    )
    assert "**Decision:**" in result
    assert re.search(r"(✅|❌|⚠️)", result), "no verdict emoji found"
    # At least one of the two section headers is required.
    assert re.search(
        r"(critical issues|recommendations)", result, flags=re.IGNORECASE
    ), "neither 'Critical Issues' nor 'Recommendations' header found"


# ---------------------------------------------------------------------------
# Ready to merge
# ---------------------------------------------------------------------------
def test_reviewer_ready_verdict_meets_contract(monkeypatch: pytest.MonkeyPatch):
    install_fake_agent(monkeypatch, _READY_REPORT)

    result = invoke(
        agents.reviewer_agent.ainvoke(
            {
                "query": "case: /app/data/inputs/case_2123.md | "
                "test: /app/data/outputs/tests/2123.spec.ts",
                "config": _basic_config(),
            }
        )
    )

    _assert_verdict_contract(result)
    assert "✅" in result


# ---------------------------------------------------------------------------
# Fixes required
# ---------------------------------------------------------------------------
def test_reviewer_blockers_meets_contract(monkeypatch: pytest.MonkeyPatch):
    install_fake_agent(monkeypatch, _FIXES_REPORT)

    result = invoke(
        agents.reviewer_agent.ainvoke(
            {
                "query": "case: /app/data/inputs/case_2123.md | "
                "test: /app/data/outputs/tests/2123.spec.ts",
                "config": _basic_config(),
            }
        )
    )

    _assert_verdict_contract(result)
    assert "❌" in result
    # Blocker reports must include the section header.
    assert re.search(r"critical issues", result, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Notes only
# ---------------------------------------------------------------------------
def test_reviewer_notes_meets_contract(monkeypatch: pytest.MonkeyPatch):
    install_fake_agent(monkeypatch, _NOTES_REPORT)

    result = invoke(
        agents.reviewer_agent.ainvoke(
            {
                "query": "case: /app/data/inputs/case_2123.md | "
                "test: /app/data/outputs/tests/2123.spec.ts",
                "config": _basic_config(),
            }
        )
    )

    _assert_verdict_contract(result)
    assert "⚠️" in result


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------
def test_reviewer_exception_is_returned_as_error(monkeypatch: pytest.MonkeyPatch):
    from agents_core import agent_creation

    async def _raise(*_a, **_kw):
        raise RuntimeError("reviewer LLM rejected the request")

    monkeypatch.setattr(agent_creation, "agent", _raise)
    monkeypatch.setattr(agents, "agent", _raise)

    result = invoke(
        agents.reviewer_agent.ainvoke(
            {"query": "review the .ts", "config": _basic_config()}
        )
    )

    assert result.startswith("Error: ")
    assert "reviewer LLM rejected" in result

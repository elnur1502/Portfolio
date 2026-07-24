"""
Unit tests for ``agents.agents.test_explorer_agent``.

The explorer is the executor of the team — it reads a test case,
writes an auto test, runs it, and returns a verdict. We mock the
LLM layer and assert the agent's *response contract*:

* the result block is present and well-formed;
* the file path is announced and lives under
  ``/app/data/outputs/tests/``;
* the response is free of leaked ``<think>`` tags;
* an exception in the LLM call is surfaced as ``Error: ...``
  instead of being raised.
"""
from __future__ import annotations

import pytest

from agents import agents
from tests.agent_evals._helpers import _basic_config, install_fake_agent, invoke


_GREEN_RESPONSE = (
    "**Result:** ✅ Done, run is green\n"
    "**File:** `/app/data/outputs/tests/2123_uz_mainpage.spec.ts`\n"
    "**Run:** passed 3 / 3, duration 1240ms\n"
    "**Brief:** validates the virtual keyboard toggles on click and the "
    "submitted value appears in the input.\n"
)

_RED_RESPONSE = (
    "**Result:** ❌ Not done, run is red after 2 rounds\n"
    "**File:** `/app/data/outputs/tests/2123_uz_mainpage.spec.ts`\n"
    "**What failed:** step 4 — expect button to be enabled after click\n"
    "**Suspicion:** case vs. product mismatch — the button is enabled "
    "only after a network response that the case does not await.\n"
)


# ---------------------------------------------------------------------------
# Green run
# ---------------------------------------------------------------------------
def test_explorer_green_run_meets_contract(monkeypatch: pytest.MonkeyPatch):
    """A green run must produce the structured result the supervisor parses."""
    install_fake_agent(monkeypatch, _GREEN_RESPONSE)

    result = invoke(
        agents.test_explorer_agent.ainvoke(
            {
                "query": "case 2123 — UZ desktop main page virtual keyboard",
                "config": _basic_config(),
            }
        )
    )

    # The agent strips <think> blocks and returns a clean string.
    assert "<think>" not in result
    assert "</think>" not in result
    # The structured result line is present.
    assert "**Result:**" in result
    assert "✅" in result
    # The file is announced and lives under outputs/tests/.
    assert "**File:**" in result
    assert "/app/data/outputs/tests/" in result


# ---------------------------------------------------------------------------
# Red run
# ---------------------------------------------------------------------------
def test_explorer_red_run_still_meets_contract(monkeypatch: pytest.MonkeyPatch):
    """A red run is *still* a valid response — the supervisor has to read it."""
    install_fake_agent(monkeypatch, _RED_RESPONSE)

    result = invoke(
        agents.test_explorer_agent.ainvoke(
            {
                "query": "case 2123 — UZ desktop main page virtual keyboard",
                "config": _basic_config(),
            }
        )
    )

    assert "<think>" not in result
    assert "**Result:**" in result
    assert "❌" in result
    assert "/app/data/outputs/tests/" in result


# ---------------------------------------------------------------------------
# Error path — LLM blows up
# ---------------------------------------------------------------------------
def test_explorer_exception_is_returned_as_error(monkeypatch: pytest.MonkeyPatch):
    """If the LLM call raises, the agent returns ``Error: <msg>`` — not propagates."""
    from agents_core import agent_creation

    async def _raise(*_a, **_kw):
        raise RuntimeError("upstream LLM exploded")

    monkeypatch.setattr(agent_creation, "agent", _raise)
    monkeypatch.setattr(agents, "agent", _raise)

    result = invoke(
        agents.test_explorer_agent.ainvoke(
            {"query": "case 2123", "config": _basic_config()}
        )
    )

    assert result.startswith("Error: "), (
        f"expected an 'Error: ' prefix, got: {result!r}"
    )
    assert "upstream LLM exploded" in result


# ---------------------------------------------------------------------------
# Threading — the explorer derives an ``_explorer``-suffixed thread id
# from the input config and forwards it to the agent factory.
# ---------------------------------------------------------------------------
def test_explorer_forwards_its_derived_config_to_the_agent(
    monkeypatch: pytest.MonkeyPatch,
):
    """The explorer's factory call is wired up with a non-empty config."""
    stub = install_fake_agent(monkeypatch, _GREEN_RESPONSE)

    invoke(
        agents.test_explorer_agent.ainvoke(
            {
                "query": "case 2123",
                "config": _basic_config("session-xyz"),
            }
        )
    )

    # The factory must have been called at least once.
    assert stub.calls, "the agent factory was never invoked"
    factory_call = stub.calls[0]
    # The state handed to the graph is always the canonical
    # ``{"messages": [(user, query)]}`` shape.
    assert factory_call["state"] == {"messages": [("user", "case 2123")]}
    # The config the explorer forwards is a dict with the expected
    # keys; the exact thread_id is the explorer's own concern.
    config = factory_call["config"]
    assert isinstance(config, dict)
    assert "configurable" in config
    assert "thread_id" in config["configurable"]

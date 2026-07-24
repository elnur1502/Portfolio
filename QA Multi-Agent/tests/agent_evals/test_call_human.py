"""
Unit tests for ``agents.human_call.call_human``.

The supervisor's only channel to the user mid-flow. Unlike the
LLM-driven agents, ``call_human`` is a pure tool: it blocks on
``langgraph.types.interrupt`` and returns ``"User responded: <raw>"``.

We mock ``interrupt`` and assert the contract:

* the user sees the question, with optional context on a new line;
* the return value wraps the user's reply verbatim — no trimming,
  no mutating;
* the contract is shape-preserving across invocations.
"""
from __future__ import annotations

import pytest

from agents.human_call import call_human
from tests.agent_evals._helpers import invoke


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _StubInterrupt:
    def __init__(self):
        self.calls: list[str] = []
        self.next_response: object = "yes, proceed"

    def __call__(self, prompt: str):
        self.calls.append(prompt)
        return self.next_response


@pytest.fixture
def stub_interrupt(monkeypatch: pytest.MonkeyPatch) -> _StubInterrupt:
    stub = _StubInterrupt()
    monkeypatch.setattr("agents.human_call.interrupt", stub)
    return stub


# ---------------------------------------------------------------------------
# Single round-trip
# ---------------------------------------------------------------------------
def test_call_human_round_trip(stub_interrupt: _StubInterrupt):
    result = invoke(
        call_human.ainvoke(
            {"query": "Proceed with the test run?", "context": None}
        )
    )

    assert result == "User responded: yes, proceed"
    # The user saw only the question, no extra content.
    assert stub_interrupt.calls == ["Proceed with the test run?"]


# ---------------------------------------------------------------------------
# Context is appended after a newline
# ---------------------------------------------------------------------------
def test_call_human_appends_context(stub_interrupt: _StubInterrupt):
    invoke(
        call_human.ainvoke(
            {
                "query": "Which URL should I use?",
                "context": "I tried https://a.example and https://b.example, both 404.",
            }
        )
    )

    sent = stub_interrupt.calls[0]
    # A newline separates the question from the context.
    head, _, tail = sent.partition("\n")
    assert head == "Which URL should I use?"
    assert "https://a.example" in tail
    assert "https://b.example" in tail


# ---------------------------------------------------------------------------
# Dict / list context is stringified
# ---------------------------------------------------------------------------
def test_call_human_stringifies_dict_context(stub_interrupt: _StubInterrupt):
    invoke(
        call_human.ainvoke(
            {
                "query": "Pick one",
                "context": {"options": ["A", "B"]},
            }
        )
    )

    sent = stub_interrupt.calls[0]
    # The dict is rendered via ``str()``; both options and the key
    # "options" must be visible.
    assert "options" in sent
    assert "'A'" in sent and "'B'" in sent


# ---------------------------------------------------------------------------
# Idempotency — calling twice with the same args yields the same shape
# ---------------------------------------------------------------------------
def test_call_human_idempotent(stub_interrupt: _StubInterrupt):
    """The contract is shape-preserving across invocations."""
    stub_interrupt.next_response = "answer-A"
    a = invoke(call_human.ainvoke({"query": "Same question?"}))

    stub_interrupt.next_response = "answer-B"
    b = invoke(call_human.ainvoke({"query": "Same question?"}))

    # Both return the wrapper prefix...
    assert a.startswith("User responded: ")
    assert b.startswith("User responded: ")
    # ...and differ *only* in the echoed reply.
    assert a == "User responded: answer-A"
    assert b == "User responded: answer-B"


# ---------------------------------------------------------------------------
# Empty / falsy response
# ---------------------------------------------------------------------------
def test_call_human_treats_empty_response_as_valid(stub_interrupt: _StubInterrupt):
    """An empty / falsy user reply still returns the contract string."""
    stub_interrupt.next_response = ""

    result = invoke(call_human.ainvoke({"query": "Anything?"}))

    assert result == "User responded: "

"""
Unit tests for the ``call_human`` tool.

It is the only way the supervisor reaches the user mid-flow, so we
verify the *contract* end-to-end:

* an empty ``context`` shows just the question to the user;
* a non-empty ``context`` is appended after a newline;
* the return value is always ``"User responded: <raw>"`` with no
  mutation of the user's text.
"""
from __future__ import annotations

import pytest

from agents.human_call import call_human


# ---------------------------------------------------------------------------
# Patch the underlying LangGraph ``interrupt`` so we can drive the call
# without a real graph execution.
# ---------------------------------------------------------------------------
@pytest.fixture
def patched_interrupt(monkeypatch: pytest.MonkeyPatch):
    """Replace ``langgraph.types.interrupt`` with a queue-driven stub.

    The function imported as ``agents.human_call.interrupt`` is the
    same object as ``langgraph.types.interrupt``; patching the local
    reference in the ``human_call`` module is enough.
    """

    class _Stub:
        def __init__(self):
            self.calls: list[str] = []
            self.next_response: object = "stub answer"

        def __call__(self, prompt: str):
            self.calls.append(prompt)
            return self.next_response

    stub = _Stub()
    monkeypatch.setattr("agents.human_call.interrupt", stub)
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_call_human_returns_user_response_verbatim(patched_interrupt):
    patched_interrupt.next_response = "yes, do it"

    result = await call_human.ainvoke({"query": "Proceed?"})

    assert result == "User responded: yes, do it"


@pytest.mark.asyncio
async def test_call_human_passes_only_query_when_context_is_none(patched_interrupt):
    """With no context the user must see the raw question."""
    await call_human.ainvoke({"query": "What URL?"})

    assert patched_interrupt.calls == ["What URL?"]


@pytest.mark.asyncio
async def test_call_human_appends_context_after_newline(patched_interrupt):
    """A non-None context is glued to the query with a newline separator."""
    await call_human.ainvoke(
        {"query": "What URL?", "context": "We tried https://a and https://b, both 404."}
    )

    sent = patched_interrupt.calls[0]
    assert sent.startswith("What URL?\n")
    assert "404" in sent


@pytest.mark.asyncio
async def test_call_human_stringifies_dict_context(patched_interrupt):
    """Dict / list context is glued via ``str(...)`` — the user still sees it."""
    await call_human.ainvoke(
        {"query": "Pick one", "context": {"options": ["A", "B"]}}
    )

    sent = patched_interrupt.calls[0]
    assert "options" in sent and "A" in sent and "B" in sent


@pytest.mark.asyncio
async def test_call_human_treats_empty_response_as_valid(patched_interrupt):
    """An empty / falsy user reply still returns the contract string."""
    patched_interrupt.next_response = ""

    result = await call_human.ainvoke({"query": "Anything?"})

    assert result == "User responded: "

"""
Unit tests for ``agents.agents.writer_agent``.

The writer is the team's documenter — it turns QA materials into
readable text artifacts (READMEs, test plans, review reports,
recaps, summaries). We assert two contracts:

* the output is markdown (heading or list present);
* the writer does *not* slip into emitting Playwright test code —
  that's the explorer's job, and a regression there would silently
  bypass self-verification;
* LLM exceptions are surfaced as ``Error: ...`` rather than raised.
"""
from __future__ import annotations

import re

import pytest

from agents import agents
from tests.agent_evals._helpers import _basic_config, install_fake_agent, invoke


_README_RESPONSE = (
    "# QA Suite — Virtual Keyboard\n"
    "\n"
    "## Scope\n"
    "Covers the manual test cases for the on-screen virtual keyboard "
    "in the UZ desktop build.\n"
    "\n"
    "## Auto tests\n"
    "- `2123_uz_mainpage.spec.ts` — toggles the keyboard and asserts "
    "the submitted value.\n"
    "- `2124_uz_mainpage.spec.ts` — dismisses the keyboard on Esc.\n"
)

_RECAP_RESPONSE = (
    "## Recap\n"
    "We generated 2 auto tests for the virtual keyboard flow. Both "
    "runs are green. The reviewer approved both without notes.\n"
    "\n"
    "**Saved to:** `/app/data/outputs/docs/recap.md`\n"
)


# ---------------------------------------------------------------------------
# Happy path — README artifact
# ---------------------------------------------------------------------------
def test_writer_readme_is_markdown(monkeypatch: pytest.MonkeyPatch):
    install_fake_agent(monkeypatch, _README_RESPONSE)

    result = invoke(
        agents.writer_agent.ainvoke(
            {
                "query": "Write a README for the virtual keyboard suite, "
                "concise, audience: devs.",
                "config": _basic_config(),
            }
        )
    )

    # Markdown signal: at least one heading.
    assert re.search(r"(^|\n)#{1,3}\s+\S", result), (
        f"writer output doesn't look like markdown: {result!r}"
    )
    # Sanity: no leaked <think> blocks.
    assert "<think>" not in result
    # The writer must not emit Playwright test code — that's the
    # explorer's job, and a regression here would bypass
    # self-verification.
    assert "```typescript" not in result
    assert "```ts\n" not in result
    assert "playwright/test" not in result


# ---------------------------------------------------------------------------
# Saved-to-path artifact
# ---------------------------------------------------------------------------
def test_writer_recap_marks_its_output(monkeypatch: pytest.MonkeyPatch):
    install_fake_agent(monkeypatch, _RECAP_RESPONSE)

    result = invoke(
        agents.writer_agent.ainvoke(
            {
                "query": "Recap the latest test run, 80 words, friendly tone.",
                "config": _basic_config(),
            }
        )
    )

    # The contract is more permissive here — a recap may or may not
    # be saved to a file. We just assert it's markdown and clean.
    assert re.search(r"(^|\n)#{1,3}\s+\S", result)
    assert "<think>" not in result
    assert "playwright/test" not in result


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------
def test_writer_exception_is_returned_as_error(monkeypatch: pytest.MonkeyPatch):
    from agents_core import agent_creation

    async def _raise(*_a, **_kw):
        raise RuntimeError("writer LLM is down")

    monkeypatch.setattr(agent_creation, "agent", _raise)
    monkeypatch.setattr(agents, "agent", _raise)

    result = invoke(
        agents.writer_agent.ainvoke(
            {"query": "Write a test plan.", "config": _basic_config()}
        )
    )

    assert result.startswith("Error: ")
    assert "writer LLM is down" in result

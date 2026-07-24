"""
Helpers shared by the team-agent tests.

The real agents are wired to an LLM and a live Playwright MCP
session. In tests we don't want either. Instead we:

* patch the ``agent`` factory that every team agent uses
  (``agents.agents.agent`` is the bound name after
  ``from agents_core.agent_creation import agent``) so the returned
  graph's ``ainvoke`` returns a deterministic ``AIMessage``;
* assert on the structural contract of the agent's response with
  plain ``assert`` statements.

No deepeval, no LLM judge, no API keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage


@dataclass
class StubGraph:
    """A drop-in replacement for the LangGraph compiled graph.

    ``ainvoke`` returns a state dict whose last message is
    ``fake_response``. The factory records every invocation so tests
    can later assert on the call shape.
    """

    fake_response: str
    extra: dict[str, Any] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)

    async def ainvoke(self, state: dict, config: Any = None) -> dict:
        self.calls.append({"state": state, "config": config})
        return {
            "messages": [AIMessage(content=self.fake_response)],
            **self.extra,
        }

    def get_graph(self):
        return self

    @property
    def checkpointer(self):
        return None


def install_fake_agent(monkeypatch, fake_response: str) -> StubGraph:
    """Patch the ``agent`` factory for every team-agent module.

    The team-agent functions do
    ``from agents_core.agent_creation import agent`` and then call
    that bound name. Patching the name *as used* in each consumer
    module is what actually short-circuits the LLM call.

    The supervisor itself doesn't import this factory — it builds its
    own graph from the four tool functions — so we skip it here.
    """
    from agents_core import agent_creation
    from agents import agents as team_agents

    stub = StubGraph(fake_response=fake_response)

    async def _fake(model, tools, sys_prompt, config, query, memory_flag=False):
        return stub

    for module in (agent_creation, team_agents):
        monkeypatch.setattr(module, "agent", _fake)

    return stub


def _basic_config(thread_id: str = "test-thread") -> dict:
    """Build a ``RunnableConfig``-shaped dict for tests."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,
    }


def invoke(coro):
    """Run an awaitable to completion in a fresh event loop.

    We use this instead of ``asyncio.run`` at the module top level
    because ``asyncio.run`` cannot be called from inside a running
    loop (which pytest-asyncio itself is). Using a tiny helper keeps
    each test self-contained.
    """
    import asyncio

    return asyncio.run(coro)

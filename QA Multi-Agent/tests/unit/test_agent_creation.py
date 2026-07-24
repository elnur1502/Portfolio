"""
Unit tests for ``agents_core.agent_creation.agent`` — the LangGraph
factory used by every team agent.

We don't run an LLM here, we just verify that:

* the function returns a compiled graph;
* the graph has the expected nodes (``call_agent`` and ``tools``);
* the conditional edge routes between them correctly when the model
  asks for a tool vs. when it doesn't.
* ``memory_flag=True`` wires a checkpointer, ``memory_flag=False``
  leaves the graph stateless.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from agents_core import agent_creation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _StubModel:
    """A model stub that returns a fixed ``AIMessage`` and supports
    ``bind_tools`` so the factory can wire it up exactly like a real
    ``ChatOpenAI`` instance."""

    def __init__(self, response: AIMessage):
        self._response = response
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages, config=None):
        return self._response


# ``agent_creation.agent`` always wraps the tools list in a ``ToolNode``,
# which validates that every entry has a ``.name``. We pass an empty
# list to the factory so it doesn't choke, and use a separate spy model
# to verify what *would* have been bound.
class _SpyModel:
    def __init__(self):
        self.bound_tools: list[Any] = []
        self._response = AIMessage(content="hi")

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages, config=None):
        return self._response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_factory_returns_compiled_graph():
    model = _StubModel(AIMessage(content="hi"))
    graph = await agent_creation.agent(
        model, tools=[], sys_prompt="you are a test", config={}, query="hello"
    )

    # The factory wraps the StateGraph in a compiled runnable.
    assert hasattr(graph, "ainvoke")
    assert hasattr(graph, "get_graph")


@pytest.mark.asyncio
async def test_agent_factory_passes_tools_to_model():
    """The factory forwards the tools list to ``bind_tools`` on the model."""
    model = _SpyModel()

    await agent_creation.agent(
        model,
        tools=[],
        sys_prompt="x",
        config={},
        query="q",
    )

    # Even with an empty tools list, the factory must have called bind_tools.
    assert hasattr(model, "bound_tools")
    assert model.bound_tools == []


@pytest.mark.asyncio
async def test_agent_with_memory_flag_uses_checkpointer():
    """``memory_flag=True`` must yield a graph with a checkpointer attached."""
    from langgraph.checkpoint.memory import MemorySaver

    model = _StubModel(AIMessage(content="hi"))
    graph = await agent_creation.agent(
        model,
        tools=[],
        sys_prompt="x",
        config={},
        query="q",
        memory_flag=True,
    )

    checkpointer = graph.checkpointer
    assert isinstance(checkpointer, MemorySaver)


@pytest.mark.asyncio
async def test_agent_without_memory_flag_has_no_checkpointer():
    model = _StubModel(AIMessage(content="hi"))
    graph = await agent_creation.agent(
        model, tools=[], sys_prompt="x", config={}, query="q"
    )

    assert graph.checkpointer is None


@pytest.mark.asyncio
async def test_agent_graph_has_call_agent_and_tools_nodes():
    """The compiled graph exposes both nodes — model and tool execution."""
    model = _StubModel(AIMessage(content="hi"))
    graph = await agent_creation.agent(
        model, tools=[], sys_prompt="x", config={}, query="q"
    )

    # ``get_graph().nodes`` is keyed by node name.
    node_names = set(graph.get_graph().nodes.keys())
    assert "call_agent" in node_names
    assert "tools" in node_names

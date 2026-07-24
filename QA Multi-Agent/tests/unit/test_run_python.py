"""
Unit tests for ``agents_core.tools.run_python``.

The tool has three independent surfaces to validate:

* happy path — captures ``sys.stdout`` and returns it;
* AST-level safety net — rejects forbidden imports, ``exec``,
  ``eval`` *before* the code is executed;
* error path — execution exceptions and malformed inputs are
  surfaced as structured ``dict`` payloads, not raised.
"""
from __future__ import annotations

import pytest

from agents_core import tools


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_python_returns_stdout_capture():
    """A trivial print should round-trip through the sandbox."""
    result = await tools.run_python.ainvoke({"code": "print('hello world')"})

    assert result["status"] == "success"
    assert "hello world" in result["output"]


@pytest.mark.asyncio
async def test_run_python_persists_state_between_calls():
    """The tool is intentionally stateful — second call sees first call's vars."""
    await tools.run_python.ainvoke({"code": "x = 41"})
    result = await tools.run_python.ainvoke({"code": "print(x + 1)"})

    assert result["status"] == "success"
    assert "42" in result["output"]


@pytest.mark.asyncio
async def test_run_python_strips_markdown_fences():
    """Code may be wrapped in ```python ... ```; both should be accepted."""
    fenced = "```python\nprint('fenced')\n```"
    result = await tools.run_python.ainvoke({"code": fenced})

    assert result["status"] == "success"
    assert "fenced" in result["output"]


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_python_returns_exception_payload():
    """A NameError must be returned as a structured dict, not raised."""
    result = await tools.run_python.ainvoke({"code": "print(undefined_name)"})

    assert result["status"] == "error exception"
    assert "NameError" in result["output"]


@pytest.mark.asyncio
async def test_run_python_handles_syntax_error_as_ast_failure():
    """A syntax error fails at the ``ast.parse`` step."""
    result = await tools.run_python.ainvoke({"code": "def broken(:\n  pass"})

    assert result["status"] == "error AST parsing"


# ---------------------------------------------------------------------------
# Safety net — forbidden modules / calls
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snippet",
    [
        "import subprocess",
        "from socket import socket",
        "import requests",
        "import webbrowser",
        "import pickle",
        "import sys",
    ],
)
async def test_run_python_blocks_forbidden_imports(snippet: str):
    """Every import of a blacklisted module must be refused up front.

    Note: ``os`` is intentionally *not* in the blacklist — the agent
    needs it for path joining inside the sandbox. Only modules that
    would let the agent escape the sandbox are blocked.
    """
    result = await tools.run_python.ainvoke({"code": snippet})

    assert "not allowed" in str(result).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("snippet", ["exec('print(1)')", "eval('1+1')"])
async def test_run_python_blocks_exec_and_eval(snippet: str):
    """``exec`` and ``eval`` are never allowed, even with no import."""
    result = await tools.run_python.ainvoke({"code": snippet})

    assert "not allowed" in str(result).lower()


@pytest.mark.asyncio
async def test_run_python_allows_safe_modules():
    """Math, json, datetime etc. must keep working after the safety net."""
    result = await tools.run_python.ainvoke(
        {"code": "import math\nimport json\nprint(math.sqrt(16))"}
    )

    assert result["status"] == "success"
    assert "4.0" in result["output"]


# ---------------------------------------------------------------------------
# reset() — short-term memory wipe
# ---------------------------------------------------------------------------
def test_reset_clears_short_memory():
    """After ``reset()`` previously assigned names are gone."""
    tools.SHORT_MEMORY["y"] = 99
    tools.reset()
    assert "y" not in tools.SHORT_MEMORY

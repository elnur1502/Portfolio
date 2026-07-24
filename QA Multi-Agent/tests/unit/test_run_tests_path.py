"""
Unit tests for ``agents_core.tools.run_tests`` *path normalisation*.

The tool spawns ``npx playwright test`` in a subprocess. We don't
want to actually launch a browser in CI, so we only validate the
input contract: the function must reject anything that doesn't look
like a path under ``/app/data/outputs/tests/`` and must normalise
the leading ``/app/`` prefix correctly.
"""
from __future__ import annotations

import pytest

from agents_core import tools


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_tests_rejects_path_outside_tests_dir():
    """Any path that is not under ``tests/`` is refused with a clear error."""
    result = await tools.run_tests.ainvoke({"test_path": "/app/data/outputs/docs/readme.md"})

    assert "Error" in result
    assert "tests/" in result


@pytest.mark.asyncio
async def test_run_tests_rejects_absolute_system_path():
    """System paths that escape ``data/`` are refused too."""
    result = await tools.run_tests.ainvoke({"test_path": "/tmp/foo.spec.ts"})

    # Stripped path doesn't start with ``tests/`` → rejected.
    assert "Error" in result


# ---------------------------------------------------------------------------
# Normalisation paths — we patch the subprocess call so no real test runs.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_tests_normalises_app_prefix(monkeypatch: pytest.MonkeyPatch):
    """The leading ``/app/`` must be stripped before resolving the cwd."""
    captured: dict = {}

    class _FakeProc:
        returncode = 0
        def communicate(self):  # noqa: D401 — async awaitable expected
            return (b"", b"")

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProc()

    # asyncio.create_subprocess_exec is a free function, not on the
    # tools module, so we patch the import the function looks up.
    import asyncio
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    await tools.run_tests.ainvoke(
        {"test_path": "/app/data/outputs/tests/spec.spec.ts"}
    )

    cmd = captured["args"]
    # The ``tests/...`` part must be present, but the leading ``data/`` / ``app/`` must NOT.
    joined = " ".join(str(p) for p in cmd)
    assert "tests/spec.spec.ts" in joined
    assert "/app/" not in joined


@pytest.mark.asyncio
async def test_run_tests_handles_no_path(monkeypatch: pytest.MonkeyPatch):
    """Calling with no path runs the whole suite — no ``tests/`` flag is appended."""
    captured: dict = {}

    class _FakeProc:
        returncode = 0
        async def communicate(self):
            return (b"", b"")

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        return _FakeProc()

    import asyncio
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    await tools.run_tests.ainvoke({"test_path": None})

    cmd = [str(p) for p in captured["args"]]
    # The default invocation is just ``npx playwright test`` —
    assert "playwright" in cmd
    assert "test" in cmd
    assert "tests/" not in cmd

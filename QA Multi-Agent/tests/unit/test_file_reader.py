"""
Unit tests for ``agents_core.tools.file_reader``.

The tool is the only way an agent reads a file, so path-safety
guarantees matter:

* the path must live under ``/app/data/``;
* directories must be rejected with a clear message;
* missing files must be reported, not raised.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agents_core import tools


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_file_reader_rejects_path_outside_data_dir():
    """Anything that escapes ``data/`` is refused outright."""
    result = await tools.file_reader.ainvoke({"path": "/etc/passwd"})

    assert "must be under" in result


@pytest.mark.asyncio
async def test_file_reader_rejects_relative_path_above_data():
    """Traversals out of the data dir are stripped and rejected."""
    result = await tools.file_reader.ainvoke({"path": "/app/../etc/passwd"})

    # Stripped path no longer starts with ``data/`` → refused.
    assert "must be under" in result or "Error" in result


# ---------------------------------------------------------------------------
# Directory vs file
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_file_reader_rejects_directory(data_sandbox: Path):
    """Passing a directory must be reported, not opened as a file."""
    # data_sandbox already contains inputs/ + outputs/ subdirs.
    result = await tools.file_reader.ainvoke({"path": "/app/data/inputs"})

    assert "directory" in result.lower()
    assert "exactly one file" in result.lower() or "single file" in result.lower()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_file_reader_returns_plain_text(data_sandbox: Path):
    """A real .txt file under data/ is converted to text."""
    target = data_sandbox / "inputs" / "case.md"
    target.write_text("# Title\n\nBody line.\n", encoding="utf-8")

    result = await tools.file_reader.ainvoke({"path": f"/app/data/inputs/{target.name}"})

    assert "Title" in result
    assert "Body line." in result
    assert "Error" not in result


@pytest.mark.asyncio
async def test_file_reader_handles_typescript(data_sandbox: Path):
    """``.ts`` files are read as text — markdown conversion is still safe."""
    target = data_sandbox / "inputs" / "spec.ts"
    target.write_text(
        "import { test, expect } from '@playwright/test';\n"
        "test('demo', async () => { expect(1).toBe(1); });\n",
        encoding="utf-8",
    )

    result = await tools.file_reader.ainvoke({"path": "/app/data/inputs/spec.ts"})

    assert "expect(1).toBe(1)" in result
    assert "Error" not in result


# ---------------------------------------------------------------------------
# Missing files
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_file_reader_reports_missing_file(data_sandbox: Path):
    result = await tools.file_reader.ainvoke(
        {"path": "/app/data/inputs/does_not_exist.md"}
    )

    assert "not found" in result.lower() or "Error" in result

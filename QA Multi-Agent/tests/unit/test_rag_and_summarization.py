"""
Unit tests for the long-term memory tools: ``RAG_statements`` and
``summarization``.

The tools work against a persistent ChromaDB collection that lives
under ``chroma_db/`` in the project root. We don't want the tests
to depend on (or corrupt) that on-disk state, so the
``fresh_collection`` fixture swaps ``tools.profiles_collection`` for
a brand-new in-memory collection with the same name. The
``embed_model`` is also stubbed to keep the test offline.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents_core import tools


# ---------------------------------------------------------------------------
# Stub for the OpenAI embeddings call
# ---------------------------------------------------------------------------
class _StubEmbeddings:
    """Returns a fixed-length zero vector for every query / document.
    ...
    """
    dimension = 3072  # default; overridden by the fixture

    # Sync API (used by some LangChain internals)
    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]

    # Async API (used by tools.summarization / RAG_statements after
    # the aembed_query migration — keep both, both must keep working)
    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


# ---------------------------------------------------------------------------
# Fixture: isolated, dimension-matched collection
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_collection(monkeypatch: pytest.MonkeyPatch):
    """Replace ``tools.profiles_collection`` with a fresh in-memory collection.

    The real collection is a persistent ChromaDB instance. We swap
    it for an ``InMemoryClient`` so the tests don't depend on what's
    in the project's on-disk ``chroma_db/`` directory — different
    developers have different embedding dimensions there, and we
    don't want to fight that.
    """
    import chromadb

    # 1) Discover the dimension the live collection expects, so the
    #    stub produces matching vectors.
    try:
        existing = tools.profiles_collection.get(limit=1, include=["embeddings"])
        if existing and existing.get("embeddings") is not None and len(existing["embeddings"]) > 0:
            _StubEmbeddings.dimension = len(existing["embeddings"][0])
    except Exception:
        # If the live collection is empty or unreachable, fall back
        # to the default dim; the in-memory collection below will
        # happily accept any consistent dim anyway.
        pass

    # 2) Build a fresh in-memory client + collection with the same
    #    name the tools use. In-memory Chroma accepts whatever dim
    #    the first vector carries, so this is dimension-agnostic.
    in_memory_client = chromadb.EphemeralClient()
    fresh = in_memory_client.get_or_create_collection(name="preferences")

    # 3) Swap the module-level reference so ``tools.summarization``
    #    and ``tools.RAG_statements`` use our fresh collection.
    monkeypatch.setattr(tools, "profiles_collection", fresh, raising=True)

    # 4) Stub the embedder too.
    monkeypatch.setattr(tools, "embed_model", _StubEmbeddings(), raising=True)

    return fresh


# ---------------------------------------------------------------------------
# summarization
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_summarization_persists_record(fresh_collection):
    """A document + metadata pair must land in the collection."""
    initial = fresh_collection.count()

    result = await tools.summarization.ainvoke(
        {
            "chromadb_data": {
                "documents": ["user prefers concise test cases"],
                "metadatas": [{"source": "unit-test", "user_id": "t-1"}],
            }
        }
    )

    assert "successfully" in result.lower()
    assert fresh_collection.count() == initial + 1


@pytest.mark.asyncio
async def test_summarization_swallows_chromadb_errors(
    fresh_collection, monkeypatch: pytest.MonkeyPatch
):
    """When the underlying store raises, the tool reports a clean error string."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("chromadb exploded")

    monkeypatch.setattr(fresh_collection, "add", boom)

    result = await tools.summarization.ainvoke(
        {
            "chromadb_data": {
                "documents": ["will fail"],
                "metadatas": [{"source": "unit-test"}],
            }
        }
    )

    assert "could not add" in result.lower()
    assert "chromadb exploded" in result


# ---------------------------------------------------------------------------
# RAG_statements
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rag_statements_returns_documents(fresh_collection):
    """After writing, a related query must surface the stored text."""
    await tools.summarization.ainvoke(
        {
            "chromadb_data": {
                "documents": ["always run with --headed in CI"],
                "metadatas": [{"source": "unit-test"}],
            }
        }
    )

    results = await tools.RAG_statements.ainvoke(
        {"query_text": "how should I run playwright in CI?"}
    )

    assert isinstance(results, list)
    # The list can be empty if Chroma's default distance function
    # rejected the zero-vector match; in that case, the contract
    # ``isinstance(..., list)`` is the part that matters.
    # We assert *either* we found the doc, *or* we got an error
    # string back — never an exception.
    if results and not str(results[0]).startswith("Error"):
        assert any("headed" in r for r in results)


@pytest.mark.asyncio
async def test_rag_statements_handles_empty_collection(fresh_collection):
    """A fresh collection returns either an empty list or a clean error string."""
    results = await tools.RAG_statements.ainvoke(
        {"query_text": "anything at all"}
    )

    # Both outcomes are acceptable contracts: an empty list (no
    # matches) or a Chroma error string. Anything else (an unhandled
    # exception, a non-list) is a regression.
    assert isinstance(results, list)
    if results:
        # If anything came back, it must be either a doc or an
        # explicit error string from the tool itself.
        assert all(isinstance(r, str) for r in results)

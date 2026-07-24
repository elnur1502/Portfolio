# `chroma_db/` — ChromaDB persistent store

This directory holds the on-disk ChromaDB index used by the agent's
retrieval-augmented preferences selection (see `agents_core/tools.py` →
`RAG_statements`).

## Collections

The agent opens collection at import time:

| Collection | Status | Purpose |
| --- | --- | --- |
| `preferences` | **used** | Statements and preferences that have been added by Supervisor agent |

# `chroma_db/` — ChromaDB persistent store

This directory holds the on-disk ChromaDB index used by the agent's
retrieval-augmented table selection (see `agents/agent_nodes.py` →
`table_rerank`).

## Collections

The agent opens three collections at import time:

| Collection | Status | Purpose |
| --- | --- | --- |
| `tables` | **used** | SQL table metadata (name, description, columns, usage_count). Source for `table_rerank`. |

## Indexing

The script that builds the `tables` collection from your warehouse is
**not** included in this repo (it lives outside the public version). A
typical pipeline looks like:

1. Pull `table_name`, `comment`, and column lists from
   `INFORMATION_SCHEMA.COLUMNS` (or your warehouse equivalent).
2. Build a short description per table — either with an LLM or a template
   like `"<table_name>: <comment>. Columns: <col1, col2, ...>"`.
3. Embed `(description + table_name + columns)` with
   `text-embedding-3-large` (configured in `agents/agent_models.py`).
4. `upsert` into the `tables` collection with `usage_count` in the metadata
   so the keyword score (`keyword_score` in `agent_nodes.py`) can re-rank
   popular tables higher.

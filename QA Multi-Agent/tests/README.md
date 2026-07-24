# QA Agent — Test Suite

This directory contains the unit-test suite for the multi-agent QA
system. The suite is plain `pytest` — **no LLM API key, no Playwright
browser, no real Telegram bot required.** Every LLM call is replaced
with a deterministic stub so tests run in seconds and on any machine.

## Quick start

```bash
# from the project root
pip install -r requirements.txt
pytest -q
```

You should see something like:

```
67 passed in ~5s
```

## Layout

```
tests/
├── README.md                          ← this file
├── conftest.py                        ← path bootstrap + env defaults
├── unit/                              ← fast, no-LLM tests
│   ├── test_run_python.py             ← AST sandbox, stdout capture, errors
│   ├── test_file_reader.py            ← path validation, file vs. dir
│   ├── test_run_tests_path.py         ← /app/ → outputs/tests/ normalisation
│   ├── test_rag_and_summarization.py  ← Chroma-backed long-term memory
│   ├── test_call_human.py             ← interrupt-based user escalation
│   ├── test_agent_creation.py         ← LangGraph factory contract
│   └── test_prompts_and_models.py     ← prompt loading, model slots, logger
└── agent_evals/                       ← one file per team agent
    ├── _helpers.py                    ← StubGraph + install_fake_agent()
    ├── test_test_explorer_agent.py    ← contract: green/red/exception
    ├── test_writer_agent.py           ← contract: markdown, no test code
    ├── test_reviewer_agent.py         ← contract: ## Verdict + Decision
    └── test_call_human.py             ← contract: User responded: <X>
```

## What's covered

### `tests/unit/` — pure logic

| File | Module under test | What it asserts |
|------|------------------|-----------------|
| `test_run_python.py`        | `agents_core.tools.run_python` | AST-level block on `subprocess` / `socket` / `requests` / `webbrowser` / `pickle` / `sys` and on `exec` / `eval`; `print()` round-trips through the sandbox; `NameError` and `SyntaxError` are returned as structured `dict`s, not raised. |
| `test_file_reader.py`       | `agents_core.tools.file_reader` | Rejects paths outside `/app/data/`; directories get a clear "exactly ONE file" message; `.md` and `.ts` files are returned as plain text. |
| `test_run_tests_path.py`    | `agents_core.tools.run_tests`  | Strips the leading `/app/` prefix; rejects paths outside `tests/`; no `--headed` is added when no path is supplied. |
| `test_rag_and_summarization.py` | `agents_core.tools.{RAG_statements,summarization}` | Records persist to Chroma; RAG retrieves by semantic similarity; storage errors are surfaced as `"Could not add to the long-term memory, error: <msg>"`. |
| `test_call_human.py`        | `agents.human_call.call_human` | `interrupt` is called with the question; `context` is appended after `\n`; dict / list context is stringified; the return value is always `"User responded: <raw>"`. |
| `test_agent_creation.py`    | `agents_core.agent_creation.agent` | Returns a compiled graph; `bind_tools` is forwarded; `memory_flag=True` attaches an `InMemorySaver` checkpointer, `False` leaves the graph stateless. |
| `test_prompts_and_models.py` | `agents_core.{prompts,models,logger}` | All four prompts load and self-identify the agent role; `models.model_sample` is shared across the four slots; the agent logger is wired to a `TimedRotatingFileHandler`. |

### `tests/agent_evals/` — agent behavioural contract

These tests do *not* judge the LLM's output. They assert that the
**agent function** (the Python wrapper) preserves the response
contract the rest of the system depends on:

| Test file | Agent | Contract enforced |
|-----------|-------|-------------------|
| `test_test_explorer_agent.py` | `agents.agents.test_explorer_agent` | Returns a `**Result:** ✅|❌` line; announces the file path under `/app/data/outputs/tests/`; never leaks raw `<think>` blocks; LLM exceptions are returned as `Error: <msg>`. |
| `test_writer_agent.py`        | `agents.agents.writer_agent`        | Output is markdown (heading or list present); never emits Playwright test code; LLM exceptions are returned as `Error: <msg>`. |
| `test_reviewer_agent.py`      | `agents.agents.reviewer_agent`      | `## Verdict` header + `**Decision:**` line + a verdict emoji; at least one of `Critical Issues` / `Recommendations` section headers is present; covers all three verdicts (✅ ready, ❌ blockers, ⚠️ notes). |
| `test_call_human.py`          | `agents.human_call.call_human`      | `User responded: <X>` wrapper; context appended after `\n`; dict context is stringified; empty reply is valid; contract is idempotent across invocations. |

#### How the LLM is stubbed

`tests/agent_evals/_helpers.py` exposes two things:

* **`StubGraph`** — a drop-in `CompiledGraph` whose `ainvoke` returns
  whatever string you gave it as an `AIMessage`. The stub also
  records every call so tests can later assert on the call shape.
* **`install_fake_agent(monkeypatch, fake_response)`** — patches the
  `agent` factory in `agents_core.agent_creation` and in
  `agents.agents` so that the four team-agent functions receive the
  stub instead of the real LLM-backed graph.

The error-path tests additionally patch the factory to raise, then
verify the agent catches the exception and returns the documented
`Error: <msg>` string instead of propagating it.

## How to run

```bash
# everything
pytest

# only unit tests
pytest tests/unit

# only agent contract tests
pytest tests/agent_evals

# one agent
pytest tests/agent_evals/test_reviewer_agent.py

# verbose, with stdout
pytest -v -s

# with a specific keyword
pytest -k "explorer"
```

## Design notes

### No real LLM calls

The project wires every team agent to an `OpenAI`-backed `ChatOpenAI`
in `agents_core/models.py`. In a unit test we want a deterministic
reply, not a network call. The stub graph is a tiny dataclass that
returns whatever you ask for — `pip install -r requirements.txt` is
all you need.

### Path safety

`agents_core.tools` normalises paths by stripping `/app/` and then
checks that the result lives under `data/`. Tests for
`file_reader` and `run_tests` cover both the *happy* path (real
files) and the *rejection* path (everything else). The
`tests/conftest.py` `data_sandbox` fixture gives every test a clean
`data/inputs/...` tree to work in so the suite is hermetic.

### Async support

`pytest-asyncio` is configured in `pytest.ini` with
`asyncio_mode = auto`, so `async def test_*` works without an
explicit `@pytest.mark.asyncio` decorator.

### Environment defaults

`tests/conftest.py` seeds `os.environ` with safe placeholders for
the keys the project reads at import time (`api_key`, `api_model`,
`rec_limit`, etc.) so a fresh clone builds and runs without a
`.env` file. Real keys, if present, win.

## What's *not* covered

* **End-to-end LLM call** — out of scope for a unit suite. If you
  want to grade real LLM output, add a separate
  `tests/integration/` directory and gate it behind an env flag.
* **Telegram bot dispatcher** — `main.py` and the
  `handle_message` flow. The `call_human` tool is tested in
  isolation, but the Telegram plumbing is not.
* **Playwright MCP server** — `agents_core.tools.pw` requires a
  live browser. Use the existing `agent_evals` infrastructure to
  drive a dry-run of the explorer if you want, but don't try to
  spawn a real browser in CI.

## Adding a new test

1. Pick the right directory: `tests/unit/` for pure logic,
   `tests/agent_evals/` for agent contract.
2. Use the existing fixtures in `tests/conftest.py` (`data_sandbox`,
   `monkeypatch`) — don't fight the path setup.
3. For a new agent test, copy the structure of
   `test_reviewer_agent.py`. The pattern is: install a fake LLM
   reply, invoke the agent via `tool.ainvoke({...})`, assert on the
   response contract.
4. Don't add new dependencies to `requirements.txt` unless you
   absolutely have to.

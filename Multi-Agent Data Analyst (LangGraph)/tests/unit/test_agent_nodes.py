"""Tests for ``agents.agent_nodes`` — only functions with non-trivial logic.

LLM-touching functions (planner_call, planner_help, executor_call,
extractor_call, selector_call) are not covered.
"""
import subprocess

import pandas as pd
import pytest
from unittest.mock import MagicMock

from agents import agent_nodes
from agents.agent_nodes import (
    code_check,
    extractor_check,
    get_file_info,
    plan_check,
    python_interpreter,
    sql_interpreter,
    sql_tables_check,
    start_container,
    step_controller,
    stop_container,
)


# --------------------------------------------------------------------------- #
# extractor_check                                                             #
# --------------------------------------------------------------------------- #

class TestExtractorCheck:
    def test_json_parsing_problem(self, state_with):
        state = state_with(extra_text={"json_parsing_problem": "bad json"}, extractor_loop=1)
        result = extractor_check(state)
        assert result["extra_check_status"] == "error"
        assert result["extractor_loop"] == 2

    def test_low_confidence(self, state_with):
        state = state_with(extra_text={"confidence": 0.3, "type": "sql"})
        result = extractor_check(state)
        assert result["extra_check_status"] == "failure"

    def test_sql_type_success(self, state_with):
        state = state_with(extra_text={
            "confidence": 0.9, "type": "sql",
            "original_query": "x", "translated_query": "SELECT 1",
        })
        result = extractor_check(state)
        assert result["extra_check_status"] == "success"
        assert result["query_text"] == "SELECT 1"
        assert "extra_file_name" not in result

    def test_other_type_success(self, state_with):
        state = state_with(extra_text={
            "confidence": 0.8, "type": "file",
            "original_query": "x", "file_name": "data.csv",
        })
        result = extractor_check(state)
        assert result["extra_check_status"] == "success"
        assert result["extra_file_name"] == "data.csv"
        assert "query_text" not in result


# --------------------------------------------------------------------------- #
# plan_check                                                                  #
# --------------------------------------------------------------------------- #

def plan_text(steps=None, required_output="data/out.csv"):
    steps = steps if steps is not None else [{"action": "python_interpreter", "output": "x"}]
    return {
        "steps": steps,
        "metadata": {"complexity": "easy", "required_output": required_output},
    }


class TestPlanCheck:
    def test_json_parsing_problem_recreates(self, state_with):
        result = plan_check(state_with(plan_text={"json_parsing_problem": "bad"}))
        assert result["plan_recreate"] is True
        assert result["plan_loop"] == 1

    def test_missing_data_folder_recreates(self, state_with):
        result = plan_check(state_with(plan_text=plan_text(required_output="output.csv")))
        assert result["plan_recreate"] is True
        assert "data/" in result["plan_to_fix"]

    def test_valid_plan_first_time_increments_current_step(self, state_with):
        result = plan_check(state_with(plan_text=plan_text(), current_step=0))
        assert result["plan_recreate"] is False
        assert result["plan_loop"] == 0
        assert result["current_step"] == 1

    def test_empty_plan_recreates(self, state_with):
        result = plan_check(state_with(plan_text=plan_text(steps=[])))
        assert result["plan_recreate"] is True


# --------------------------------------------------------------------------- #
# step_controller                                                             #
# --------------------------------------------------------------------------- #

PLAN = [
    {"action": "python_interpreter", "output": "a"},
    {"action": "sql_query", "output": "b"},
    {"action": "python_interpreter", "output": "c"},
]


class TestStepController:
    def test_middle_step_clears_file_name(self, state_with):
        result = step_controller(state_with(current_step=1, plan=PLAN))
        assert result["last_step"] is False
        assert result["current_step_plan"]["output"] == "a"
        assert result["current_step_plan"]["file_name"] == ""

    def test_last_step_keeps_file_name(self, state_with):
        result = step_controller(state_with(current_step=3, plan=PLAN))
        assert result["last_step"] is True
        assert "file_name" not in result["current_step_plan"]

    def test_past_end(self, state_with):
        result = step_controller(state_with(current_step=4, plan=PLAN))
        assert result["end_of_steps"] is True


# --------------------------------------------------------------------------- #
# safety check                               #
# --------------------------------------------------------------------------- #

class TestCodeCheckSqlQuery:
    def test_clean_select_passes(self, state_with):
        assert code_check(state_with(code="SELECT * FROM users", tool_type="sql_query"))["step_code_ok"] is True

    def test_block_comment_is_rejected(self, state_with):
        result = code_check(state_with(code="SELECT 1 /* DROP TABLE x */", tool_type="sql_query"))
        assert result["step_code_ok"] is False
        assert "sql injections" in result["message_to_solve"]

    @pytest.mark.parametrize("forbidden", [
        "DELETE FROM users",
        "DROP TABLE x",
        "UPDATE users SET name=1",
        "INSERT INTO users VALUES (1)",
    ])
    def test_destructive_sql_is_rejected(self, state_with, forbidden):
        result = code_check(state_with(code=forbidden, tool_type="sql_query"))
        assert result["step_code_ok"] is False

    def test_parse_error_is_rejected(self, state_with):
        result = code_check(state_with(code="asdf qwer zxcv", tool_type="sql_query"))
        assert result["step_code_ok"] is False


class TestCodeCheckPython:
    def test_clean_python_passes(self, state_with):
        assert code_check(state_with(code="```python\nx = 1\nx```", tool_type="python_interpreter"))["step_code_ok"] is True

    @pytest.mark.parametrize("forbidden", [
        "```python\nimport os\nos```",
        "```python\nimport subprocess\nx```",
        "```python\nfrom subprocess import call\nfoo```",
        "```python\nexec('x=1')\nfoo```",
        "```python\neval('1+1')\nfoo```",
    ])
    def test_forbidden_import_or_call(self, state_with, forbidden):
        result = code_check(state_with(code=forbidden, tool_type="python_interpreter"))
        assert result["step_code_ok"] is False

    def test_to_csv_on_non_last_step_rejected(self, state_with):
        result = code_check(state_with(code="```python\ndf.to_csv('x')\n```", tool_type="python_interpreter"))
        assert result["step_code_ok"] is False

    def test_to_csv_on_last_step_allowed(self, state_with):
        result = code_check(state_with(
            code="```python\ndf.to_csv('x')\n```",
            tool_type="python_interpreter", last_step=True,
        ))
        assert result["step_code_ok"] is True

    def test_no_save_on_last_step_rejected(self, state_with):
        result = code_check(state_with(
            code="```python\nx = 1\nx```",
            tool_type="python_interpreter", last_step=True,
        ))
        assert result["step_code_ok"] is False


# --------------------------------------------------------------------------- #
# File info check                                        #
# --------------------------------------------------------------------------- #

class TestGetFileInfo:
    def test_csv_uses_read_csv(self, monkeypatch, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
        real_read_csv = agent_nodes.pd.read_csv
        monkeypatch.setattr(
            agent_nodes.pd, "read_csv",
            lambda p, *a, **k: real_read_csv(str(path), *a, **k),
        )
        result = get_file_info({"extra_file_name": "data.csv"})
        assert "a" in result["file_info"]


# --------------------------------------------------------------------------- #
# sql_tables_check                                  #
# --------------------------------------------------------------------------- #

def _fake_collection(items):
    """Build a chromadb-like MagicMock backed by a simple dict."""
    storage = {it["table_name"]: it for it in items}

    def _get(where=None, **_):
        names = []
        if where and "table_name" in where:
            names = [n for n in where["table_name"].get("$in", []) if n in storage]
        return {
            "documents": [storage[n]["doc"] for n in names],
            "metadatas": [storage[n]["meta"] for n in names],
            "ids": names,
        }

    coll = MagicMock()
    coll.get.side_effect = _get
    return coll


class TestSqlTablesCheck:
    def test_json_parsing_problem(self, state_with, monkeypatch):
        monkeypatch.setattr(agent_nodes, "tables_collection", MagicMock())
        result = sql_tables_check(state_with(sql_tables={"json_parsing_problem": "bad"}))
        assert result["sql_check_status"] == "error"

    def test_empty_table_list_increases_top_k(self, state_with, monkeypatch):
        monkeypatch.setattr(agent_nodes, "tables_collection", MagicMock())
        result = sql_tables_check(state_with(sql_tables={"selected_tables": []}, top_k=10))
        assert result["sql_check_status"] == "failure"
        assert result["top_k"] == 15

    def test_hallucinated_tables_increases_top_k(self, state_with, monkeypatch):
        coll = _fake_collection([{"table_name": "t1", "doc": "d", "meta": {"table_name": "t1"}}])
        monkeypatch.setattr(agent_nodes, "tables_collection", coll)
        result = sql_tables_check(state_with(
            sql_tables={"selected_tables": [{"table_name": "t1"}, {"table_name": "ghost"}]},
            top_k=20,
        ))
        assert result["sql_check_status"] == "failure"
        assert result["top_k"] == 25

    def test_success_attaches_info(self, state_with, monkeypatch):
        coll = _fake_collection([
            {"table_name": "orders", "doc": "orders doc", "meta": {"table_name": "orders"}},
            {"table_name": "users", "doc": "users doc", "meta": {"table_name": "users"}},
        ])
        monkeypatch.setattr(agent_nodes, "tables_collection", coll)
        result = sql_tables_check(state_with(sql_tables={
            "selected_tables": [{"table_name": "orders"}, {"table_name": "users"}],
        }))
        assert result["sql_check_status"] == "success"
        assert "orders doc" in result["selected_tables"]


# --------------------------------------------------------------------------- #
# sql_interpreter / python_interpreter — success/error/last_step              #
# --------------------------------------------------------------------------- #

class TestSqlInterpreter:
    def test_success_increments_step_and_clears_loops(self, monkeypatch, state_with):
        monkeypatch.setattr(
            agent_nodes, "run_sql",
            lambda c, o, l: {"status": "success", "output": [{"x": 1}]},
        )
        state = state_with(code="SELECT 1", current_step_plan={"output": "rows"}, current_step=2)
        result = sql_interpreter(state)
        assert result["sql_status"] == "success"
        assert result["current_step"] == 3
        assert result["sql_loop"] == 0

    def test_error_increments_sql_loop(self, monkeypatch, state_with):
        monkeypatch.setattr(
            agent_nodes, "run_sql",
            lambda c, o, l: {"status": "error", "output": "syntax error"},
        )
        state = state_with(code="BAD", current_step_plan={"output": "rows"}, current_step=1, sql_loop=0)
        result = sql_interpreter(state)
        assert result["sql_status"] == "error"
        assert result["sql_loop"] == 1


class TestPythonInterpreter:
    def test_success_strips_markdown(self, monkeypatch, state_with):
        captured = {}
        monkeypatch.setattr(
            agent_nodes, "run_code",
            lambda c, o, l: captured.update(code=c) or {"status": "success", "output": "5"},
        )
        state = state_with(code="```python\nx = 5\nx```", current_step_plan={"output": "r"}, current_step=1)
        result = python_interpreter(state)
        assert result["python_status"] == "success"
        assert "```" not in captured["code"]

    def test_no_result_means_error(self, monkeypatch, state_with):
        monkeypatch.setattr(
            agent_nodes, "run_code",
            lambda c, o, l: {"status": "success", "output": "No result"},
        )
        state = state_with(code="```python\nx = 1\n```", current_step_plan={"output": "r"}, current_step=1)
        result = python_interpreter(state)
        assert result["python_status"] == "error"
        assert result["python_loop"] == 1

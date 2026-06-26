"""Tests for ``route_next_step`` — graph routing for every phase/loop combo."""
import pytest
from langgraph.graph import END

from agents.agent_loop import route_next_step


# --------------------------------------------------------------------------- #
# Loop limits — one parametrized test covers all four loop counters          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("loop_key", ["plan_loop", "extractor_loop"])
def test_loop_above_3_stops(loop_key, state_with):
    state = state_with(**{loop_key: 4})
    assert route_next_step(state) == "stop_container"


@pytest.mark.parametrize("loop_key", ["same_step_loop", "python_loop", "sql_loop"])
def test_loop_above_3_first_time_asks_planner(loop_key, state_with):
    state = state_with(**{loop_key: 4}, help_loop=0)
    assert route_next_step(state) == "planner_help"


@pytest.mark.parametrize("loop_key", ["same_step_loop", "python_loop", "sql_loop"])
def test_loop_above_3_with_help_already_used_stops(loop_key, state_with):
    state = state_with(**{loop_key: 4}, help_loop=1)
    assert route_next_step(state) == "stop_container"


@pytest.mark.parametrize("status", ["error", "failure"])
def test_tables_check_loop_above_4_stops(status, state_with):
    state = state_with(tables_check_loop=5, sql_check_status=status)
    assert route_next_step(state) == "stop_container"


# --------------------------------------------------------------------------- #
# Phase dispatch                                                              #
# --------------------------------------------------------------------------- #

class TestStartContainer:
    def test_success(self, state_with):
        assert route_next_step(state_with(phase="START_CONTAINER", status_container="success")) == "extractor_call"

    def test_failure_ends(self, state_with):
        assert route_next_step(state_with(phase="START_CONTAINER", status_container="error")) == END


class TestExtractionCheck:
    def test_error_recalls_extractor(self, state_with):
        assert route_next_step(state_with(phase="EXTRACTION_CHECK", extra_check_status="error")) == "extractor_call"

    def test_failure_stops(self, state_with):
        state = state_with(phase="EXTRACTION_CHECK", extra_check_status="failure")
        assert route_next_step(state) == "stop_container"

    @pytest.mark.parametrize("extra_type,expected", [("sql", "table_rerank"), ("file", "get_file_info")])
    def test_success_routes_by_type(self, state_with, extra_type, expected):
        state = state_with(phase="EXTRACTION_CHECK", extra_check_status="success", extra_type=extra_type)
        assert route_next_step(state) == expected


class TestSqlTablesCheck:
    def test_success(self, state_with):
        assert route_next_step(state_with(phase="SQL_TABLES_CHECK", sql_check_status="success")) == "planner_call"

    def test_failure(self, state_with):
        assert route_next_step(state_with(phase="SQL_TABLES_CHECK", sql_check_status="failure")) == "table_rerank"

    def test_error(self, state_with):
        assert route_next_step(state_with(phase="SQL_TABLES_CHECK", sql_check_status="error")) == "selector_call"


class TestPlanCheck:
    def test_recreate_true(self, state_with):
        assert route_next_step(state_with(phase="PLAN_CHECK", plan_recreate=True)) == "planner_call"

    def test_recreate_false(self, state_with):
        assert route_next_step(state_with(phase="PLAN_CHECK", plan_recreate=False)) == "step_controller"


class TestStepController:
    def test_end_of_steps_stops(self, state_with):
        state = state_with(
            phase="STEP_CONTROLLER",
            current_step=1,
            end_of_steps=True,
        )
        assert route_next_step(state) == "stop_container"

    def test_not_end_calls_executor(self, state_with):
        state = state_with(phase="STEP_CONTROLLER", current_step=1, end_of_steps=False)
        assert route_next_step(state) == "executor_call"


class TestExecutorAndCodeCheck:
    def test_executor_goes_to_code_check(self, state_with):
        assert route_next_step(state_with(phase="EXECUTOR")) == "code_check"

    def test_code_check_failure_recalls_executor(self, state_with):
        assert route_next_step(state_with(phase="CODE_CHECK", step_code_ok=False)) == "executor_call"

    def test_code_check_ok_routes_by_tool_type(self, state_with):
        s1 = state_with(phase="CODE_CHECK", step_code_ok=True, tool_type="sql_query")
        s2 = state_with(phase="CODE_CHECK", step_code_ok=True, tool_type="python_interpreter")
        assert route_next_step(s1) == "sql_interpreter"
        assert route_next_step(s2) == "python_interpreter"


# --------------------------------------------------------------------------- #
# Finalization phases (PYTHON / SQL)                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("phase,status_key", [("PYTHON", "python_status"), ("SQL", "sql_status")])
def test_finalization_error_recalls_executor(phase, status_key, state_with):
    state = state_with(phase=phase, **{status_key: "error"})
    assert route_next_step(state) == "executor_call"


@pytest.mark.parametrize("phase,status_key", [("PYTHON", "python_status"), ("SQL", "sql_status")])
def test_finalization_success_to_step_controller(phase, status_key, state_with):
    state = state_with(phase=phase, **{status_key: "success"})
    assert route_next_step(state) == "step_controller"


# --------------------------------------------------------------------------- #
# Stop / unknown                                                              #
# --------------------------------------------------------------------------- #

def test_stop_container_ends(state_with):
    assert route_next_step(state_with(phase="STOP_CONTAINER")) == END


def test_unknown_phase_ends(state_with):
    assert route_next_step(state_with(phase="NOT_A_PHASE")) == END

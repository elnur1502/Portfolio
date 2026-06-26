import os

# Dummy API credentials
os.environ.setdefault("api_key", "test-dummy-key")
os.environ.setdefault("Openrouter_api", "test-dummy-key")
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

import pytest
from state import MessagesState


@pytest.fixture
def base_state() -> dict:
    return {
        "messages": [],
        "code": "",
        "tool_type": "",
        "current_step": 0,
        "end_of_steps": False,
        "last_step": False,
        "phase": "",
        "plan": [],
        "plan_text": {},
        "plan_recreate": False,
        "plan_to_fix": "",
        "current_step_plan": {},
        "required_output": "",
        # метрики
        "planner_input_tokens": 0,
        "planner_output_tokens": 0,
        "planner_sec": 1.0,
        "executor_input_tokens": 0,
        "executor_output_tokens": 0,
        "executor_sec": 1.0,
        "selector_input_tokens": 0,
        "selector_output_tokens": 0,
        "selector_sec": 1.0,
        "extractor_input_tokens": 0,
        "extractor_output_tokens": 0,
        "extractor_sec": 1.0,
        # errors
        "message_to_solve": "",
        "python_error": "",
        "python_status": "",
        "sql_error": "",
        "sql_status": "",
        "step_code_ok": False,
        # loops
        "plan_loop": 0,
        "sql_loop": 0,
        "python_loop": 0,
        "same_step_loop": 0,
        "help_loop": 0,
        "extractor_loop": 0,
        "tables_check_loop": 0,
        # sql-specific
        "query_text": "",
        "top_n": 10,
        "top_k": 30,
        "result_top_k_info": [],
        "selector_calls": 0,
        "sql_tables": {},
        "selected_tables": "",
        "sql_check_message": "",
        # extractor
        "extra_text": {},
        "original_task": "",
        "extra_check_status": "",
        "extra_check_message": "",
        "extra_type": "",
        "extra_confidence": "",
        "extractor_calls": 0,
        "extra_file_name": "",
        # container
        "status_container": "success",
        # file
        "file_info": "",
    }


@pytest.fixture
def state_with(base_state):
    def _make(**overrides):
        return {**base_state, **overrides}
    return _make

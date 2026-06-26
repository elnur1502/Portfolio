from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    code: str
    tool_type: str
    
    # Steps
    current_step: int
    end_of_steps: bool
    last_step: bool
    phase: str

    # Plan
    plan: list
    plan_text: dict
    plan_recreate: bool
    plan_to_fix: str
    current_step_plan: dict
    required_output: str

    # Metrics
    planner_input_tokens: int
    planner_output_tokens: int
    planner_sec: float
    executor_input_tokens: int
    executor_output_tokens: int
    executor_sec: float
    selector_input_tokens: int
    selector_output_tokens: int
    selector_sec: float
    extractor_input_tokens: int
    extractor_output_tokens: int
    extractor_sec: float
    
    # Errors
    message_to_solve: str
    python_error: str
    python_status: str
    sql_error: str
    sql_status: str
    step_code_ok: bool

    # Loops
    plan_loop: int
    sql_loop: int
    python_loop: int
    same_step_loop: int
    help_loop: int
    extractor_loop: int
    tables_check_loop: int

    # SQL-specific
    query_text: str
    top_n: int
    top_k: int
    result_top_k_info: list
    selector_calls: int
    sql_tables: dict
    selected_tables: str
    sql_check_message: str

    # Extractor-specific
    extra_text: dict
    original_task: str
    extra_check_status: str
    extra_check_message: str
    extra_type: str
    extra_confidence: str
    extractor_calls: int
    extra_file_name: str

    # Container
    status_container: str

    # File
    file_info: str
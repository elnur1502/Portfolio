import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from agents.agent_nodes import extractor_call
from deepeval.models import LocalModel
import os
from dotenv import load_dotenv


load_dotenv()

judge_model = LocalModel(
    base_url="https://ai.sumopod.com/v1",
    api_key=os.getenv("api_key"),
    model='MiniMax-M3',
    temperature = 0.0
)

correctness_metric = GEval(
    name="TaskTypeCorrectness",
    criteria=(
        "Determine whether the agent correctly classified the user task as "
        "'sql' (requires database) or 'file' (requires local file). "
        "The classification is correct if it matches the expected type."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=judge_model
)


@pytest.mark.parametrize("user_msg,expected_type,expected_file", [
    ("How many clients from Astana?", "sql", ""),
    ("Calculate income from 2025 to 2026", "sql", ""),
    ("Analyse this report: sales.csv", "file", "sales.csv"),
    ("Build a bar-chart from the data.xlsx", "file", "data.xlsx"),
    ("Give top 10 clients with the highest payments", "sql", ""),
])
def test_extractor_classifies_task_type(user_msg, expected_type, expected_file, state_with):
    """Task type correctness check"""

    state = state_with(messages=[user_msg])
    result = extractor_call(state)
    
    actual_output = result['extra_text']
    
    test_case = LLMTestCase(
        input=user_msg,
        actual_output=str(actual_output),
        expected_output=f"type={expected_type}, file_name={expected_file}",
    )
    
    assert_test(test_case, [correctness_metric])
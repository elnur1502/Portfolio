import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from agents.agent_nodes import planner_call
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

plan_validity_metric = GEval(
    name="PlanValidity",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    criteria=(
        "Evaluate the generated plan against these criteria:\n"
        "1. Are steps atomic (each does ONE logical operation)?\n"
        "2. Are variable names consistent and follow df_*/file_* convention?\n"
        "3. Does the LAST step use action='save_to_file'?\n"
        "4. Do intermediate steps avoid file I/O?\n"
        "5. Is the plan executable without interpretation?\n"
        "Score HIGH if 4-5 criteria met, MEDIUM if 2-3, LOW if 0-1."
    ),
    evaluation_steps=[
                "Check that the plan has at least one step.",
                "Check that each step has a clear instruction.",
                "Check that required_output starts with 'data/' and includes a file extension.",
                "Check that the last step is 'save_to_file' if the plan produces a final artifact.",
                "Check that step variables are referenced consistently.",
    ],
    threshold=0.7,
    model=judge_model
)


def test_planner_creates_valid_plan_for_simple_aggregation(state_with):
    task = "Calculate the sales sum from sales.csv and save the result to report.xlsx"
    
    state = state_with(file_info='sales.csv', original_task=task)
    result = planner_call(state)
    
    plan_dict = result['plan_text']
    
    test_case = LLMTestCase(
        input=task,
        actual_output=str(plan_dict),
        expected_output="Valid 2-3 step plan ending with save_to_file to report.xlsx",
    )
    
    assert_test(test_case, [plan_validity_metric])


import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from agents.agent_nodes import executor_call
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

correctness_metric_python = GEval(
    name="Code Quality Python",
    criteria=(
        "Evaluate whether the generated code is correct, executable, "
        "and follows the step instructions. The code must assign to the "
        "specified output variable, must not use forbidden modules "
        "(os, subprocess, socket, requests, webbrowser, pickle, sys), "
        "and must not use exec/eval."
    ),
    evaluation_steps=[
            "Check that the code is syntactically valid Python (compiles).",
            "Check that the code assigns to the required output variable.",
            "Check that no forbidden module is imported (os, subprocess, etc).",
            "Check that no exec/eval call is present.",
            "Check that the code implements the step's instructions.",
        ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=judge_model
)

def test_generates_correct_python(state_with):
    """Generated python correctness check"""

    state = state_with(current_step_plan={
                "output": "result",
                "instructions": (
                    "Compute 2 + 2 and assign to result. "
                    "Verify result == 4 before finishing."
                )
            })
    
    result = executor_call(state)
    actual_output = result['code']
    
    test_case = LLMTestCase(
        input="Compute 2 + 2 and assign to result. Verify result == 4.",
        actual_output=str(actual_output),
        expected_output="Valid python code",
    )

    assert_test(test_case, [correctness_metric_python])


correctness_metric_sql = GEval(
    name="Code Quality SQL",
    criteria=(
        "Evaluate whether the generated code is correct, executable, "
        "and follows the step instructions. The code must must not use forbidden commands "
        "(Drop,Delete,Insert,Update,TruncateTable,Grant,Alter,Create,Merge,Command) "
    ),
    evaluation_steps=[
            "Check that the code is syntactically valid SQL (compiles).",
            "Check that no forbidden command is used ((Drop,Delete,Insert, etc).",
            "Check that the code implements the step's instructions.",
        ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=judge_model
)

def test_generates_correct_sql(state_with):
    """Generated sql correctness check"""

    state = state_with(current_step_plan={
                "output": "result",
                "instructions": (
                    "Get client_id and tabnumber columns from ca_manager table. "
                    "Verify that result has exact 2 columns before finishing."
                )
            })
    
    result = executor_call(state)
    actual_output = result['code']
    
    test_case = LLMTestCase(
        input="Get client_id and tabnumber columns from ca_manager table. Verify that result has exact 2 columns.",
        actual_output=str(actual_output),
        expected_output="Valid sql code",
    )

    assert_test(test_case, [correctness_metric_sql])

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from agents.agent_nodes import planner_help
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

plan_help_validity_metric = GEval(
    name="PlanHelpValidity",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    criteria=(
        "Evaluate the generated helping plan against these criteria:\n"
        "1. Are steps atomic (each does ONE logical operation)?\n"
        "2. Are variable names consistent and follow df_*/file_* convention?\n"
        "3. Does the LAST step use action='save_to_file'?\n"
        "4. Do intermediate steps avoid file I/O?\n"
        "5. Is the plan executable without interpretation?\n"
        "6. Do previous steps remain unchanged? "
        "7. Does current and future(not necessary) steps are changed to make it more detailed to help executor?"
    ),
    evaluation_steps=[
                "Check that the plan has at least one step.",
                "Check that each step has a clear instruction.",
                "Check that required_output starts with 'data/' and includes a file extension.",
                "Check that the last step is 'save_to_file' if the plan produces a final artifact.",
                "Check that step variables are referenced consistently.",
                "Check that previous steps untouched."
                "Check that current and future(not neccessary) steps have been changed and become more detailed to help for executor."
    ],
    threshold=0.7,
    model=judge_model
)


def test_planner_creates_easier_plan_for_executor(state_with):
    task = "Extract all customer_account_id for BIN 220840036565, save as 22084bin.xlsx"
    
    state = state_with(
        plan=[{'step_id': 1, 'action': 'sql_query', 'description': 'Extract all customer account IDs (LS) for BIN 220840036565 from dmp_bdas dimension table', 'input': 'datamarts.dmp_bdas', 'output': 'df_ls_by_bin', 'instructions': "Select customer_account_id from datamarts.dmp_bdas where identification_number = '220840036565'. Ensure the result contains customer_account_id column and is not empty. Return all matching rows."}, {'step_id': 2, 'action': 'save_to_file', 'description': 'Save all LS numbers for BIN 220840036565 to Excel file', 'input': 'df_ls_by_bin', 'output': 'file_ls_report', 'file_name': 'data/22084bin.xlsx', 'instructions': 'Save df_ls_by_bin to XLSX file without index. Confirm file is created successfully.'}],
        current_step_plan=[{'step_id': 1, 'action': 'sql_query', 'description': 'Extract all customer account IDs (LS) for BIN 220840036565 from dmp_bdas dimension table', 'input': 'datamarts.dmp_bdas', 'output': 'df_ls_by_bin', 'instructions': "Select customer_account_id from datamarts.dmp_bdas where identification_number = '220840036565'. Ensure the result contains customer_account_id column and is not empty. Return all matching rows."}],
        code="Select customer_account_id from dmp_bdas where identification_number = '220840036565'", 
        python_error='', 
        sql_error="User does not have privileges to execute 'SELECT' on: default.dmp_bdas", 
        message_to_solve='')
    
    result = planner_help(state)
    
    plan_dict = result['plan_text']
    
    test_case = LLMTestCase(
        input=task,
        actual_output=str(plan_dict),
        expected_output="Valid more detailed plan for executore ending with save_to_file to data/22084bin.xlsx",
    )
    
    assert_test(test_case, [plan_help_validity_metric])


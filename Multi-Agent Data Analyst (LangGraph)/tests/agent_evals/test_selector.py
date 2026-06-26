import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, HallucinationMetric
from agents.agent_nodes import selector_call
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

selector_metric = GEval(
    name="Table Selection",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
    criteria=(
                "Evaluate whether the selector chose tables that are actually "
                "relevant to the user query AND exist in the candidate pool. "
                "A correct answer picks 1-5 tables from the candidates, not "
                "from outside. A 'failure' entry is acceptable when no "
                "candidate is relevant."
            ),
            evaluation_steps=[
                "Check that every selected table_name appears in the candidate list.",
                "Check that the number of selected tables is between 1 and 5.",
                "Check that the selected tables are relevant to the user query.",
                "Check that the reason for each selection mentions relevant columns or join keys.",
            ],
    threshold=0.7,
    model=judge_model
)

hallucination_metric = HallucinationMetric(
            model=judge_model,
            threshold=0.0,  # any hallucination fails
            async_mode=False,
        )


def test_selects_relevant_tables_for_client_query(state_with):
    candidates = [
            {
                "table_name": "orders",
                "doc": "orders: id, client_id, amount, created_at, status",
            },
            {
                "table_name": "clients",
                "doc": "clients: id, name, region, registration_date",
            },
            {
                "table_name": "products",
                "doc": "products: id, name, price, category",
            },
        ]
    
    task = "How many orders has client 42 made since last year?"
    
    state = state_with(result_top_k_info=[c["doc"] for c in candidates], query_text=task)
    result = selector_call(state)
    
    sql_tables = result['sql_tables']
    
    test_case = LLMTestCase(
        input=task,
        actual_output=str(sql_tables),
        context=[c["doc"] for c in candidates],
    )
    
    assert_test(test_case, [selector_metric])
    assert_test(test_case, [hallucination_metric])


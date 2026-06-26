import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(CURRENT_DIR, "..", "prompts")


sys_prompt_planner = open(os.path.join(PROMPTS_DIR, 'sys_prompt_planner.md'), encoding='UTF-8').read()
sys_prompt_executor = open(os.path.join(PROMPTS_DIR, 'sys_prompt_executor.md'), encoding='UTF-8').read()
reranker_prompt = open(os.path.join(PROMPTS_DIR, 'tables_reranker_prompt.md'), 'r', encoding='UTF-8').read()
extractor_prompt = open(os.path.join(PROMPTS_DIR, 'task_retriever_prompt.md'), 'r', encoding='UTF-8').read()
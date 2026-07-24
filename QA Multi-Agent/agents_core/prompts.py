import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(CURRENT_DIR, "..", "prompts")

supervisor_prompt = open(os.path.join(PROMPTS_DIR, 'supervisor_agent.md'), encoding='UTF-8').read()
test_explorer_prompt = open(os.path.join(PROMPTS_DIR, 'test_explorer_agent.md'), encoding='UTF-8').read()
reviewer_prompt = open(os.path.join(PROMPTS_DIR, 'reviewer_agent.md'), encoding='UTF-8').read()
writer_prompt = open(os.path.join(PROMPTS_DIR, 'writer_agent.md'), encoding='UTF-8').read()
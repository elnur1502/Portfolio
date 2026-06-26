from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

from agents.agent_schemas import *

load_dotenv()

# Free OpenRouter variants:
# google/gemma-4-26b-a4b-it:free
# poolside/laguna-xs.2:free
# nvidia/nemotron-3-super-120b-a12b:free
# openai/gpt-oss-120b:free

free_model = 'openai/gpt-oss-120b:free' ## good enough for coding and overal reasoning

#actually text-embedding-3-small is quite enough for small database, but there are a lot of business termins 
# therefore it's better to use larger embedding model
embed_model = OpenAIEmbeddings(
        base_url="https://ai.sumopod.com/v1",
        api_key=os.getenv("api_key"),
        model='text-embedding-3-large',
        max_retries=5,
        timeout=60.0
        )

# Executor and selector are cheaper models or even free as provided in this agentic workflow
model_executor = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("Openrouter_api"),
        model=free_model,
        temperature=0.5, # higher temperature to be able to solve errors
        max_retries=5,
        timeout=60.0
        )

model_extractor = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("Openrouter_api"),
        model=free_model,
        temperature=0.0, # strong prompt follow for strict json output
        max_retries=5,
        timeout=60.0
        )

model_selector = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("Openrouter_api"),
        model=free_model,
        temperature=0.0, # strong prompt follow for strict json output
        max_retries=5,
        timeout=60.0
        )

#Planner - smarter model
model_planner = ChatOpenAI(
        base_url="https://ai.sumopod.com/v1",
        api_key=os.getenv("api_key"),
        model='MiniMax-M2.7-highspeed',
        temperature=0.1, # strong prompt follow for strict json output
        top_p=0.95,
        max_retries=5,
        timeout=60.0
        )
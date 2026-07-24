from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

model_sample = ChatOpenAI(
        base_url=os.getenv("llm_base_url"), 
        api_key=os.getenv("api_key"),
        model=os.getenv("api_model"),
        temperature=0.0,
        max_retries=5,
        timeout=300.0
        )

#actually text-embedding-3-small is quite enough for small database, but there are a lot of business termins 
# therefore it's better to use larger embedding model
embed_model = OpenAIEmbeddings(
        base_url=os.getenv("openrouter_base_url"),
        api_key=os.getenv("openrouter_api"),
        model=os.getenv("embedding_model"),
        max_retries=5,
        timeout=60.0
        )

supervisor_model = model_sample

test_explorer_model = model_sample

reviewer_model = model_sample

writer_model = model_sample

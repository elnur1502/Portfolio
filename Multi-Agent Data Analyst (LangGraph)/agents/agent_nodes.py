import re
import pandas as pd
import numpy as np
from langchain_core.messages import SystemMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
import json
from jsonschema import validate
import time
import chromadb
import subprocess
import io
from agents.agent_models import *
from agents.agent_prompts import *
from agents.agent_tools import *
from agents.agent_logs import logger
from langchain_core.callbacks import UsageMetadataCallbackHandler
from pydantic import ValidationError
import sqlglot
from sqlglot import exp
import ast

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
Chroma_DB_DIR = os.path.join(CURRENT_DIR, "..", "chroma_db")

callback = UsageMetadataCallbackHandler()
client = chromadb.PersistentClient(path=Chroma_DB_DIR)
tables_collection = client.get_or_create_collection(name="tables")

def start_container(state: dict):
    try:
        subprocess.run(["docker", "start", 'agent_runner'], timeout=30, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Docker container started")
        return {"phase": "START_CONTAINER", "status_container": "success", "top_k": 30, "top_n": 10}
    except:
        try:
            subprocess.run(["docker", "stop", 'agent_runner'], timeout=30, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(10)
            subprocess.run(["docker", "start", 'agent_runner'], timeout=30, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Docker container started")
            return {"phase": "START_CONTAINER", "status_container": "success", "top_k": 30, "top_n": 10}
        except subprocess.CalledProcessError as e:
            logger.info("ERROR: Coudln't start Docker container. Error: " + str(e))
            return {"phase": "START_CONTAINER", "status_container": "error"}

def stop_container(state: dict):
    try:
        subprocess.run(["docker", "stop", 'agent_runner'], timeout=30, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Docker container stopped")
        return {"phase": "STOP_CONTAINER", "status_container": "success"}
    except subprocess.CalledProcessError as e:
        logger.info("ERROR: Coudln't stop Docker container. Error: " + str(e))
        return {"phase": "STOP_CONTAINER", "status_container": "error"}

def get_file_info(state:dict):
    file_name = state["extra_file_name"]
    file_extension = file_name.split('.')[1]
    if file_extension == 'xlsx':
        df = pd.read_excel('data/' + file_name)
    elif file_extension == 'csv':
        df = pd.read_csv('data/' + file_name)

    buffer = io.StringIO()
    df.info(buf=buffer)

    return {"phase": "FILE_INFO", "file_info": buffer.getvalue()}

def keyword_score(query, table_doc, table_metadata):
    score = 0
    
    for word in query.lower().replace('.', '').replace(',', '').split():
        score += table_metadata['usage_count']
        if word in table_metadata["table_name"].lower():
            score += 3
        if word in table_metadata["columns"].lower():
            score += 2
        if word in str(table_doc).lower():
            score += 1
            
    return score

def structured_invoke(model, schema, messages, callbacks):
    try:
        try:
            return model.with_structured_output(schema, method="json_mode").invoke(messages, config={"callbacks": callbacks})
        except:
            return model.with_structured_output(schema).invoke(messages, config={"callbacks": callbacks})
    except Exception as e:
        try:
            return model.with_structured_output(schema, method="json_mode").invoke(messages + ['JSON parsing problem: ' + str(e)], config={"callbacks": callbacks})
        except:
            return model.with_structured_output(schema).invoke(messages + ['JSON parsing problem: ' + str(e)], config={"callbacks": callbacks})

def extractor_check(state:dict):
    extra_pre = state['extra_text']
    if 'json_parsing_problem' in extra_pre.keys():
        logger.info('ERROR: Problem in JSON validation: ' + str(extra_pre['json_parsing_problem']))
        return {'phase': "EXTRACTION_CHECK", "extractor_loop": state.get("extractor_loop", 0)+1, "extra_check_status": 'error', 'extra_check_message': 'Problem in JSON validation: ' + str(extra_pre['json_parsing_problem'])}
    else:
        if extra_pre['confidence'] < 0.6:
            logger.info("ERROR: Coudln't get acceptable confidence more than 0.6. Please provide new task with additional information.")
            return {'phase': "EXTRACTION_CHECK", "extra_check_status": 'failure', 'extra_check_message': "Coudln't get acceptable confidence more than 0.6. Please provide new task with additional information.", "extra_coinf": extra_pre['confidence']}
        else:
            if extra_pre['type'] == 'sql':
                return {'phase': "EXTRACTION_CHECK", "extra_check_status": 'success', "extra_type": extra_pre['type'], "original_task": extra_pre['original_query'], "extra_confidence": str(extra_pre['confidence']), "query_text": str(extra_pre['translated_query']), "extractor_loop": 0, 'extra_check_message': ''}
            else:
                return {'phase': "EXTRACTION_CHECK", "extra_check_status": 'success', "extra_type": extra_pre['type'], "original_task": extra_pre['original_query'], "extra_confidence": str(extra_pre['confidence']), "extra_file_name": str(extra_pre['file_name']), "extractor_loop": 0, 'extra_check_message': ''}

def table_rerank(state: dict):
    query_text = state['query_text']
    query_embed = embed_model.embed_query(query_text)
    query = np.array([query_embed]).astype("float32")

    top_n = state['top_n']
    top_k = state['top_k']
    result_top_n = []
    result_top_k = []

    results = tables_collection.query(
        query_embeddings=query,
        n_results=top_n
        )

    for i in range(len(results['ids'][0])):
        score = keyword_score(query_text, results['documents'][0][i], results['metadatas'][0][i])
        table_data = {}
        embedding_score = top_n/(i+1)
        final_score = 0.6 * embedding_score + 0.4 * score
        table_data['table_name'] = str(results['metadatas'][0][i]['table_name'])
        table_data['score'] = final_score
        table_data['info'] = str(results['documents'][0][i])
        result_top_n.append(table_data)
        
    result_top_k = sorted(result_top_n, key = lambda x: x['score'], reverse=True)[:top_k]
    result_top_k_info = [t['info'] for t in result_top_k]

    logger.info('Selecting neccessary tables...')
    return {'phase': 'TABLE_RERANKING', 'result_top_k_info': result_top_k_info}

def sql_tables_check(state:dict):
    top_k = state['top_k']
    sql_tables = state['sql_tables']
    if 'json_parsing_problem' in sql_tables.keys():
        logger.info('ERROR: Problem in JSON validation: ' + str(sql_tables['json_parsing_problem']))
        return {'phase': "SQL_TABLES_CHECK", "tables_check_loop": state.get("tables_check_loop", 0)+1, "sql_check_status": 'error', 'sql_check_message': 'Problem in JSON validation: ' + str(sql_tables['json_parsing_problem'])}
    else:
        tables = sql_tables['selected_tables']
        result_tab = tables_collection.get(where={'table_name':{"$in": [t['table_name'] for t in tables]}})['documents']
        if len(tables) == 0:
            logger.info("ERROR: You coudln't select tables, but now the number of suggested tables is increased. Try again and select neccessary tables for the user's task.")
            return {'phase': "SQL_TABLES_CHECK", "tables_check_loop": state.get("tables_check_loop", 0)+1, "sql_check_status": 'failure', 'top_k': top_k+5, 'sql_check_message': "You coudln't select tables, but now the number of suggested tables is increased. Try again and select neccessary tables for the user's task."}
        else:
            if len(tables) != len(result_tab):
                logger.info("ERROR: One or more tables you selected not actually exists. For now the number of suggested tables is increased. Try again and select neccessary tables for the user's task and DO NOT hallucinate.")
                return {'phase': "SQL_TABLES_CHECK", "tables_check_loop": state.get("tables_check_loop", 0)+1, "sql_check_status": 'failure', 'top_k': top_k+5, 'sql_check_message': "One or more tables you selected not actually exists. For now the number of suggested tables is increased. Try again and select neccessary tables for the user's task and DO NOT hallucinate."}
            else:
                for i in range(len(result_tab)):
                    tables[i]['info'] = result_tab[i]
                if tables[0]['table_name'] == 'failure':
                    logger.info("ERROR: You coudln't select tables, but now the number of suggested tables is increased. Try again and select neccessary tables for the user's task.")
                    return {'phase': "SQL_TABLES_CHECK", "tables_check_loop": state.get("tables_check_loop", 0)+1, "sql_check_status": 'failure', 'top_k': top_k+5, 'sql_check_message': "You coudln't select tables, but now the number of suggested tables is increased. Try again and select neccessary tables for the user's task."}
                else:
                    return {'phase': "SQL_TABLES_CHECK", "sql_check_status": 'success', 'selected_tables': str(tables), "tables_check_loop": 0, 'sql_check_message': ''}

def sql_interpreter(state: dict):
    """
    An SQL interpreter for data extraction via DOCKER sandbox
    ONLY SELECT allowed.
    All other operations are prohibited, ESPECIALLY ATTEMPTS TO DELETE DATA!
    """

    code = state["code"]
    output_name = str(state["current_step_plan"]["output"])
    last_step = state.get('last_step', False)

    res = run_sql(code, output_name, last_step)

    if res["status"] == 'success':
            return {"sql_status": "success",
                    "list_of_variables": [{output_name: {"schema": res['output'], "created_at_step": state["current_step"], "used_code": str(code)}}],
                    "current_step": state['current_step'] + 1,
                    "message_to_solve": '',
                    "sql_error": '',
                    "sql_loop": 0,
                    "python_error": '',
                    "python_loop": 0,
                    "help_loop": 0,
                    "phase": 'SQL'}
    else:
        return {"sql_status": "error",
                    "sql_error": "Query error:" + str(res['output']) + ". Your code: " + str(code),
                    "message_to_solve": '',
                    "python_error": '',
                    "python_loop": 0,
                    "sql_loop": state.get("sql_loop", 0) + 1,
                    "phase": 'SQL'}

def python_interpreter(state: dict):
        """
        A secure Python interpreter for data analysis using DOCKER sandbox
        """

        try:
            code = state["code"].split('python')[1].replace('```', '')
        except:
            code = state["code"].replace('```', '')
        output_name = str(state["current_step_plan"]["output"])
        last_step = state.get('last_step', False)

        res = run_code(code, output_name, last_step)
        if res["status"] == 'success':
                if str(res["output"]) == 'No result':
                    logger.info("ERROR: No output variable named exactly as in the current step plan. Your code: " + str(code))
                    return {"python_status": "error",
                        "python_error": "No output variable named exactly as in the current step plan. Your code: " + str(code),
                        "message_to_solve": '',
                        "sql_error": '',
                        "sql_loop": 0,
                        "python_loop": state.get("python_loop", 0) + 1,
                        "phase": 'PYTHON'}
                else:
                    return {"python_status": "success",
                            "list_of_variables": [{output_name: {"schema": res['output'], "created_at_step": state["current_step"], "used_code": str(code)}}],
                            "current_step": state['current_step'] + 1,
                            "message_to_solve": '',
                            "sql_error": '',
                            "sql_loop": 0,
                            "python_error": '',
                            "python_loop": 0,
                            "help_loop": 0,
                            "phase": 'PYTHON'}
        else:
            logger.info("ERROR: Interpreter error:" + str(res['output']) + ". Your code: " + str(code))
            return {"python_status": "error",
                        "python_error": "Interpreter error:" + str(res['output']) + ". Your code: " + str(code),
                        "message_to_solve": '',
                        "sql_error": '',
                        "sql_loop": 0,
                        "python_loop": state.get("python_loop", 0) + 1,
                        "phase": 'PYTHON'}

def plan_check(state:dict):
    logger.info("Plan checking...")
    help_loop = state.get('help_loop', 0)
    plan_pre = state["plan_text"]
    if 'json_parsing_problem' in plan_pre.keys():
        logger.info('ERROR: Problem in JSON validation: ' + str(plan_pre['json_parsing_problem']))
        if help_loop == 0:
            return {"plan_to_fix": 'Problem in JSON validation: ' + str(plan_pre['json_parsing_problem']),
                                        "plan_recreate": True,
                                        "plan_loop": state.get("plan_loop", 0) + 1,
                                        "current_step": 0,
                                        "phase": 'PLAN_CHECK'
                                        }
        else:
            return {"plan_to_fix": 'Problem in JSON validation: ' + str(plan_pre['json_parsing_problem']),
                                        "plan_recreate": True,
                                        "plan_loop": state.get("plan_loop", 0) + 1,
                                        "phase": 'PLAN_CHECK'
                                        }
    else:
        plan = plan_pre['steps']
        metadata = plan_pre['metadata']
        logger.info('Metadata: ' + str(metadata))
        logger.info('Plan: ' + str(plan))
        
        if 'data/' not in metadata["required_output"]:
            logger.info("ERROR: Folder does not stated in the file name. Provide file name with data/ folder in it.")
            if help_loop == 0:
                return {"plan_to_fix": "Folder does not stated in the file name. Provide file name with data/ folder in it.",
                                    "plan_recreate": True,
                                    "plan_loop": state.get("plan_loop", 0) + 1,
                                    "current_step": 0,
                                    "phase": 'PLAN_CHECK'
                                    }      
            else:
                return {"plan_to_fix": "Folder does not stated in the file name. Provide file name with data/ folder in it.",
                                    "plan_recreate": True,
                                    "plan_loop": state.get("plan_loop", 0) + 1,
                                    "phase": 'PLAN_CHECK'
                                    }      
        else:
            pass
        #len_check
        if help_loop == 0:
            if len(plan) >= 1:
                #json_check
                return {"current_step": state.get('current_step', 0) + 1,
                                "phase": 'PLAN_CHECK',
                                "plan_to_fix": '',
                                "plan_loop": 0,
                                "plan_recreate": False,
                                "plan": plan,
                                'complexity': metadata['complexity'],
                                'required_output': metadata['required_output']
                                }
            else:
                logger.info("ERROR: Plan doesn't have any step. Provide executable plan at least with one step.")
                return {"plan_to_fix": "Plan doesn't have any step. Provide executable plan at least with one step.",
                                    "plan_recreate": True,
                                    "plan_loop": state.get("plan_loop", 0) + 1,
                                    "current_step": 0,
                                    "phase": 'PLAN_CHECK'
                                    }      
        else:
            if len(plan) >= 1:
                return {"phase": 'PLAN_CHECK',
                                "plan_to_fix": '',
                                "plan_loop": 0,
                                "python_loop": 0,
                                "same_step_loop": 0,
                                "sql_error": '',
                                "sql_loop": 0,
                                "message_to_solve": '',
                                "plan_recreate": False,
                                "plan": plan,
                                'complexity': metadata['complexity'],
                                'required_output': metadata['required_output']
                                }
            else:
                logger.info("ERROR: Plan doesn't have any step. Provide executable plan at least with one step.")
                return {"plan_to_fix": "Plan doesn't have any step. Provide executable plan at least with one step.",
                                    "plan_recreate": True,
                                    "plan_loop": state.get("plan_loop", 0) + 1,
                                    "phase": 'PLAN_CHECK',
                                    "python_loop": 0,
                                    "same_step_loop": 0,
                                    "sql_error": '',
                                    "sql_loop": 0
                                    }      
            
def step_controller(state: dict):
    cur_step = state["current_step"]
    plan = state["plan"]
    if cur_step <= len(plan):
        step_plan = plan[cur_step-1]
        if cur_step == len(plan):
            return {"last_step": True,
                    "current_step_plan": step_plan,
                    "tool_type": step_plan['action'],
                    "phase": 'STEP_CONTROLLER',
                    "end_of_steps": False}
        else:
            step_plan["file_name"] = ''
            return {"current_step_plan": step_plan,
                    "tool_type": step_plan['action'],
                    "phase": 'STEP_CONTROLLER',
                    "last_step": False,
                    "end_of_steps": False}
    else:
        return {"end_of_steps": True,
                "current_step_plan": {},
                "last_step": False,
                "phase": 'STEP_CONTROLLER'}

def code_check(state: dict):
    code = str(state["code"])
    logger.info('Checking code:\n' + str(code))
    tool_type = state["tool_type"]
    if tool_type == 'sql_query':
        # 1. Проверка на инъекции через комментарии
        if re.search(r'/\*.*?\*/', code, re.DOTALL):
            logger.info('ERROR: Problem in ' + str(code) + '\n' + 'You are not allowed to use sql injections.\nYou must rewrite executable query code without using sql injections!')
            return {"message_to_solve": 'Problem in ' + str(code) + '\n' + 'You are not allowed to use sql injections.\nYou must rewrite executable query code without using sql injections!',
                            "step_code_ok": False,
                            "same_step_loop": state.get("same_step_loop", 0) + 1,
                            "python_error": '',
                            "sql_error": '',
                            "phase": 'CODE_CHECK'}
        
        # 2. Проверка на запрещенные команды (case-insensitive)
        forbidden_functions = (exp.Drop,exp.Delete,exp.Insert, exp.Update,exp.TruncateTable,exp.Grant,exp.Alter,exp.Create,exp.Merge,exp.Command)
        try:
            parsed = sqlglot.parse_one(code, read='hive')
            for func in parsed.walk():
                if isinstance(func, forbidden_functions):
                    func_name = func.__class__.__name__.upper()
                    logger.info('ERROR: Problem in ' + str(code) + '\n' + 'You are not allowed to use command: ' + str(func_name) + '\nYou must rewrite executable query code without using forbidden sql command!')
                    return {"message_to_solve": 'Problem in ' + str(code) + '\n' + 'You are not allowed to use command: ' + str(func_name) + '\nYou must rewrite executable query code without using forbidden sql command!',
                                "step_code_ok": False,
                                "same_step_loop": state.get("same_step_loop", 0) + 1,
                                "python_error": '',
                                "sql_error": '',
                                "phase": 'CODE_CHECK'}
                
            return {"step_code_ok": True,
                    "phase": 'CODE_CHECK',
                    "python_error": '',
                    "sql_error": '',
                    "message_to_solve": '',
                    "same_step_loop": 0}
        
        except sqlglot.errors.ParseError as e:
            logger.info('ERROR: Problem with parsing: ' + str(code) + '\n' + 'Solve this error: ' + str(e))
            return {"message_to_solve": 'Problem with parsing: ' + str(code) + '\n' + 'Solve this error: ' + str(e),
                                "step_code_ok": False,
                                "same_step_loop": state.get("same_step_loop", 0) + 1,
                                "python_error": '',
                                "sql_error": '',
                                "phase": 'CODE_CHECK'}
        
    elif tool_type == 'python_interpreter' or tool_type == 'save_to_file':
        forbidden_modules = {'os', 'subprocess', 'socket', 'requests', 'webbrowser', 'pickle', 'sys'}
        try:
            clean_code = code.split('python')[1].replace('```', '')
        except:
            clean_code = code.replace('```', '')
        try:
            parsed = ast.parse(clean_code)
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base_module = alias.name.split('.')[0]
                        if base_module in forbidden_modules:
                            logger.info('ERROR: Problem in ' + str(code) + '\n' + "You are not allowed to use command: " + str(alias.name) + '\nYou must rewrite executable code without prohibited commands and libraries')
                            return {"message_to_solve": 'Problem in ' + str(code) + '\n' + "You are not allowed to use command: " + str(alias.name) + '\nYou must rewrite executable code without prohibited commands and libraries',
                                    "step_code_ok": False,
                                    "same_step_loop": state.get("same_step_loop", 0) + 1,
                                    "python_error": '',
                                    "phase": 'CODE_CHECK'}
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        base_module = node.module.split('.')[0]
                        if base_module in forbidden_modules:
                            logger.info('ERROR: Problem in ' + str(code) + '\n' + "You are not allowed to use command: " + str(node.module) + '\nYou must rewrite executable code without prohibited commands and libraries')
                            return {"message_to_solve": 'Problem in ' + str(code) + '\n' + "You are not allowed to use command: " + str(node.module) + '\nYou must rewrite executable code without prohibited commands and libraries',
                                    "step_code_ok": False,
                                    "same_step_loop": state.get("same_step_loop", 0) + 1,
                                    "python_error": '',
                                    "phase": 'CODE_CHECK'}
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in {'exec', 'eval'}:
                            logger.info('ERROR: Problem in ' + str(code) + '\n' + "You are not allowed to use command: " + str(node.func.id) + '\nYou must rewrite executable code without prohibited commands and libraries')
                            return {"message_to_solve": 'Problem in ' + str(code) + '\n' + "You are not allowed to use command: " + str(node.func.id) + '\nYou must rewrite executable code without prohibited commands and libraries',
                                    "step_code_ok": False,
                                    "same_step_loop": state.get("same_step_loop", 0) + 1,
                                    "python_error": '',
                                    "phase": 'CODE_CHECK'}

            last_step = state.get("last_step", False)
            if last_step == False:
                if '.to_csv' in str(code) or '.to_excel' in str(code) or '.to_json' in str(code):
                    logger.info('ERROR: Problem in ' + str(code) + '\n' + "You are not allowed to save to the file!" + '\nYou must rewrite executable code without saving the current result to the file.')
                    return {"message_to_solve": 'Problem in ' + str(code) + '\n' + "You are not allowed to save to the file!" + '\nYou must rewrite executable code without saving the current result to the file.',
                    "step_code_ok": False,
                    "same_step_loop": state.get("same_step_loop", 0) + 1,
                    "python_error": '',
                    "phase": 'CODE_CHECK'}
                else:
                    return {"step_code_ok": True,
                            "python_error": '',
                            "message_to_solve": '',
                            "phase": 'CODE_CHECK',
                            "same_step_loop": 0}
            else:
                if '.to_csv' in str(code) or '.to_excel' in str(code) or '.to_json' in str(code):
                    return {"step_code_ok": True,
                            "python_error": '',
                            "message_to_solve": '',
                            "phase": 'CODE_CHECK',
                            "same_step_loop": 0}
                else:
                    logger.info('ERROR: Problem in ' + str(code) + '\n' + "You must save to the file!" + '\nYou must rewrite executable code with saving the current result to the file.')
                    return {"message_to_solve": 'Problem in ' + str(code) + '\n' + "You must save to the file!" + '\nYou must rewrite executable code with saving the current result to the file.',
                    "step_code_ok": False,
                    "same_step_loop": state.get("same_step_loop", 0) + 1,
                    "python_error": '',
                    "phase": 'CODE_CHECK'} 
                    
        except Exception as e:
            print(e)
            logger.info('ERROR: Problem with parsing: ' + str(code) + '\n' + 'Solve this error: ' + str(e))
            return {"message_to_solve": 'Problem with parsing: ' + str(code) + '\n' + 'Solve this error: ' + str(e),
                                "step_code_ok": False,
                                "same_step_loop": state.get("same_step_loop", 0) + 1,
                                "python_error": '',
                                "sql_error": '',
                                "phase": 'CODE_CHECK'}
    else:
            return {"step_code_ok": True,
                    "python_error": '',
                    "message_to_solve": '',
                    "phase": 'CODE_CHECK',
                    "same_step_loop": 0} # to-do: if it's the question about the database for example

def planner_call(state: dict):
    """Planner gives a usefull plan for an executor"""
    file_info = state.get("file_info", "")
    sql_info = state.get("selected_tables", "")
    if file_info != "":
        additional_info = "File_info:\n" + file_info
    elif sql_info != "":
        additional_info = "SQL_tables:\n" + sql_info
    else:
        additional_info = ""
    logger.info('Plan creation...')
    
    try:
        start = time.time()
        plan_total = [
                structured_invoke(model_planner, Plan_Schema,
                    [
                        SystemMessage(
                            content=sys_prompt_planner
                        )
                    ]
                    + [state["original_task"]]
                    + [additional_info]
                    + [state.get("plan_to_fix",'')]
                , [callback])
            ]
        
        end = time.time()
        latency = end - start

        model_name = next(iter(callback.usage_metadata))
        usage_metadata = callback.usage_metadata[model_name]
        plan_text = plan_total[0].model_dump()

        input_toks = usage_metadata['input_tokens']
        output_toks = usage_metadata['output_tokens']
        tps = output_toks/latency
        logger.info('Planner info: input tokens = ' + str(input_toks) + ', output tokens = ' + str(output_toks) + ', latency = ' + str(latency) + ', tps = ' + str(tps))
        
        return {
            "messages": [plan_text],
            "planner_calls": state.get('planner_calls', 0) + 1,
            "plan_text": plan_text,
            "phase": 'PLAN',
            "planner_input_tokens": state.get("planner_input_tokens", 0) + input_toks,
            "planner_output_tokens": state.get("planner_output_tokens", 0) + output_toks,
            "planner_sec": state.get("planner_sec", 0) + latency
        }
    
    except Exception as e:
        stated_error = 'Error: ' + str(e) + 'Type' + str(type(e).__name__)
        return {
            "planner_calls": state.get('planner_calls', 0) + 1,
            "plan_text": {'json_parsing_problem': stated_error},
            "phase": 'PLAN'
        }

def planner_help(state: dict):
    """Planner helps the executor with additional information or by a given sub steps"""
    try:
        start = time.time()
        plan_total = [
                structured_invoke(model_planner, Plan_Schema,
                    [
                            SystemMessage(
                                content=sys_prompt_planner
                            )
                        ]
                        + ['Previous plan: ' + str(state["plan"])]
                        + ['Executor tried three times and could not properly execute the current step: ' + str(state['current_step_plan'])]
                        + ['Executor wrote this code for this step: ' + str(state['code'])]
                        + ['Executor got the very last error: ' + str(state["python_error"]) + str(state["sql_error"]) + str(state["message_to_solve"])]
                        + ['Your task is to help executor with the current step. You can NOT change previous steps, but you can change current and future steps to make it easier for executor. Also, you CAN add more info and hints into instructions section in your plan. DO NOT remove previous steps and DO NOT forget to add folowing steps until the task is finished.']
                    , [callback])
            ]

        end = time.time()
        latency = end - start

        model_name = next(iter(callback.usage_metadata))
        usage_metadata = callback.usage_metadata[model_name]
        plan_text = plan_total[0].model_dump()

        input_toks = usage_metadata['input_tokens']
        output_toks = usage_metadata['output_tokens']
        tps = output_toks/latency
        logger.info('Planner_help info: input_tokens = ' + str(input_toks) + ', output_tokens = ' + str(output_toks) + ', latency = ' + str(latency) + ', tps = ' + str(tps))

        return {
            "messages": [plan_text],
            "help_loop": state.get('help_loop', 0) + 1,
            "plan_text": plan_text,
            "phase": 'PLAN_HELP',
            "planner_input_tokens": state.get("planner_input_tokens", 0) + input_toks,
            "planner_output_tokens": state.get("planner_output_tokens", 0) + output_toks,
            "planner_sec": state.get("planner_sec", 0) + latency
        }

    except Exception as e:
        stated_error = 'Error: ' + str(e) + 'Type' + str(type(e).__name__)
        return {
            "help_loop": state.get('help_loop', 0) + 1,
            "plan_text": {'json_parsing_problem': stated_error},
            "phase": 'PLAN_HELP'
        }


def executor_call(state: dict):
    """Executor follows the plan given by planner"""
    list_of_variables = state.get("list_of_variables", [])
    start = time.time()
    if list_of_variables == []:
        code_total = model_executor.invoke(
                    [
                        SystemMessage(
                            content=sys_prompt_executor
                        )
                    ]
                    + ['\n' + str(state["current_step_plan"]) + '\n' + state.get("message_to_solve", '') + '\n' + state.get("python_error", '') + state.get("sql_error", '')]
                , config={"callbacks": [callback]})
    else:
        code_total = model_executor.invoke(
                    [
                        SystemMessage(
                            content=sys_prompt_executor
                        )
                    ]
                    + ['\n' + str(state["current_step_plan"]) + '\nCurrent step: ' + str(state["current_step"]) + '\nAvailable list of variables from previous steps: ' + str(state["list_of_variables"])+ '\n' + state.get("message_to_solve", '')+ '\n' + state.get("python_error", '') + state.get("sql_error", '')]
                   , config={"callbacks": [callback]}) 
        
    model_name = next(iter(callback.usage_metadata))
    usage_metadata = callback.usage_metadata[model_name]
    code = code_total.content

    end = time.time()
    latency = end - start

    input_toks = usage_metadata['input_tokens']
    output_toks = usage_metadata['output_tokens']
    tps = output_toks/latency
    logger.info('Executor info: input_tokens = ' + str(input_toks) + ', output_tokens = ' + str(output_toks) + ', latency = ' + str(latency) + ', tps = ' + str(tps))

    return {
        "messages": ['Executor: ' + str(code)],
        "executor_calls": state.get('executor_calls', 0) + 1,
        "code": str(code),
        "phase": "EXECUTOR",
        "executor_input_tokens": state.get("executor_input_tokens", 0) + input_toks,
        "executor_output_tokens": state.get("executor_output_tokens", 0) + output_toks,
        "executor_sec": state.get("executor_sec", 0) + latency
    }

def extractor_call(state: dict):
    """Extractor obtain task type (sql, file) and helps reranker while type is sql"""
    try:
        start = time.time()
        extraction_total = structured_invoke(model_extractor, Extra_Schema,
                    [
                        SystemMessage(
                            content=extractor_prompt
                        )
                    ]
                    + state["messages"]
                    + [state.get("extra_check_message",'')]
                , [callback])
        
        end = time.time()
        latency = end - start

        model_name = next(iter(callback.usage_metadata))
        usage_metadata = callback.usage_metadata[model_name]
        extra_text = extraction_total.model_dump()
        input_toks = usage_metadata['input_tokens']
        output_toks = usage_metadata['output_tokens']
        tps = output_toks/latency
        logger.info('Extractor info: input tokens = ' + str(input_toks) + ', output tokens = ' + str(output_toks) + ', latency = ' + str(latency) + ', tps = ' + str(tps))
        
        return {
            "messages": ['Extractor: ' + str(extra_text)],
            "extra_text": extra_text,
            "phase": 'EXTRACTOR',
            "extractor_calls": state.get('extractor_calls', 0) + 1,
            "extractor_input_tokens": state.get("extractor_input_tokens", 0) + input_toks,
            "extractor_output_tokens": state.get("extractor_output_tokens", 0) + output_toks,
            "extractor_sec": state.get("extractor_sec", 0) + latency
        }
    except Exception as e:
        stated_error = 'Error: ' + str(e) + 'Type' + str(type(e).__name__)
        return {
            "extra_text": {'json_parsing_problem': stated_error},
            "phase": 'EXTRACTOR',
            "extractor_calls": state.get('extractor_calls', 0) + 1
        }

def selector_call(state: dict):
    """Selector choose neccessary sql tables to complete the user's task"""
    result_top_k_info = state["result_top_k_info"]
    query_text = state['query_text']
    try:
        start = time.time()
        selected_tables = structured_invoke(model_selector, SQL_tables_Schema,
                        [
                            SystemMessage(
                                content=reranker_prompt
                            )
                        ] + ["\nUser query: " + str(query_text)]
                        + ['\nNeccessary information:\n' + str(result_top_k_info)]
                        + ['\n' + state.get('sql_check_message', '')]
                    , [callback]) 
        
        model_name = next(iter(callback.usage_metadata))
        usage_metadata = callback.usage_metadata[model_name]
        sql_tables = selected_tables.model_dump()

        end = time.time()
        latency = end - start

        input_toks = usage_metadata['input_tokens']
        output_toks = usage_metadata['output_tokens']
        tps = output_toks/latency
        logger.info('Selector info: input_tokens = ' + str(input_toks) + ', output_tokens = ' + str(output_toks) + ', latency = ' + str(latency) + ', tps = ' + str(tps))

        return {
            "messages": ['Selector: ' + str(sql_tables)],
            "selector_calls": state.get('selector_calls', 0) + 1,
            "sql_tables": sql_tables,
            "phase": "SELECTOR",
            "selector_input_tokens": state.get("selector_input_tokens", 0) + input_toks,
            "selector_output_tokens": state.get("selector_output_tokens", 0) + output_toks,
            "selector_sec": state.get("selector_sec", 0) + latency
        }
    except Exception as e:
        stated_error = 'Error: ' + str(e) + 'Type' + str(type(e).__name__)
        return {
            "selector_calls": state.get('selector_calls', 0) + 1,
            "sql_tables": {'json_parsing_problem': stated_error},
            "phase": "SELECTOR"
        }
        
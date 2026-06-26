from langgraph.graph import StateGraph, START, END
from agents.agent_logs import logger

def route_next_step(state: dict):
    plan_loop_num = state.get("plan_loop",0)
    same_step_loop = state.get("same_step_loop",0)
    sql_loop = state.get("sql_loop",0)
    python_loop = state.get('python_loop', 0)
    help_loop = state.get('help_loop', 0)
    extractor_loop = state.get('extractor_loop', 0)
    tables_check_loop = state.get('tables_check_loop', 0)
    phase = state["phase"]
    if plan_loop_num > 3: # no more than 3 retries for planning
        logger.info("Couldn't create a valid plan. Multi-agent stopped.")
        return "stop_container"
    elif same_step_loop > 3: # no more than 3 retries for executor code
        if help_loop == 0:
            logger.info("Couldn't create a valid code for a given step. Asking planner for help...") # no more than once for any step
            return "planner_help"
        else:
            logger.info("Couldn't create a valid code for a given step even with a planner help. Multi-agent stopped.")
            return "stop_container"
    elif python_loop > 3:
        if help_loop == 0:
            logger.info("Couldn't run the code. Asking planner for help...") # no more than once for any step
            return "planner_help"
        else:
            logger.info("Couldn't run the code even with a planner help. Multi-agent stopped.")
            return "stop_container"
    elif sql_loop > 3:
        if help_loop == 0:
            logger.info("Couldn't create a valid query code for a given step. Asking planner for help...") # no more than once for any step
            return "planner_help"
        else:
            logger.info("Couldn't create a valid query code for a given step even with a planner help. Multi-agent stopped.")
            return "stop_container"
    elif tables_check_loop > 4:
        if state['sql_check_status'] == 'error':
            logger.info("Couldn't create a valid JSON from a given top-k tables. Multi-agent stopped.")
            return "stop_container"
        else:
            logger.info("Couldn't select tables from a given top-k tables. Multi-agent stopped.")
            return "stop_container"
    elif extractor_loop > 3:
        logger.info("Couldn't create a valid JSON for a given task. Multi-agent stopped.")
        return "stop_container"
    else:
        if phase == 'START_CONTAINER':
            status_container = state["status_container"]
            if status_container == 'success':
                logger.info('Understanding the task...')
                return "extractor_call"
            else:
                return END
        #elif phase == "EXTRACTOR":
        #    return "extractor_check"
        elif phase == "EXTRACTION_CHECK":
            extra_check_status = state["extra_check_status"]
            if extra_check_status == 'error':
                logger.info('Fixing task extraction error...')
                return "extractor_call"
            elif extra_check_status == 'failure':
                logger.info(str(state['extra_check_message']))
                return "stop_container"
            else:
                extra_type = state["extra_type"]
                if extra_type == 'sql':
                    logger.info('Searching for sql tables and reranking them...')
                    return "table_rerank"
                else:
                    logger.info('Getting file information...')
                    return "get_file_info"
        elif phase == "SQL_TABLES_CHECK":
            sql_check_status = state["sql_check_status"]
            if sql_check_status == 'success':
                return "planner_call"
            elif sql_check_status == 'failure':
                logger.info("Selector coudln't choose any table. Giving more tables to select...")
                return "table_rerank"
            else:
                logger.info('Fixing tables selection error...')
                return "selector_call"
        elif phase == 'PLAN_CHECK':
            plan_recreate = state.get("plan_recreate", False)
            if plan_recreate == True:
                logger.info('Plan recreation...')
                return "planner_call"
            else:
                if help_loop == 0:
                    logger.info('Step controller...\nFirst step\n' + str('*')*50)
                else:
                    logger.info('Step controller...\nContinue with the steps from planner help\n' + str('*')*50)
                return "step_controller"
        elif phase == 'STEP_CONTROLLER':
            end_of_steps = state.get("end_of_steps", False)
            current_step = state["current_step"]
            if current_step >= 1:
                if end_of_steps == True:
                    logger.info('All done')
                    planner_input_tokens = state['planner_input_tokens']
                    planner_output_tokens = state['planner_output_tokens']
                    executor_input_tokens = state['executor_input_tokens']
                    executor_output_tokens = state['executor_output_tokens']
                    extractor_input_tokens = state['extractor_input_tokens']
                    extractor_output_tokens = state['extractor_output_tokens']
                    selector_input_tokens = state.get('selector_input_tokens', 0)
                    selector_output_tokens = state.get('selector_output_tokens', 0)
                    planner_sec = state.get('planner_sec', 1)
                    executor_sec = state.get('executor_sec', 1)
                    extractor_sec = state.get('extractor_sec', 1)
                    selector_sec = state.get('selector_sec', 1)
                    print(executor_sec)
                    try:
                        selector_tps = selector_output_tokens/selector_sec
                    except:
                        selector_tps = 0

                    logger.info('Total info:' + '\nPlanner: input_tokens = ' + str(planner_input_tokens) + ', output_tokens = ' 
                        + str(planner_output_tokens) + ', latency = ' + str(planner_sec) + ', average tps = ' + str(planner_output_tokens/planner_sec)
                        + '\nExecutor: input_tokens = ' + str(executor_input_tokens) + ', output_tokens = ' 
                        + str(executor_output_tokens) + ', latency = ' + str(executor_sec) + ', average tps = ' + str(executor_output_tokens/executor_sec)
                        + '\nExtractor: input_tokens = ' + str(extractor_input_tokens) + ', output_tokens = ' 
                        + str(extractor_output_tokens) + ', latency = ' + str(extractor_sec) + ', average tps = ' + str(extractor_output_tokens/extractor_sec)
                        + '\nSelector: input_tokens = ' + str(selector_input_tokens) + ', output_tokens = ' 
                        + str(selector_output_tokens) + ', latency = ' + str(selector_sec) + ', average tps = ' + str(selector_tps))
                    return "stop_container"
                else:
                    logger.info('Code creation...')
                    return "executor_call"
        elif phase == 'EXECUTOR':
            return "code_check"
        elif phase == 'CODE_CHECK':
            step_code_ok = state["step_code_ok"]
            if step_code_ok == False:
                logger.info('Code recreation...')
                return "executor_call"
            else:
                tool_type = state["tool_type"]
                if tool_type == 'sql_query':
                    logger.info('Query execution...')
                    return "sql_interpreter"
                else:
                    logger.info('Code execution...')
                    return "python_interpreter"
        elif phase == 'PYTHON':
            python_status = state['python_status']
            if python_status == 'error':
                logger.info('\nCode recreation...')
                return "executor_call"
            else:
                logger.info('\nNext step\n' + str('*')*50)
                return "step_controller"
        elif phase == 'SQL':
            sql_status = state['sql_status']
            if sql_status == 'error':
                logger.info('\nCode recreation...')
                return "executor_call"
            else:
                logger.info('\nNext step\n' + str('*')*50)
                return "step_controller"
        elif phase == 'STOP_CONTAINER':
            return END
        else:
            return END
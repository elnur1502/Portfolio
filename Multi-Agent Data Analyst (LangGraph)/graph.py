from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from state import MessagesState
from agents.agent_nodes import *
from agents.agent_loop import route_next_step

agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("start_container", start_container)
agent_builder.add_node("extractor_call", extractor_call)
agent_builder.add_node("extractor_check", extractor_check)
agent_builder.add_node("get_file_info", get_file_info)
agent_builder.add_node("table_rerank", table_rerank)
agent_builder.add_node("selector_call", selector_call)
agent_builder.add_node("sql_tables_check", sql_tables_check)
agent_builder.add_node("planner_call", planner_call)
agent_builder.add_node("planner_help", planner_help)
agent_builder.add_node("plan_check", plan_check)
agent_builder.add_node("step_controller", step_controller)
agent_builder.add_node("executor_call", executor_call)
agent_builder.add_node("code_check", code_check)
agent_builder.add_node("python_interpreter", python_interpreter)
agent_builder.add_node("sql_interpreter",sql_interpreter)
agent_builder.add_node("stop_container", stop_container)

# Add edges to connect nodes
agent_builder.add_edge(START, "start_container")
agent_builder.add_edge("extractor_call", "extractor_check")
agent_builder.add_edge("table_rerank", "selector_call")
agent_builder.add_edge("selector_call", "sql_tables_check")
agent_builder.add_edge("get_file_info", "planner_call")
agent_builder.add_edge("planner_call", "plan_check")
agent_builder.add_edge("planner_help", "plan_check")
agent_builder.add_edge("stop_container", END)

agent_builder.add_conditional_edges("start_container", route_next_step, ["extractor_call", END])
agent_builder.add_conditional_edges("extractor_check", route_next_step, ["extractor_call", "get_file_info", "table_rerank", "stop_container"])
agent_builder.add_conditional_edges("sql_tables_check", route_next_step, ["table_rerank", "selector_call", "planner_call", "stop_container"])
agent_builder.add_conditional_edges("plan_check", route_next_step, ["planner_call", "step_controller", "stop_container"])
agent_builder.add_conditional_edges("step_controller", route_next_step, ["executor_call", "stop_container"])
agent_builder.add_conditional_edges("executor_call", route_next_step, ["code_check"])
agent_builder.add_conditional_edges("code_check", route_next_step, ["executor_call", "python_interpreter", "sql_interpreter", "planner_help", "stop_container"])
agent_builder.add_conditional_edges("python_interpreter", route_next_step, ["step_controller", "executor_call", "planner_help", "stop_container"])
agent_builder.add_conditional_edges("sql_interpreter", route_next_step, ["step_controller", "executor_call", "planner_help", "stop_container"])

# Compile the agent
agent = agent_builder.compile()

# png_data = agent.get_graph(xray=True).draw_mermaid_png()

# with open("graph.png", "wb") as f:
#     f.write(png_data)
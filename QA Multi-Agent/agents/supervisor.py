from agents_core.models import supervisor_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from agents.agents import writer_agent, test_explorer_agent, reviewer_agent
from agents.human_call import call_human
from agents_core.tools import summarization, RAG_statements
from agents_core.prompts import supervisor_prompt
import asyncio

memory = MemorySaver()

async def build_supervisor():
    supervisor_tools = [writer_agent, test_explorer_agent, reviewer_agent, summarization, call_human, RAG_statements]
    supervisor_builder = StateGraph(MessagesState)
    agent_supervisor = supervisor_model.bind_tools(supervisor_tools)
    tools_node = ToolNode(supervisor_tools)

    async def call_supervisor(state):
        result = await agent_supervisor.ainvoke(
                            [
                                SystemMessage(
                                    content=supervisor_prompt
                                )
                            ] + state['messages'])
        return {"messages": [result]}
        
    def should_continue(state):
        last = state["messages"][-1]
        return "tools" if last.tool_calls else "__end__"
        
    supervisor_builder.add_node('call_supervisor', call_supervisor)
    supervisor_builder.add_node('tools', tools_node)

    supervisor_builder.add_edge(START, 'call_supervisor')
    supervisor_builder.add_edge('tools', 'call_supervisor')

    supervisor_builder.add_conditional_edges('call_supervisor', should_continue, ['tools', END])

    return supervisor_builder.compile(checkpointer=memory)
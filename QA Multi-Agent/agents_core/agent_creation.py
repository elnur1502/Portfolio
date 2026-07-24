from typing import Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from agents_core.logger import logger

async def agent(model, tools: list, sys_prompt: str, config, query, memory_flag: Optional[bool] = False):
    
    agent_builder = StateGraph(MessagesState)
    agent = model.bind_tools(tools)
    tools_node = ToolNode(tools)
     
    async def call_agent(state):
        result = await agent.ainvoke(
                            [
                                SystemMessage(
                                    content=sys_prompt
                                )
                            ] + state['messages'], config)
        
        return {"messages": [result]}
        
    def should_continue(state):
        last = state["messages"][-1]
        return "tools" if last.tool_calls else "__end__"
        
    agent_builder.add_node('call_agent', call_agent)
    agent_builder.add_node('tools', tools_node)

    agent_builder.add_edge(START, 'call_agent')
    agent_builder.add_edge('tools', 'call_agent')

    agent_builder.add_conditional_edges('call_agent', should_continue, ['tools', END])

    if memory_flag:
        memory = MemorySaver()
        graph = agent_builder.compile(checkpointer=memory)
    else:
        graph = agent_builder.compile()

    logger.info(f'Agent with model {model} and tools: {tools} has been successfully compiled.')
    return graph
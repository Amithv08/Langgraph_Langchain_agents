from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os 
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from typing import Literal
from langsmith import Client

from dotenv import load_dotenv
load_dotenv()

os.environ['LANGSMITH_TRACING']="true"
LANGSMITH_PROJECT= os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT=os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")





model = ChatOpenAI(model_name="gpt-4", temperature=0.2)

@tool
def multiply(a:int, b:int)->int:
    """Multiplies two integers and returns the result."""
    return a * b

@tool 
def add(a:int, b:int)->int:
    """Adds two integers and returns the result."""
    return a + b



tools = [multiply, add]

tools_by_name = {tool.name: tool for tool in tools}

# print(tools_by_name)

model_with_tools = model.bind_tools(tools)




# print(res.tool_calls)


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int 

#used to call the LLM and decide where to call tool or not
def llm_call(state:MessageState):

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(content= "You are a helpful assistant tasked with performing the arithematic operations")

                ]
                + state["messages"]
            )
        ], 
        "llm_calls" : (state.get("llm_calls") or 0) + 1
    }

#tool node to call the tools and return the results.
def tool_node(state:MessageState):
    "Performs the tool call"

    result = []
    for tool_call in state['messages'][-1].tool_calls:
        tool = tools_by_name[tool_call['name']]
        obseravation = tool.invoke(tool_call['args'])
        result.append(ToolMessage(content= obseravation, tool_call_id=tool_call['id']))
        return {"messages": result}
    

#conditional edge function to route to the tool node or end based on whether LLM made a tool call


def should_continue(state:MessageState) -> object:

    messages = state['messages']
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    

    return END

agent_builder = StateGraph(MessageState)


agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call",
                                     should_continue, 
                                     ["tool_node", END]
                                     )

agent_builder.add_edge("tool_node", "llm_call")



agent = agent_builder.compile()

# Expose the LangGraph graph for `langgraph dev`.
# The config in `langgraph.json` points to `./langgraph1.py:graph`.
graph = agent


# ----------------------------
# Simple agent (no tools)
# ----------------------------
# This is a minimal graph that just calls the chat model once and ends.
# It’s useful as a “hello world” to verify your LangGraph + LangSmith setup.

def simple_llm_call(state: MessageState):
    return {
        "messages": [
            model.invoke(
                [
                    SystemMessage(content="You are a concise, helpful assistant."),
                ]
                + state["messages"]
            )
        ],
        "llm_calls": (state.get("llm_calls") or 0) + 1,
    }


simple_builder = StateGraph(MessageState)
simple_builder.add_node("simple_llm_call", simple_llm_call)
simple_builder.add_edge(START, "simple_llm_call")
simple_builder.add_edge("simple_llm_call", END)

# Expose a second graph for `langgraph dev`.
simple_graph = simple_builder.compile()


def arithematic_agent(inputs:dict) -> dict:
    """
    Target function for Langsmith evaluation.
    """

    messages = HumanMessage(content= inputs['question'])
    result = agent.invoke({'messages': [messages]})


    final_message = result['messages'][-1].content

    return {
        "answer": final_message,
        "llm_calls": result.get("llm_calls", 0)
    }


client = Client()





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
from langsmith.evaluation import EvaluationResult

from dotenv import load_dotenv
load_dotenv()

os.environ['LANGSMITH_TRACING']="true"
LANGSMITH_PROJECT= "Evaluation_project"
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


def should_continue(state:MessageState) -> Literal["tool_node", END]:

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


def arithmetic_agent(inputs:dict) -> dict:
    """
    Target function for Langsmith evaluation.
    """

    messages = HumanMessage(content= inputs['question'])
    result = agent.invoke({'messages': [messages]})


    final_message = result['messages'][-1].content

    return {
        "answer": final_message,
    }


client = Client()

dataset_name = "arithematic_eval_dataset"



def numeric_evaluator(run, example) -> EvaluationResult:
    def _coerce_int(value):
        if value is None:
            raise ValueError("value is None")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            # Many answers include operands plus the final result.
            # Prefer the LAST integer-like token (often the final result),
            # e.g. "15 multiplied by 14 ... is 220." -> 220.
            import re
            matches = re.findall(r"-?\d+", value)
            if not matches:
                raise ValueError(f"no integer found in string: {value!r}")
            return int(matches[-1])
        raise TypeError(f"unsupported type: {type(value)}")

    def _get_answer(outputs: dict):
        if not outputs:
            return None
        # Prefer the key we use in this script  
        if "answer" in outputs:
            return outputs["answer"]                
        # Fallbacks if a different key was used
        for k in ("output", "result", "response"):
            if k in outputs:
                return outputs[k]
        return None

    try:
        predicted_raw = _get_answer(run.outputs)
        expected_raw = _get_answer(example.outputs)
        predicted = _coerce_int(predicted_raw)
        expected = _coerce_int(expected_raw)
        score = int(predicted == expected)
        comment = f"predicted={predicted} expected={expected}"
    except Exception as e:
        score = 0
        comment = f"evaluator_error={type(e).__name__}: {e}; run.outputs={run.outputs}; example.outputs={example.outputs}"

    return EvaluationResult(key="numeric_correctness", score=score, comment=comment)

from langsmith.evaluation import evaluate

evaluate(
    arithmetic_agent,
    data=dataset_name,
    evaluators=[numeric_evaluator],
    client=client,
    experiment_prefix="Arithmetic-Agent-Eval",
)







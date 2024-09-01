from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from agent.tools import hsk_vocab
from agent.llama_guard import llama_guard, LlamaGuardOutput


class AgentState(MessagesState):
    is_last_step: IsLastStep

# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True),
    "llama-3.1-70b": ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)
}

tools = [hsk_vocab]
instructions = f"""
    You are a helpful chinese language tutor with the ability to search the HSK Curriculum for relevant vocabulary.
    
    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Always use the HSK tool to retrieve HSK Curriculum vocabulary. 
    """

def wrap_model(model: BaseChatModel):
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

async def acall_model(state: AgentState, config: RunnableConfig):
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)
    if state["is_last_step"] and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))

agent.set_entry_point("model")


# Always run "model" after "tools"
agent.add_edge("tools", "model")

# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", END: END})

research_assistant = agent.compile(
    checkpointer=MemorySaver(),
)


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4
    from dotenv import load_dotenv

    load_dotenv()
    
    async def main():
        inputs = {"messages": [("user", "tell me hsk words for legal systems")]}
        result = await research_assistant.ainvoke(
            inputs,
            config=RunnableConfig(configurable={"thread_id": uuid4()}),
        )
        result["messages"][-1].pretty_print()

    asyncio.run(main())

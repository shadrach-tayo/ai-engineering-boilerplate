"""LangGraph agent that uses LiveMigrate memory tools over MCP + OAuth."""

from __future__ import annotations

import logging
import operator
import os
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from agent.oauth import DEFAULT_MCP_SERVER_URL, create_oauth_provider

logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)

# Shared across invocations so in-memory tokens / client registration persist
# for the lifetime of the langgraph process.
_oauth_auth = create_oauth_provider(MCP_SERVER_URL)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPT = """
You are a helpful assistant that help users manage their memory.

You can help users with the following tasks:
- Answer questions about the user's memory
- Help users create new memories
- Help users update existing memories
- Help users delete memories
- Help users search for memories

You will be given a set of tools to accomplish any of these tasks.
"""


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    user_id: str
    thread_id: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    messages: Annotated[list[BaseMessage], operator.add]


def _mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "memory": {
                "transport": "http",
                "url": MCP_SERVER_URL,
                "auth": _oauth_auth,
            }
        }
    )


async def get_tools(client: MultiServerMCPClient) -> list[BaseTool]:
    """Load tools from configured MCP servers."""
    return await client.get_tools()

async def run_tools(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Authenticate to MCP, bind tools, and invoke the model."""
    tools = await get_tools(client)
    return await ToolNode(tools).ainvoke(state)

async def call_model(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Authenticate to MCP, bind tools, and invoke the model."""
    client = _mcp_client()
    tools = await get_tools(client)
    llm_with_tools = model.bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state.messages,
    ]
    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}

client = _mcp_client()

graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_node("tools", run_tools)
    .add_edge("__start__", "call_model")
    .add_conditional_edges('call_model', tools_condition)
    .add_edge("tools", "call_model")
    .compile(name="New Graph")
)

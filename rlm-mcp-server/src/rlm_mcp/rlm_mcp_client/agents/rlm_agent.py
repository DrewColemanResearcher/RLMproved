from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

import httpx
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from rlm_mcp.rlm_mcp_client.prompts.prompts import RLM_SYSTEM_PROMPT


def build_mcp_http_client(
    mcp_headers: Mapping[str, str] | None = None,
    timeout: httpx.Timeout = httpx.Timeout(30.0, read=300.0),
) -> httpx.AsyncClient:
    headers = {
        "Accept": "application/json, text/event-stream",
    }
    if mcp_headers:
        headers.update(dict(mcp_headers))

    return httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
    )


def build_agent(
    *,
    mcp_headers: Mapping[str, str] | None = None,
    openrouter_api_key: str | None = None,
    root_model: str | None = 'openrouter/free',
    mcp_server_url: str = 'http://localhost:8000/mcp',
) -> Agent:
    # Get from env if available
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    resolved_api_key = openrouter_api_key or OPENROUTER_API_KEY
    if not resolved_api_key:
        raise ValueError("An OpenRouter API key must be provided via argument or OPENROUTER_API_KEY.")

    mcp_http_client = build_mcp_http_client(mcp_headers=mcp_headers)

    mcp_server = MCPServerStreamableHTTP(
        mcp_server_url,
        http_client=mcp_http_client,
        tool_prefix="mcp",              # optional, avoids name collisions
        include_instructions=True,      # optional, inject server instructions into agent context
        timeout=30,                     # init / call timeout
        read_timeout=300,               # long-running tool calls
    )

    model = OpenAIChatModel(
        root_model,
        provider=OpenRouterProvider(
            api_key=resolved_api_key,
        ),
    )

    return Agent(
        model=model,
        toolsets=[mcp_server],
        instructions=RLM_SYSTEM_PROMPT,
    )


async def agent_demo() -> None:
    from dotenv import load_dotenv
    # Only necessary for the demo
    load_dotenv(".env")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    mcp_headers = {
        "X-User-Id": "DemoAgent",
        "X-Chat-Id" : "Chat-DemoAgent",
        "X-Sub-LLM-Model": "openrouter/free",
        "X-Openrouter-API-Key": openrouter_api_key,
    }

    agent = build_agent(mcp_headers=mcp_headers, openrouter_api_key=openrouter_api_key, root_model="openrouter/free")

    # Opening the agent context up front is the recommended efficient pattern
    # when using MCP toolsets.
    async with agent:
        result = await agent.run(
            """
            You are an mcp tool tester, you test every mcp tools and write a report on exactly what works.
            Run all tools to see if you can and if everything is working.
            Your client is interested in the input and output of the tools you test.
            Your client is also interested in the problems of each tool if there are any.
            """
        )
        print(result.output)


if __name__ == "__main__":
    asyncio.run(agent_demo())
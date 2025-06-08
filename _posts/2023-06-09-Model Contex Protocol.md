---
layout: post
title: Model Context Protocol (MCP) in Python
published: true
date: 2025-06-06
---
Model Context Protocol (MCP) was open-sourced by Anthropic in November 2024 to standardize the way AI assistants connects to the systems where data lives, including content repositories, business tools and development environments.

AI assistants (LLMs and SLMs) currently rely on knowledge encoded into their parameters during training. However, as investment in agentic AI systems gain more adoption and with increasing investment on reasoning capability of language models, MCP allows users build truly agentic and complex workflows on top of AI models. Language models often require integration with data and tools to orchestrate and automate complex workflows. 

Other benefits of MCP includes:

- Flexibility to integrate LLMs and SLMs with a growing list of pre-built tools to directly plug into.
- The flexibility to leverage commoditized LLMs, offering users the room to switch between LLM providers and vendors.
- Best practices for securing your data within on-premise infrastructure.

#### MCP Servers

MCP servers relies on three (3) transport mechanisms:
- Standard Input Output: here servers run as a subprocess of users application running locally.
- HTTP over Server Side Encryption: servers run remotely and can be connected to via a url.
- Streamable HTTP where servers run remotely using the streamable HTTP transport defined in the MCP specification.

##### MCP Server in python

![](https://github.com/sodiqadewole/AI-Agents/blob/main/Creating_MCP_Server.md)


#### Run Server

 ```python 
 import asyncio
import os
import shutil
import subprocess
import time
from typing import Any

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStreamableHttp
from agents.model_settings import ModelSettings


async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to answer the questions.",
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(tool_choice="required"),
    )

    # Use the `add` tool to add two numbers
    message = "Add these numbers: 7 and 22."
    print(f"Running: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Run the `get_weather` tool
    message = "What's the weather in Tokyo?"
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Run the `get_secret_word` tool
    message = "What's the secret word?"
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)


async def main():
    async with MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": "http://localhost:8000/mcp",
        },
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="Streamable HTTP Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(server)

```

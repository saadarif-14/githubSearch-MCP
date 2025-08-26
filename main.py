import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.claude import Claude
from core.openai_provider import OpenAIProvider

from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()

# Provider selection
provider = os.getenv("PROVIDER", "openai").lower()

# Anthropic Config
# claude_model = os.getenv("CLAUDE_MODEL", "")
# anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")

# OpenAI Config
openai_model = os.getenv("OPENAI_MODEL", "")
openai_api_key = os.getenv("OPENAI_API_KEY", "")


if provider == "claude":
    assert claude_model, "Error: CLAUDE_MODEL cannot be empty. Update .env"
    assert anthropic_api_key, (
        "Error: ANTHROPIC_API_KEY cannot be empty. Update .env"
    )
elif provider == "openai":
    assert openai_model, "Error: OPENAI_MODEL cannot be empty. Update .env"
    assert openai_api_key, (
        "Error: OPENAI_API_KEY cannot be empty. Update .env"
    )
else:
    raise AssertionError("Error: PROVIDER must be 'claude' or 'openai'")


async def main():
    if provider == "claude":
        model_service = Claude(model=claude_model)
    else:
        model_service = OpenAIProvider(model=openai_model)

    server_scripts = sys.argv[1:]
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            claude_service=model_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())

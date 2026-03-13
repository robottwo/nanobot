"""LoadMCPTools meta-tool for incremental MCP tool loading."""

from typing import Any

from nanobot.agent.mcp_manager import MCPManager
from nanobot.agent.tools.base import Tool


class LoadMCPTools(Tool):
    """Meta-tool to load all tools from an MCP server."""

    def __init__(self, manager: MCPManager):
        self._manager = manager

    @property
    def name(self) -> str:
        return "load_mcp_tools"

    @property
    def description(self) -> str:
        servers = self._manager.get_available_servers()
        if not servers:
            return "Load tools from an MCP server (none configured)"
        return f"Load all tools from an MCP server. Available servers: {', '.join(servers)}"

    @property
    def parameters(self) -> dict[str, Any]:
        servers = self._manager.get_available_servers()
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "Name of the MCP server to load tools from",
                    "enum": servers,
                }
            },
            "required": ["server"],
        }

    async def execute(self, server: str) -> str:
        return self._manager.load_server(server)

"""MCP server manager for incremental tool loading."""

import asyncio
import html
from contextlib import AsyncExitStack
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.mcp import MCPToolWrapper, mcp_tool_name
from nanobot.agent.tools.registry import ToolRegistry


class MCPManager:
    """
    Manages MCP server connections with incremental tool loading.

    On startup, introspects each server to discover and cache tool definitions.
    The LLM can then use load_mcp_tools to register those cached tools.
    On session reset (/new), tools are unregistered but cache persists.

    Architecture:
    - MCP connection + tool definitions = persistent cache (survives /new)
    - Tool registration in registry = session-scoped (cleared on /new)
    - load_mcp_tools uses cache, no MCP round-trip needed
    """

    def __init__(self, configs: dict[str, Any], registry: ToolRegistry):
        self._configs = configs
        self._registry = registry
        self._tool_defs: dict[str, list[Any]] = {}  # server -> [tool_def objects]
        self._loaded_servers: set[str] = set()
        self._sessions: dict[str, Any] = {}  # server_name -> session
        self._stack: AsyncExitStack | None = None
        self._introspected = False
        self._introspecting = False

    async def introspect_all(self) -> None:
        """Connect to each server in parallel, discover and cache tool definitions."""
        if self._introspected or self._introspecting:
            return
        self._introspecting = True

        try:
            self._stack = AsyncExitStack()
            await self._stack.__aenter__()

            async def _introspect_one(name: str, cfg: Any) -> None:
                try:
                    session = await self._connect_server(name, cfg)
                    if session:
                        self._sessions[name] = session
                        tools = await session.list_tools()
                        self._tool_defs[name] = list(tools.tools)
                        logger.debug("MCP '{}': cached {} tool definitions", name, len(tools.tools))
                    else:
                        self._tool_defs[name] = []
                except Exception as e:
                    logger.warning("MCP '{}': failed to introspect: {}", name, e)
                    self._tool_defs[name] = []

            await asyncio.gather(
                *(_introspect_one(name, cfg) for name, cfg in self._configs.items())
            )

            self._introspected = True
        except Exception as e:
            logger.error("MCP introspection failed: {}", e)
            if self._stack:
                try:
                    await self._stack.aclose()
                except Exception:
                    pass
                self._stack = None
            raise
        finally:
            self._introspecting = False

    async def _connect_server(self, name: str, cfg: Any) -> Any | None:
        """Connect to a single MCP server, return session."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client

        timeout = cfg.connect_timeout

        transport_type = cfg.type
        if not transport_type:
            if cfg.command:
                transport_type = "stdio"
            elif cfg.url:
                transport_type = "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
            else:
                logger.warning("MCP '{}': no command or url configured", name)
                return None

        try:
            if transport_type == "stdio":
                params = StdioServerParameters(
                    command=cfg.command, args=cfg.args, env=cfg.env or None
                )
                read, write = await asyncio.wait_for(
                    self._stack.enter_async_context(stdio_client(params)),
                    timeout=timeout,
                )
            elif transport_type == "sse":

                def httpx_client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    merged_headers = {**(cfg.headers or {}), **(headers or {})}
                    return httpx.AsyncClient(
                        headers=merged_headers or None,
                        follow_redirects=True,
                        timeout=timeout,
                        auth=auth,
                    )

                read, write = await self._stack.enter_async_context(
                    sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
                )
            elif transport_type == "streamableHttp":
                from mcp.client.streamable_http import streamable_http_client

                http_client = await self._stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.headers or None,
                        follow_redirects=True,
                        timeout=None,
                    )
                )
                read, write, _ = await asyncio.wait_for(
                    self._stack.enter_async_context(
                        streamable_http_client(cfg.url, http_client=http_client)
                    ),
                    timeout=timeout,
                )
            else:
                logger.warning("MCP '{}': unknown transport type '{}'", name, transport_type)
                return None

            session = await self._stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            return session
        except asyncio.TimeoutError:
            logger.error("MCP '{}': connection timed out after {}s", name, timeout)
            return None
        except Exception as e:
            logger.error("MCP '{}': failed to connect: {}", name, e)
            return None

    def build_summary(self) -> str:
        """Build XML summary of MCP servers for system prompt."""
        if not self._tool_defs:
            return ""

        def escape(s: str) -> str:
            return html.escape(str(s))

        lines = ["<mcp_servers>"]
        for name, tools in self._tool_defs.items():
            loaded = name in self._loaded_servers
            desc = self._get_server_description(name, tools)
            lines.append(f'  <server loaded="{str(loaded).lower()}">')
            lines.append(f"    <name>{escape(name)}</name>")
            lines.append(f"    <description>{escape(desc)}</description>")
            lines.append(f"    <tool_count>{len(tools)}</tool_count>")
            lines.append("  </server>")
        lines.append("</mcp_servers>")
        return "\n".join(lines)

    def _get_server_description(self, name: str, tools: list) -> str:
        """Generate a high-level description from tool names."""
        if not tools:
            return f"MCP server: {name}"

        tool_names = [t.name for t in tools[:5]]
        desc = f"Provides {len(tools)} tool(s)"
        if tool_names:
            desc += f" including: {', '.join(tool_names)}"
            if len(tools) > 5:
                desc += ", ..."
        return desc

    def load_server(self, name: str) -> str:
        """Register cached tool definitions for a specific MCP server.

        Uses cached definitions - no MCP round-trip needed.
        Must be called after introspect_all() succeeds.
        """
        if name not in self._configs:
            available = list(self._configs.keys())
            return f"Error: MCP server '{name}' not found. Available: {', '.join(available)}"

        if not self._introspected:
            return "Error: MCP servers not yet introspected. Wait for startup to complete."

        if name in self._loaded_servers:
            return f"MCP server '{name}' tools already loaded."

        tool_defs = self._tool_defs.get(name, [])
        if not tool_defs:
            return (
                f"Error: No cached tool definitions for MCP server '{name}'. "
                "The server may have failed to connect during startup."
            )

        session = self._sessions.get(name)
        if not session:
            return (
                f"Error: No active session for MCP server '{name}'. "
                "The connection may have been lost."
            )

        loaded = []
        for tool_def in tool_defs:
            wrapper = MCPToolWrapper(session, name, tool_def, self._configs[name].tool_timeout)
            self._registry.register(wrapper)
            loaded.append(wrapper.name)

        self._loaded_servers.add(name)
        logger.info("MCP '{}': registered {} tools", name, len(loaded))
        return f"Loaded {len(loaded)} tools from MCP server '{name}': {', '.join(loaded)}"

    def unload_all(self) -> None:
        """Unregister all MCP tools from the registry.

        Keeps cache and sessions intact - tools can be reloaded without MCP round-trip.
        Called on /new to reset session-scoped tool state.
        """
        unloaded_count = 0
        for server_name in list(self._loaded_servers):
            tool_defs = self._tool_defs.get(server_name, [])
            for tool_def in tool_defs:
                tool_name = mcp_tool_name(server_name, tool_def.name)
                self._registry.unregister(tool_name)
                unloaded_count += 1
            logger.debug("MCP '{}': unregistered tools", server_name)
        self._loaded_servers.clear()
        if unloaded_count:
            logger.debug("Unregistered {} MCP tools total", unloaded_count)

    def is_loaded(self, name: str) -> bool:
        """Check if a server's tools have been loaded."""
        return name in self._loaded_servers

    def get_available_servers(self) -> list[str]:
        """Get list of configured MCP server names."""
        return list(self._configs.keys())

    async def close(self) -> None:
        """Close all MCP connections and clear all cached state."""
        if self._stack:
            try:
                await self._stack.aclose()
            except (RuntimeError, BaseExceptionGroup) as e:
                logger.debug("Error closing MCP stack: {}", e)
            self._stack = None
        self._sessions.clear()
        self._loaded_servers.clear()
        self._tool_defs.clear()
        self._introspected = False
        self._introspecting = False

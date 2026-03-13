"""Unit tests for MCPManager incremental tool loading."""

from pathlib import Path
from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.agent.context import ContextBuilder
from nanobot.agent.mcp_manager import MCPManager
from nanobot.agent.tools.load_mcp import LoadMCPTools
from nanobot.agent.tools.mcp import MCPToolWrapper, mcp_tool_name
from nanobot.agent.tools.registry import ToolRegistry


class MockMCPServerConfig:
    """Mock MCP server config."""

    def __init__(self, command=None, url=None, tool_timeout=30):
        self.command = command
        self.url = url
        self.args = []
        self.env = {}
        self.headers = {}
        self.tool_timeout = tool_timeout


class MockToolDef:
    """Mock MCP tool definition."""

    def __init__(self, name, description="A tool", input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


class MockToolList:
    """Mock MCP tool list response."""

    def __init__(self, tools):
        self.tools = tools


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def mcp_configs():
    return {
        "filesystem": MockMCPServerConfig(command="mcp-filesystem"),
        "github": MockMCPServerConfig(url="http://github-mcp.local"),
    }


class TestMCPToolName:
    """Tests for mcp_tool_name helper function."""

    def test_mcp_tool_name_format(self):
        assert mcp_tool_name("filesystem", "read_file") == "mcp_filesystem_read_file"
        assert mcp_tool_name("github", "create_issue") == "mcp_github_create_issue"

    def test_mcp_tool_name_consistency_with_wrapper(self, registry, mcp_configs):
        """Tool name from helper should match wrapper's name property."""
        tool_def = MockToolDef("test_tool", "Test")
        session = AsyncMock()
        wrapper = MCPToolWrapper(session, "myserver", tool_def)

        assert wrapper.name == mcp_tool_name("myserver", "test_tool")


class TestMCPManager:
    """Tests for MCPManager."""

    def test_get_available_servers(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        assert set(manager.get_available_servers()) == {"filesystem", "github"}

    def test_get_available_servers_empty(self, registry):
        manager = MCPManager({}, registry)
        assert manager.get_available_servers() == []

    def test_build_summary_empty(self, registry):
        manager = MCPManager({}, registry)
        assert manager.build_summary() == ""

    def test_build_summary_no_introspection(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        summary = manager.build_summary()
        assert summary == ""

    def test_build_summary_with_tools(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._tool_defs = {
            "filesystem": [
                MockToolDef("read_file", "Read a file"),
                MockToolDef("write_file", "Write a file"),
            ],
            "github": [
                MockToolDef("create_issue", "Create an issue"),
            ],
        }

        summary = manager.build_summary()
        assert "<mcp_servers>" in summary
        assert "</mcp_servers>" in summary
        assert "<name>filesystem</name>" in summary
        assert "<name>github</name>" in summary
        assert 'loaded="false"' in summary
        assert "<tool_count>2</tool_count>" in summary
        assert "<tool_count>1</tool_count>" in summary

    def test_build_summary_shows_loaded_status(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._tool_defs = {
            "filesystem": [MockToolDef("read_file", "Read a file")],
        }
        manager._loaded_servers.add("filesystem")

        summary = manager.build_summary()
        assert 'loaded="true"' in summary

    def test_build_summary_escapes_xml_special_chars(self, registry):
        """Server names and descriptions with XML chars should be escaped."""
        configs = {"test<server>": MockMCPServerConfig(command="mcp")}
        manager = MCPManager(configs, registry)
        manager._tool_defs = {
            "test<server>": [MockToolDef("tool&name", "Tool")],
        }

        summary = manager.build_summary()
        assert "<name>test&lt;server&gt;</name>" in summary
        assert "tool&amp;name" in summary
        assert "<name>test<server></name>" not in summary

    def test_get_server_description_empty_tools(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        desc = manager._get_server_description("test", [])
        assert desc == "MCP server: test"

    def test_get_server_description_with_tools(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        tools = [MockToolDef("read_file", "Read"), MockToolDef("write_file", "Write")]
        desc = manager._get_server_description("test", tools)
        assert "2 tool(s)" in desc
        assert "read_file" in desc
        assert "write_file" in desc

    def test_get_server_description_truncates_many_tools(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        tools = [MockToolDef(f"tool_{i}", f"Tool {i}") for i in range(10)]
        desc = manager._get_server_description("test", tools)
        assert "10 tool(s)" in desc
        assert "..." in desc

    def test_is_loaded(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        assert not manager.is_loaded("filesystem")
        manager._loaded_servers.add("filesystem")
        assert manager.is_loaded("filesystem")
        assert not manager.is_loaded("github")

    @pytest.mark.asyncio
    async def test_introspect_all_caches_tool_definitions(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)

        mock_session = AsyncMock()
        mock_session.list_tools.return_value = MockToolList(
            [
                MockToolDef("tool1", "First tool"),
                MockToolDef("tool2", "Second tool"),
            ]
        )

        with patch.object(manager, "_connect_server", return_value=mock_session):
            await manager.introspect_all()

        assert manager._introspected
        assert "filesystem" in manager._tool_defs
        assert len(manager._tool_defs["filesystem"]) == 2
        assert manager._tool_defs["filesystem"][0].name == "tool1"

    @pytest.mark.asyncio
    async def test_introspect_all_handles_connection_failure(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)

        with patch.object(manager, "_connect_server", return_value=None):
            await manager.introspect_all()

        assert manager._introspected
        assert manager._tool_defs.get("filesystem") == []

    @pytest.mark.asyncio
    async def test_introspect_all_only_runs_once(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True

        await manager.introspect_all()

        assert "filesystem" not in manager._tool_defs

    def test_load_server_registers_cached_tools(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True

        mock_session = AsyncMock()
        manager._sessions["filesystem"] = mock_session
        manager._tool_defs["filesystem"] = [
            MockToolDef("read_file", "Read a file"),
            MockToolDef("write_file", "Write a file"),
        ]

        result = manager.load_server("filesystem")

        assert "Loaded 2 tools" in result
        assert "mcp_filesystem_read_file" in result
        assert manager.is_loaded("filesystem")
        assert registry.has("mcp_filesystem_read_file")
        assert registry.has("mcp_filesystem_write_file")

    def test_load_server_already_loaded(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        manager._loaded_servers.add("filesystem")

        result = manager.load_server("filesystem")

        assert "already loaded" in result

    def test_load_server_not_found(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True

        result = manager.load_server("nonexistent")

        assert "not found" in result
        assert "Available:" in result

    def test_load_server_not_introspected(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)

        result = manager.load_server("filesystem")

        assert "not yet introspected" in result

    def test_load_server_no_cached_definitions(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        manager._tool_defs["filesystem"] = []

        result = manager.load_server("filesystem")

        assert "No cached tool definitions" in result

    def test_load_server_no_session(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        manager._tool_defs["filesystem"] = [MockToolDef("tool1", "Tool")]
        manager._sessions.pop("filesystem", None)

        result = manager.load_server("filesystem")

        assert "No active session" in result

    def test_unload_all_removes_tools_from_registry(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        manager._sessions["filesystem"] = AsyncMock()
        manager._tool_defs["filesystem"] = [
            MockToolDef("read_file", "Read"),
            MockToolDef("write_file", "Write"),
        ]
        manager.load_server("filesystem")

        assert registry.has("mcp_filesystem_read_file")
        assert manager.is_loaded("filesystem")

        manager.unload_all()

        assert not registry.has("mcp_filesystem_read_file")
        assert not registry.has("mcp_filesystem_write_file")
        assert not manager.is_loaded("filesystem")

    def test_unload_all_preserves_cache_and_session(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        mock_session = AsyncMock()
        manager._sessions["filesystem"] = mock_session
        manager._tool_defs["filesystem"] = [MockToolDef("read_file", "Read")]
        manager.load_server("filesystem")

        manager.unload_all()

        assert "filesystem" in manager._sessions
        assert "filesystem" in manager._tool_defs
        assert len(manager._tool_defs["filesystem"]) == 1

    def test_reload_after_unload_uses_cache(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        mock_session = AsyncMock()
        manager._sessions["filesystem"] = mock_session
        manager._tool_defs["filesystem"] = [MockToolDef("read_file", "Read")]

        manager.load_server("filesystem")
        assert registry.has("mcp_filesystem_read_file")

        manager.unload_all()
        assert not registry.has("mcp_filesystem_read_file")

        result = manager.load_server("filesystem")
        assert "Loaded 1 tools" in result
        assert registry.has("mcp_filesystem_read_file")

    def test_unload_all_handles_missing_tools(self, registry, mcp_configs):
        """unload_all should complete even if some tools were already unregistered."""
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        manager._sessions["filesystem"] = AsyncMock()
        manager._tool_defs["filesystem"] = [
            MockToolDef("read_file", "Read"),
            MockToolDef("write_file", "Write"),
        ]
        manager.load_server("filesystem")

        # Unregister one tool manually to simulate partial state
        registry.unregister("mcp_filesystem_read_file")

        # unload_all should still complete and clear _loaded_servers
        manager.unload_all()

        assert not manager.is_loaded("filesystem")
        assert not registry.has("mcp_filesystem_write_file")

    @pytest.mark.asyncio
    async def test_close_clears_everything(self, registry, mcp_configs):
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        manager._introspecting = True
        manager._loaded_servers.add("filesystem")
        manager._tool_defs["filesystem"] = [MockToolDef("tool1", "Tool")]

        await manager.close()

        assert not manager._introspected
        assert not manager._introspecting
        assert len(manager._loaded_servers) == 0
        assert len(manager._sessions) == 0
        assert len(manager._tool_defs) == 0


class TestLoadMCPTools:
    """Tests for LoadMCPTools meta-tool."""

    @pytest.fixture
    def manager(self, registry, mcp_configs):
        mgr = MCPManager(mcp_configs, registry)
        mgr._introspected = True
        mgr._tool_defs = {
            "filesystem": [MockToolDef("read_file", "Read")],
            "github": [MockToolDef("create_issue", "Create issue")],
        }
        return mgr

    def test_tool_name(self, manager):
        tool = LoadMCPTools(manager)
        assert tool.name == "load_mcp_tools"

    def test_tool_description_lists_servers(self, manager):
        tool = LoadMCPTools(manager)
        desc = tool.description
        assert "filesystem" in desc
        assert "github" in desc

    def test_tool_description_no_servers(self, registry):
        manager = MCPManager({}, registry)
        tool = LoadMCPTools(manager)
        assert "none configured" in tool.description

    def test_tool_parameters_has_server_enum(self, manager):
        tool = LoadMCPTools(manager)
        params = tool.parameters
        assert params["type"] == "object"
        assert "server" in params["properties"]
        assert params["properties"]["server"]["type"] == "string"
        assert set(params["properties"]["server"]["enum"]) == {"filesystem", "github"}
        assert params["required"] == ["server"]

    @pytest.mark.asyncio
    async def test_execute_delegates_to_manager(self, manager):
        tool = LoadMCPTools(manager)
        manager._sessions["filesystem"] = AsyncMock()

        result = await tool.execute(server="filesystem")

        assert "Loaded" in result


class TestMCPManagerIntegration:
    """Integration tests for MCPManager with AgentLoop patterns."""

    def test_new_session_flow(self, registry, mcp_configs):
        """Simulate /new behavior: unload tools but keep cache."""
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        manager._sessions["filesystem"] = AsyncMock()
        manager._tool_defs["filesystem"] = [
            MockToolDef("read_file", "Read"),
            MockToolDef("write_file", "Write"),
        ]

        # Initial load
        manager.load_server("filesystem")
        assert registry.has("mcp_filesystem_read_file")
        assert manager.is_loaded("filesystem")

        # Simulate /new
        manager.unload_all()

        # Tools removed from registry
        assert not registry.has("mcp_filesystem_read_file")
        assert not manager.is_loaded("filesystem")

        # But cache preserved
        assert "filesystem" in manager._tool_defs
        assert len(manager._tool_defs["filesystem"]) == 2

        # Can reload without MCP call
        result = manager.load_server("filesystem")
        assert "Loaded 2 tools" in result
        assert registry.has("mcp_filesystem_read_file")

    def test_multiple_servers_load_and_unload(self, registry, mcp_configs):
        """Test loading and unloading multiple MCP servers."""
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        manager._sessions["filesystem"] = AsyncMock()
        manager._sessions["github"] = AsyncMock()
        manager._tool_defs["filesystem"] = [MockToolDef("read_file", "Read")]
        manager._tool_defs["github"] = [MockToolDef("create_issue", "Create")]

        # Load both
        manager.load_server("filesystem")
        manager.load_server("github")

        assert registry.has("mcp_filesystem_read_file")
        assert registry.has("mcp_github_create_issue")
        assert manager.is_loaded("filesystem")
        assert manager.is_loaded("github")

        # Unload all
        manager.unload_all()

        assert not registry.has("mcp_filesystem_read_file")
        assert not registry.has("mcp_github_create_issue")
        assert not manager.is_loaded("filesystem")
        assert not manager.is_loaded("github")

        # Cache still intact
        assert "filesystem" in manager._tool_defs
        assert "github" in manager._tool_defs

    def test_partial_reload_after_unload(self, registry, mcp_configs):
        """Test reloading only one server after unload_all."""
        manager = MCPManager(mcp_configs, registry)
        manager._introspected = True
        manager._sessions["filesystem"] = AsyncMock()
        manager._sessions["github"] = AsyncMock()
        manager._tool_defs["filesystem"] = [MockToolDef("read_file", "Read")]
        manager._tool_defs["github"] = [MockToolDef("create_issue", "Create")]

        manager.load_server("filesystem")
        manager.load_server("github")
        manager.unload_all()

        # Reload only filesystem
        manager.load_server("filesystem")

        assert registry.has("mcp_filesystem_read_file")
        assert not registry.has("mcp_github_create_issue")
        assert manager.is_loaded("filesystem")
        assert not manager.is_loaded("github")

    def test_load_mcp_tools_tool_updates_description(self, registry, mcp_configs):
        """LoadMCPTools tool should reflect current server state."""
        manager = MCPManager(mcp_configs, registry)
        manager._tool_defs = {
            "filesystem": [MockToolDef("read_file", "Read")],
        }
        tool = LoadMCPTools(manager)

        desc = tool.description
        assert "filesystem" in desc

        params = tool.parameters
        assert "filesystem" in params["properties"]["server"]["enum"]


class TestContextBuilderWithMCP:
    """Tests for ContextBuilder integration with MCPManager."""

    def test_system_prompt_includes_mcp_summary(self, tmp_path):
        """System prompt should include MCP servers section when manager has tools."""
        registry = ToolRegistry()
        configs = {"filesystem": MockMCPServerConfig(command="mcp-fs")}
        manager = MCPManager(configs, registry)
        manager._tool_defs["filesystem"] = [MockToolDef("read_file", "Read a file")]

        builder = ContextBuilder(tmp_path, manager)
        prompt = builder.build_system_prompt()

        assert "# MCP Servers" in prompt
        assert "<mcp_servers>" in prompt
        assert "<name>filesystem</name>" in prompt
        assert 'loaded="false"' in prompt

    def test_system_prompt_shows_loaded_status(self, tmp_path):
        """System prompt should show loaded="true" for loaded servers."""
        registry = ToolRegistry()
        configs = {"filesystem": MockMCPServerConfig(command="mcp-fs")}
        manager = MCPManager(configs, registry)
        manager._introspected = True
        manager._tool_defs["filesystem"] = [MockToolDef("read_file", "Read")]
        manager._sessions["filesystem"] = AsyncMock()
        manager.load_server("filesystem")

        builder = ContextBuilder(tmp_path, manager)
        prompt = builder.build_system_prompt()

        assert 'loaded="true"' in prompt

    def test_system_prompt_no_mcp_section_when_empty(self, tmp_path):
        """System prompt should not have MCP section when no servers configured."""
        registry = ToolRegistry()
        manager = MCPManager({}, registry)

        builder = ContextBuilder(tmp_path, manager)
        prompt = builder.build_system_prompt()

        assert "# MCP Servers" not in prompt
        assert "<mcp_servers>" not in prompt

    def test_system_prompt_no_mcp_section_when_no_manager(self, tmp_path):
        """System prompt should not have MCP section when manager is None."""
        builder = ContextBuilder(tmp_path, None)
        prompt = builder.build_system_prompt()

        assert "# MCP Servers" not in prompt

    def test_mcp_summary_after_unload_shows_unloaded(self, tmp_path):
        """After unload_all, summary should show loaded="false"."""
        registry = ToolRegistry()
        configs = {"filesystem": MockMCPServerConfig(command="mcp-fs")}
        manager = MCPManager(configs, registry)
        manager._introspected = True
        manager._tool_defs["filesystem"] = [MockToolDef("read_file", "Read")]
        manager._sessions["filesystem"] = AsyncMock()
        manager.load_server("filesystem")

        builder = ContextBuilder(tmp_path, manager)
        prompt1 = builder.build_system_prompt()
        assert 'loaded="true"' in prompt1

        manager.unload_all()
        prompt2 = builder.build_system_prompt()
        assert 'loaded="false"' in prompt2

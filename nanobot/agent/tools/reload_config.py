"""Reload config tool — re-reads config.json and applies live changes to the agent."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


class ReloadConfigTool(Tool):
    """Tool that reloads config.json and applies any changed values to the running agent."""

    def __init__(self, agent: "AgentLoop") -> None:
        self._agent = agent

    @property
    def name(self) -> str:
        return "reload_config"

    @property
    def description(self) -> str:
        return (
            "Reload the config.json file from disk and apply any changes to the running agent. "
            "Use this after manually editing config.json to pick up new model settings, "
            "temperature, token limits, API keys, or other configuration without restarting."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Reload config.json and update live agent settings."""
        from nanobot.config.loader import load_config

        try:
            new_config = load_config()
        except Exception as e:
            return f"Failed to reload config: {e}"

        agent = self._agent
        changes: list[str] = []

        defaults = new_config.agents.defaults

        new_model = defaults.model or agent.provider.get_default_model()
        if new_model != agent.model:
            changes.append(f"model: {agent.model!r} → {new_model!r}")
            agent.model = new_model

        if defaults.temperature != agent.temperature:
            changes.append(f"temperature: {agent.temperature} → {defaults.temperature}")
            agent.temperature = defaults.temperature

        if defaults.max_tokens != agent.max_tokens:
            changes.append(f"max_tokens: {agent.max_tokens} → {defaults.max_tokens}")
            agent.max_tokens = defaults.max_tokens

        if defaults.max_tool_iterations != agent.max_iterations:
            changes.append(
                f"max_iterations: {agent.max_iterations} → {defaults.max_tool_iterations}"
            )
            agent.max_iterations = defaults.max_tool_iterations

        if defaults.memory_window != agent.memory_window:
            changes.append(f"memory_window: {agent.memory_window} → {defaults.memory_window}")
            agent.memory_window = defaults.memory_window

        new_reasoning = defaults.reasoning_effort
        if new_reasoning != agent.reasoning_effort:
            changes.append(f"reasoning_effort: {agent.reasoning_effort!r} → {new_reasoning!r}")
            agent.reasoning_effort = new_reasoning

        new_brave_key = new_config.tools.web.search.api_key or None
        if new_brave_key != agent.brave_api_key:
            changes.append(
                f"brave_api_key: {'set' if agent.brave_api_key else 'unset'} → "
                f"{'set' if new_brave_key else 'unset'}"
            )
            agent.brave_api_key = new_brave_key
            if web_search := agent.tools.get("web_search"):
                web_search.api_key = new_brave_key

        new_proxy = new_config.tools.web.proxy or None
        if new_proxy != agent.web_proxy:
            changes.append(f"web_proxy: {agent.web_proxy!r} → {new_proxy!r}")
            agent.web_proxy = new_proxy
            if web_search := agent.tools.get("web_search"):
                web_search.proxy = new_proxy
            if web_fetch := agent.tools.get("web_fetch"):
                web_fetch.proxy = new_proxy

        new_exec = new_config.tools.exec
        if (
            new_exec.timeout != agent.exec_config.timeout
            or new_exec.path_append != agent.exec_config.path_append
        ):
            changes.append(
                f"exec: timeout {agent.exec_config.timeout} → {new_exec.timeout}, "
                f"path_append {agent.exec_config.path_append!r} → {new_exec.path_append!r}"
            )
            agent.exec_config = new_exec
            if exec_tool := agent.tools.get("exec"):
                exec_tool.timeout = new_exec.timeout
                if new_exec.path_append:
                    exec_tool.path_append = new_exec.path_append

        if agent._mcp_manager:
            old_mcp = set(agent._mcp_manager.get_available_servers())
            new_mcp_configs = new_config.tools.mcp_servers
            new_mcp = set(new_mcp_configs.keys())

            if old_mcp != new_mcp:
                for tool_name in list(agent.tools._tools.keys()):
                    if tool_name.startswith("mcp_"):
                        agent.tools.unregister(tool_name)
                await agent._mcp_manager.close()
                changes.append(
                    f"mcp_servers: {len(old_mcp)} server(s) → {len(new_mcp)} server(s), "
                    "will reconnect on next message"
                )

        if changes:
            summary = "\n".join(f"  - {c}" for c in changes)
            return f"Config reloaded. Applied {len(changes)} change(s):\n{summary}"
        else:
            return "Config reloaded. No changes detected."

# squish/agent — Agentic tool dispatch layer (Wave 72)
from squish.agent.tool_registry import ToolRegistry, ToolDefinition, ToolResult
from squish.agent.builtin_tools import register_builtin_tools

__all__ = ["ToolRegistry", "ToolDefinition", "ToolResult", "register_builtin_tools"]

"""squish/agent/tool_registry.py — Tool Registry for Agentic Execution.

Wave 72: centralised registry for tool definitions, JSON Schema validation, and
dispatch. Supports built-in tools, user-registered tools, and MCP-discovered tools.

Usage::

    from squish.agent.tool_registry import ToolRegistry, ToolDefinition, ToolResult

    registry = ToolRegistry()

    @registry.tool(
        description="Read a file from disk and return its contents.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "start_line": {"type": "integer", "description": "First line (1-based)"},
                "end_line":   {"type": "integer", "description": "Last line inclusive"},
            },
            "required": ["path"],
        }
    )
    def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
        ...

    # Validate and call:
    result = registry.call("read_file", {"path": "/etc/hosts"})
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


__all__ = [
    "ToolDefinition",
    "ToolResult",
    "ToolRegistry",
    "ToolCallError",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """A registered tool with its JSON Schema and implementation.

    Attributes:
        name: Tool name (snake_case, globally unique within a registry).
        description: Human-readable description passed in the system prompt.
        parameters: JSON Schema ``object`` with ``properties`` + optional
            ``required`` list.
        fn: The Python callable that implements the tool.
        source: Where the tool came from — ``"builtin"``, ``"user"``, or
            ``"mcp:<server_id>"``.
    """

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    source: str = "user"

    def to_openai_schema(self) -> dict:
        """Return the OpenAI ``tools`` array element for this definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """The outcome of a single tool call.

    Attributes:
        tool_name: Name of the called tool.
        call_id: Unique identifier for this call (used for threading in
            multi-tool responses).
        output: Serialisable output value (string, dict, or list).
        error: Error message if the call failed; ``None`` on success.
        elapsed_ms: Wall-clock time for the call in milliseconds.
    """

    tool_name: str
    call_id: str
    output: Any
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_message(self) -> dict:
        """Return an OpenAI ``tool`` role message for injection back into context."""
        content = (
            self.output if isinstance(self.output, str)
            else json.dumps(self.output, ensure_ascii=False)
        )
        if self.error:
            content = f"[ERROR] {self.error}"
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": content,
        }


class ToolCallError(Exception):
    """Raised when a tool call fails validation or execution."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for agent tools.

    Provides registration via decorator or explicit ``register()``, JSON Schema
    validation of call arguments, and synchronous dispatch via ``call()``.

    The registry is intentionally simple and synchronous (async wrappers belong
    in :mod:`squish.serving.agent_executor`).
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, defn: ToolDefinition) -> None:
        """Register a :class:`ToolDefinition`."""
        if defn.name in self._tools:
            raise ValueError(f"Tool {defn.name!r} is already registered")
        self._tools[defn.name] = defn

    def tool(
        self,
        name: Optional[str] = None,
        *,
        description: str = "",
        parameters: Optional[dict] = None,
        source: str = "user",
    ) -> Callable:
        """Decorator for registering a function as a tool.

        Args:
            name: Override the tool name (defaults to the function name).
            description: Human-readable tool description.
            parameters: JSON Schema for the ``arguments`` object.
            source: Provenance tag.

        Returns:
            The unmodified decorated function (no wrapping).
        """
        def decorator(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            params = parameters or {
                "type": "object",
                "properties": {},
                "required": [],
            }
            self.register(ToolDefinition(
                name=tool_name,
                description=description or (fn.__doc__ or "").strip(),
                parameters=params,
                fn=fn,
                source=source,
            ))
            return fn
        return decorator

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op if it does not exist."""
        self._tools.pop(name, None)

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Return the :class:`ToolDefinition` or ``None``."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def to_openai_schemas(self) -> list[dict]:
        """Return the full ``tools`` array for an OpenAI chat completion request."""
        return [d.to_openai_schema() for d in self._tools.values()]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_call(self, name: str, arguments: dict) -> None:
        """Validate *arguments* against the tool's JSON Schema.

        Performs a lightweight structural check:
        - All ``required`` properties must be present.
        - Property values must match the declared ``type`` where specified.

        Args:
            name: Tool name.
            arguments: Parsed argument dict.

        Raises:
            ToolCallError: If the tool is unknown or arguments are invalid.
        """
        defn = self._tools.get(name)
        if defn is None:
            raise ToolCallError(f"Unknown tool: {name!r}")

        params = defn.parameters
        required = params.get("required", [])
        properties = params.get("properties", {})

        missing = [r for r in required if r not in arguments]
        if missing:
            raise ToolCallError(
                f"Tool {name!r} missing required arguments: {missing}"
            )

        _TYPE_MAP = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for arg_name, value in arguments.items():
            schema = properties.get(arg_name, {})
            expected_type = schema.get("type")
            if expected_type and expected_type in _TYPE_MAP:
                if not isinstance(value, _TYPE_MAP[expected_type]):
                    raise ToolCallError(
                        f"Tool {name!r} argument {arg_name!r}: expected "
                        f"{expected_type}, got {type(value).__name__}"
                    )
            # Enforce enum values if present
            enum = schema.get("enum")
            if enum is not None and value not in enum:
                raise ToolCallError(
                    f"Tool {name!r} argument {arg_name!r}: value {value!r} "
                    f"not in allowed enum {enum}"
                )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def call(
        self,
        name: str,
        arguments: dict,
        *,
        call_id: Optional[str] = None,
        validate: bool = True,
    ) -> ToolResult:
        """Validate and execute a tool call.

        Args:
            name: Tool name.
            arguments: Keyword arguments dict.
            call_id: Optional identifier (defaults to ``f"{name}_{timestamp}"``).
            validate: If ``True`` (default), validate arguments before calling.

        Returns:
            :class:`ToolResult` — always returned; check ``result.ok``.
        """
        import uuid
        cid = call_id or f"{name}_{uuid.uuid4().hex[:8]}"

        if validate:
            try:
                self.validate_call(name, arguments)
            except ToolCallError as exc:
                return ToolResult(
                    tool_name=name,
                    call_id=cid,
                    output=None,
                    error=str(exc),
                )

        defn = self._tools.get(name)
        if defn is None:
            return ToolResult(
                tool_name=name,
                call_id=cid,
                output=None,
                error=f"Unknown tool: {name!r}",
            )

        t0 = time.perf_counter()
        try:
            output = defn.fn(**arguments)
            elapsed = (time.perf_counter() - t0) * 1000
            return ToolResult(
                tool_name=name,
                call_id=cid,
                output=output,
                elapsed_ms=elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - t0) * 1000
            return ToolResult(
                tool_name=name,
                call_id=cid,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=elapsed,
            )

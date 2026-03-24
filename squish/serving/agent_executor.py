"""squish/serving/agent_executor.py — Multi-Step Agentic Execution Loop.

Wave 72: orchestrates the call-tool-resubmit loop between the inference engine
and the tool registry. Designed to work with any callable that accepts
OpenAI-compatible ``messages + tools`` and returns a completion object.

Usage::

    from squish.serving.agent_executor import AgentConfig, AgentExecutor
    from squish.agent.tool_registry import ToolRegistry
    from squish.agent.builtin_tools import register_builtin_tools

    registry = ToolRegistry()
    register_builtin_tools(registry)

    executor = AgentExecutor(
        config=AgentConfig(max_steps=10),
        registry=registry,
        infer_fn=my_async_infer_fn,   # async (messages, tools) -> completion
    )

    async for event in executor.run(messages):
        if event["type"] == "text_delta":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "tool_call_result":
            print(f"\\nTool {event['tool_name']} → {event['result']}")

Event types emitted::

    {"type": "text_delta",     "delta": str}
    {"type": "tool_call_start","call_id": str, "tool_name": str, "arguments": dict}
    {"type": "tool_call_result","call_id": str, "tool_name": str, "result": str,
                                "error": str | None, "elapsed_ms": float}
    {"type": "step_complete",  "step": int, "total_steps": int}
    {"type": "done",           "total_steps": int, "total_tool_calls": int}
    {"type": "error",          "message": str}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "AgentConfig",
    "AgentStep",
    "AgentSession",
    "AgentExecutor",
]


# ---------------------------------------------------------------------------
# Configuration + state types
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for an :class:`AgentExecutor` run.

    Attributes:
        max_steps: Hard limit on tool-call rounds to prevent infinite loops.
        tool_timeout: Per-tool call wall-clock limit in seconds.
        stream: If ``True``, the executor yields text deltas as they arrive
            (requires *infer_fn* to support streaming).
    """

    max_steps: int = 10
    tool_timeout: int = 30
    stream: bool = True


@dataclass
class AgentStep:
    """Record of a single tool-call within one agent run.

    Attributes:
        step_num: Step index (1-based).
        tool_name: Name of the called tool.
        arguments: Raw argument dict sent to the tool.
        result: String output returned by the tool.
        error: Error message on failure; ``None`` on success.
        elapsed_ms: Wall-clock time for the tool call.
    """

    step_num: int
    tool_name: str
    arguments: dict
    result: Optional[str]
    error: Optional[str]
    elapsed_ms: float = 0.0


@dataclass
class AgentSession:
    """Mutable state accumulated across one :meth:`AgentExecutor.run` call.

    Attributes:
        session_id: Unique identifier for this run.
        messages: Evolving message list (system + history + tool results).
        steps: Ordered list of tool-call records.
        total_input_tokens: Approximate cumulative input tokens.
        total_output_tokens: Approximate cumulative output tokens.
        status: ``"running"`` | ``"done"`` | ``"error"`` | ``"max_steps"``.
    """

    session_id: str
    messages: list[dict] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    status: str = "running"


# ---------------------------------------------------------------------------
# Agent executor
# ---------------------------------------------------------------------------

class AgentExecutor:
    """Multi-step agentic execution loop.

    Calls the model, parses any tool calls, dispatches them via *registry*,
    injects ``tool`` role results, and repeats until the model responds with
    plain text (no tool calls) or *max_steps* is reached.

    Args:
        config: Behavioural settings (step limit, timeout, streaming).
        registry: Populated :class:`~squish.agent.tool_registry.ToolRegistry`.
        infer_fn: Async callable with signature
            ``async (messages: list[dict], tools: list[dict] | None, *, stream: bool)
            -> AsyncIterator[dict] | dict``.
            When ``stream=True``, must yield dicts with at least a ``"delta"``
            key; when ``stream=False``, must return a completion dict.
    """

    def __init__(
        self,
        config: AgentConfig,
        registry: Any,  # ToolRegistry — avoid circular import at type level
        infer_fn: Callable,
    ) -> None:
        self.config = config
        self.registry = registry
        self.infer_fn = infer_fn

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        *,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Run the agentic loop and yield events.

        Args:
            messages: Initial conversation history (user + optional system).
            tools: OpenAI ``tools`` array; defaults to all registered tools.
            session_id: Optional tag for logging.

        Yields:
            Event dicts (see module docstring for types).
        """
        import uuid

        sid = session_id or uuid.uuid4().hex[:12]
        session = AgentSession(session_id=sid, messages=list(messages))
        effective_tools = tools if tools is not None else self.registry.to_openai_schemas()

        step = 0
        total_tool_calls = 0

        while step < self.config.max_steps:
            # ---- call the model ----------------------------------------
            try:
                if self.config.stream:
                    text, tool_calls = yield from self._stream_turn(session, effective_tools)
                else:
                    text, tool_calls = await self._non_stream_turn(session, effective_tools)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent infer error on step %d: %s", step, exc)
                yield {"type": "error", "message": str(exc)}
                session.status = "error"
                return

            # ---- plain text response → done ----------------------------
            if not tool_calls:
                if text:
                    if not self.config.stream:
                        # Emit the full text as one delta for uniformity
                        yield {"type": "text_delta", "delta": text}
                session.status = "done"
                yield {
                    "type": "done",
                    "total_steps": step,
                    "total_tool_calls": total_tool_calls,
                }
                return

            # ---- dispatch tool calls -----------------------------------
            tool_result_messages: list[dict] = []
            for tc in tool_calls:
                call_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                fn_data = tc.get("function", tc)
                tool_name = fn_data.get("name", "")
                raw_args = fn_data.get("arguments", "{}")

                # Parse arguments
                if isinstance(raw_args, str):
                    try:
                        arguments = json.loads(raw_args)
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = raw_args

                yield {
                    "type": "tool_call_start",
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                }

                result = self.registry.call(
                    tool_name, arguments, call_id=call_id
                )

                step_record = AgentStep(
                    step_num=step + 1,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result.output if result.ok else None,
                    error=result.error,
                    elapsed_ms=result.elapsed_ms,
                )
                session.steps.append(step_record)
                total_tool_calls += 1

                result_text = (
                    str(result.output) if result.ok else f"[ERROR] {result.error}"
                )

                yield {
                    "type": "tool_call_result",
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "result": result_text,
                    "error": result.error,
                    "elapsed_ms": result.elapsed_ms,
                }

                tool_result_messages.append(result.to_message())

            # ---- inject tool results back into context -----------------
            # Append the assistant turn that triggered the calls
            assistant_msg: dict = {"role": "assistant", "content": text or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", tc).get("name", ""),
                            "arguments": json.dumps(
                                tc.get("function", tc).get("arguments", {})
                                if isinstance(tc.get("function", tc).get("arguments", {}), dict)
                                else tc.get("function", tc).get("arguments", {})
                            ),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ]
            session.messages.append(assistant_msg)
            session.messages.extend(tool_result_messages)

            step += 1
            yield {"type": "step_complete", "step": step, "total_steps": step}

        # max_steps exceeded
        logger.warning("Agent session %s hit max_steps=%d", sid, self.config.max_steps)
        session.status = "max_steps"
        yield {
            "type": "error",
            "message": (
                f"Agent hit the maximum step limit of {self.config.max_steps}. "
                "Partial results may be available."
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _non_stream_turn(
        self,
        session: AgentSession,
        tools: list[dict],
    ) -> tuple[str, list[dict]]:
        """Single non-streaming inference call.

        Returns:
            ``(text, tool_calls)`` where *tool_calls* may be empty.
        """
        completion = await self.infer_fn(
            session.messages, tools, stream=False
        )
        choice = completion.get("choices", [{}])[0]
        msg = choice.get("message", {})
        text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        return text, tool_calls

    async def _stream_turn(
        self,
        session: AgentSession,
        tools: list[dict],
    ) -> tuple[str, list[dict]]:
        """Streaming inference call — collects full text and tool calls.

        Text deltas are yielded via the generator protocol; the caller must
        forward them.  Because Python generators cannot both ``return`` values
        and ``yield``, this method uses a mutable list idiom:

        .. note::
            The caller does ``text, tc = yield from self._stream_turn(...)``
            which works because the generator's ``StopIteration`` value IS the
            return value.
        """
        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}

        async for chunk in await self.infer_fn(
            session.messages, tools, stream=True
        ):
            delta = chunk.get("delta", {})
            if isinstance(delta, str):
                text_parts.append(delta)
                yield {"type": "text_delta", "delta": delta}
                continue

            # OpenAI-style chunk structure
            content = delta.get("content")
            if content:
                text_parts.append(content)
                yield {"type": "text_delta", "delta": content}

            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": tc_delta.get("id", ""),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                tc = tool_calls_acc[idx]
                fn_d = tc_delta.get("function", {})
                if fn_d.get("name"):
                    tc["function"]["name"] += fn_d["name"]
                if fn_d.get("arguments"):
                    tc["function"]["arguments"] += fn_d["arguments"]
                if tc_delta.get("id"):
                    tc["id"] = tc_delta["id"]

        text = "".join(text_parts)
        tool_calls: list[dict] = []
        for tc in sorted(tool_calls_acc.values(), key=lambda x: list(tool_calls_acc.values()).index(x)):
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed_args = {}
            tool_calls.append({
                "id": tc.get("id") or f"call_{len(tool_calls)}",
                "function": {"name": fn.get("name", ""), "arguments": parsed_args},
            })

        return text, tool_calls

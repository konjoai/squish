"""
squish/disc_router.py

DISC — Dynamic Instruction-driven Sub-task Composition Router.

Inspired by:
  "DISC: Dynamic Instruction-driven Sub-task Composition for Long-text
   Tasks" — (2025)

Problem
-------
Long-form requests (e.g. "Summarize this 10k-word document and compare with
the other one, then write a report") are too complex to handle in a single
monolithic prompting pass.  Models hallucinate more, lose context, and
under-perform on structure.

DISC Solution
-------------
Decompose the user request into a DAG of sub-tasks using a lightweight
"planner" LLM call.  Each sub-task has:
  * type tag (summarize | compare | retrieve | generate | qa | agg)
  * input specification (which prior outputs or raw context segments)
  * output variable name
  * optional dependencies on other sub-tasks

The router then dispatches sub-tasks to specialised handlers (or the same
LLM with task-type system prompts) and aggregates results.

This module provides:
  * ``TaskType`` — enum of supported sub-task types
  * ``SubTask`` — data class describing a single decomposed task
  * ``DISCPlan`` — ordered list of SubTask objects with dependency ordering
  * ``DISCRouter`` — executes a plan against a callable LLM function

Integration::

    from squish.disc_router import DISCRouter, DISCRouterConfig, SubTask, TaskType

    router = DISCRouter(
        llm_fn=my_model_fn,
        config=DISCRouterConfig(max_subtasks=12),
    )
    result = router.execute(user_request, context=document_text)
"""

from __future__ import annotations

import enum
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TaskType",
    "SubTask",
    "DISCPlan",
    "DISCRouterConfig",
    "DISCRouter",
]


# ---------------------------------------------------------------------------
# Task taxonomy
# ---------------------------------------------------------------------------

class TaskType(str, enum.Enum):
    """Canonical sub-task types recognised by DISC."""
    SUMMARIZE = "summarize"
    COMPARE   = "compare"
    RETRIEVE  = "retrieve"
    GENERATE  = "generate"
    QA        = "qa"
    CODE      = "code"
    AGGREGATE = "aggregate"
    UNKNOWN   = "unknown"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubTask:
    """
    A single decomposed sub-task in a DISC plan.

    Parameters
    ----------
    task_id      : str — unique identifier (e.g. "t1")
    task_type    : TaskType
    prompt       : str — instruction for this sub-task (already templated)
    inputs       : list of variable names referencing prior sub-task outputs
    output_var   : str — name of the output variable produced
    depends_on   : list of task_ids that must complete before this
    context_key  : str or None — key into the context dict, if this task
                   needs a raw context segment
    metadata     : dict — arbitrary extra info
    """
    task_id:     str
    task_type:   TaskType
    prompt:      str
    inputs:      list[str] = field(default_factory=list)
    output_var:  str       = ""
    depends_on:  list[str] = field(default_factory=list)
    context_key: str | None = None
    metadata:    dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if not self.output_var:
            self.output_var = self.task_id + "_out"


@dataclass
class DISCPlan:
    """
    An ordered list of :class:`SubTask` objects with topological ordering.

    Attributes
    ----------
    tasks : list of SubTask (in execution order after topological sort)
    """
    tasks: list[SubTask] = field(default_factory=list)

    def add(self, task: SubTask) -> None:
        """Append a sub-task to the plan."""
        self.tasks.append(task)

    def topological_order(self) -> list[SubTask]:
        """
        Return tasks in topological execution order.

        Uses Kahn's algorithm on the dependency graph.

        Raises
        ------
        ValueError if there is a dependency cycle.
        """
        task_map: dict[str, SubTask]  = {t.task_id: t for t in self.tasks}
        in_deg:   dict[str, int]      = {t.task_id: 0 for t in self.tasks}
        for t in self.tasks:
            for dep in t.depends_on:
                if dep in in_deg:
                    in_deg[t.task_id] += 1

        queue  = [tid for tid, deg in in_deg.items() if deg == 0]
        result: list[SubTask] = []

        while queue:
            tid = queue.pop(0)
            result.append(task_map[tid])
            for candidate in self.tasks:
                if tid in candidate.depends_on:
                    in_deg[candidate.task_id] -= 1
                    if in_deg[candidate.task_id] == 0:
                        queue.append(candidate.task_id)

        if len(result) != len(self.tasks):
            raise ValueError("Dependency cycle detected in DISC plan")
        return result

    def __len__(self) -> int:
        return len(self.tasks)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DISCRouterConfig:
    """
    Configuration for the DISC router.

    Parameters
    ----------
    max_subtasks : int
        Maximum number of sub-tasks allowed in a plan.
    parallel_execution : bool
        If True, independent sub-tasks (no shared dependencies at the same
        topological level) can be dispatched concurrently.  Requires the
        caller's ``llm_fn`` to be thread-safe.
    task_prompt_templates : dict
        Optional mapping from ``TaskType`` value → system-prompt prefix.
        When missing, sensible defaults are used.
    """
    max_subtasks:          int  = 12
    parallel_execution:    bool = False
    task_prompt_templates: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_subtasks < 1:
            raise ValueError("max_subtasks must be ≥ 1")

    def get_system_prompt(self, task_type: TaskType) -> str:
        """Return the system prompt prefix for a given task type."""
        if task_type.value in self.task_prompt_templates:
            return self.task_prompt_templates[task_type.value]
        return _DEFAULT_SYSTEM_PROMPTS.get(task_type, "")


_DEFAULT_SYSTEM_PROMPTS: dict[TaskType, str] = {
    TaskType.SUMMARIZE: "You are a summarization assistant. Be concise.",
    TaskType.COMPARE:   "You are a comparison assistant. List differences clearly.",
    TaskType.RETRIEVE:  "You are a retrieval assistant. Find relevant excerpts.",
    TaskType.GENERATE:  "You are a creative writing assistant.",
    TaskType.QA:        "You are a question-answering assistant. Cite sources.",
    TaskType.CODE:      "You are a coding assistant. Write clean, correct code.",
    TaskType.AGGREGATE: "You are a synthesis assistant. Combine the provided outputs.",
    TaskType.UNKNOWN:   "",
}


# ---------------------------------------------------------------------------
# DISC Router
# ---------------------------------------------------------------------------

class DISCRouter:
    """
    Executes a DISC decomposition plan against a language model function.

    Parameters
    ----------
    llm_fn : callable(prompt: str, system: str, context: str) -> str
        The language model to call.  Must return a plain string.
    config : DISCRouterConfig
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str, str], str],
        config: DISCRouterConfig | None = None,
    ) -> None:
        self._llm    = llm_fn
        self._cfg    = config or DISCRouterConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(
        self,
        user_request: str,
        context:      str = "",
    ) -> DISCPlan:
        """
        Generate a DISC plan from a user request using the LLM.

        The planner prompt asks the model to output JSON-like sub-task specs.
        If the LLM returns unstructured text, a single-task fallback plan
        is returned.

        Parameters
        ----------
        user_request : str — the original user message
        context      : str — relevant document / context text

        Returns
        -------
        DISCPlan
        """
        planner_prompt = self._build_planner_prompt(user_request, context)
        raw            = self._llm(planner_prompt, "You are a task planner.", "")
        plan           = self._parse_plan(raw, user_request)
        if len(plan) > self._cfg.max_subtasks:
            plan.tasks = plan.tasks[: self._cfg.max_subtasks]
        return plan

    def execute(
        self,
        user_request: str,
        context:      str = "",
        plan:         DISCPlan | None = None,
    ) -> str:
        """
        Plan (if not provided) and execute a DISC decomposition.

        Parameters
        ----------
        user_request : str
        context      : str
        plan         : optional pre-built DISCPlan

        Returns
        -------
        str — the final aggregated answer
        """
        if plan is None:
            plan = self.plan(user_request, context)

        ordered        = plan.topological_order()
        variables: dict[str, str] = {}   # output_var -> result string

        for task in ordered:
            result = self._execute_task(task, variables, context)
            variables[task.output_var] = result

        # Return the last task's output (usually an AGGREGATE task)
        if ordered:
            return variables.get(ordered[-1].output_var, "")
        return ""

    def execute_plan(
        self,
        plan:     DISCPlan,
        context:  str = "",
    ) -> dict[str, str]:
        """
        Execute a pre-built plan and return all output variables.

        Returns
        -------
        dict mapping output_var name -> result string
        """
        ordered        = plan.topological_order()
        variables: dict[str, str] = {}
        for task in ordered:
            result = self._execute_task(task, variables, context)
            variables[task.output_var] = result
        return variables

    # ── Private ───────────────────────────────────────────────────────────────

    def _execute_task(
        self,
        task:      SubTask,
        variables: dict[str, str],
        context:   str,
    ) -> str:
        """Execute a single sub-task and return its string output."""
        # Assemble prompt: template + any required prior outputs
        system        = self._cfg.get_system_prompt(task.task_type)
        prior_outputs = "\n\n".join(
            f"[{var}]: {variables.get(var, '')}" for var in task.inputs
        )
        task_context = context if task.context_key is None else (
            variables.get(task.context_key, context)
        )
        full_prompt = task.prompt
        if prior_outputs:
            full_prompt = prior_outputs + "\n\n" + full_prompt

        return self._llm(full_prompt, system, task_context)

    @staticmethod
    def _build_planner_prompt(user_request: str, context: str) -> str:
        ctx_snippet = context[:500] if context else "(none)"
        return textwrap.dedent(f"""
            Decompose the following user request into a minimal list of sub-tasks.
            For each sub-task output a line: TASK|<id>|<type>|<depends_on>|<output_var>|<instruction>

            Types: summarize, compare, retrieve, generate, qa, code, aggregate
            depends_on: comma-separated task ids (or empty)

            User request:
            {user_request}

            Context snippet:
            {ctx_snippet}
        """).strip()

    @staticmethod
    def _parse_plan(raw: str, user_request: str) -> DISCPlan:
        """
        Parse LLM planner output into a DISCPlan.

        Expected line format:
            TASK|t1|summarize||summary_out|Summarize the document.

        Falls back to a single-task plan if no TASK lines are found.
        """
        plan  = DISCPlan()
        lines = raw.splitlines()
        found = False
        for line in lines:
            line = line.strip()
            if not line.upper().startswith("TASK|"):
                continue
            parts = line.split("|")
            if len(parts) < 6:
                continue
            task_id    = parts[1].strip()
            type_str   = parts[2].strip().lower()
            depends_on = [d.strip() for d in parts[3].split(",") if d.strip()]
            output_var = parts[4].strip() or (task_id + "_out")
            prompt     = "|".join(parts[5:]).strip()
            task_type  = TaskType(type_str) if type_str in TaskType._value2member_map_ else TaskType.UNKNOWN
            plan.add(SubTask(
                task_id    = task_id,
                task_type  = task_type,
                prompt     = prompt,
                depends_on = depends_on,
                output_var = output_var,
            ))
            found = True

        if not found:
            # Fallback: single GENERATE task
            plan.add(SubTask(
                task_id   = "t1",
                task_type = TaskType.GENERATE,
                prompt    = user_request,
                output_var = "final_out",
            ))
        return plan

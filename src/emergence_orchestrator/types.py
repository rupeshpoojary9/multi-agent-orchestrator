"""Shared data model for the orchestrator.

Everything the meta-agent produces or consumes is a plain dataclass here so the
planner, executor, verifier, synthesizer, and governance layer all speak the
same language and each piece stays independently testable.

Design note on success criteria: each Step carries BOTH a machine-checkable
predicate (`check`) and a natural-language `success_criteria` string. The
heuristic verifier evaluates `check` deterministically (so the system is fully
testable offline); the LLM verifier reads `success_criteria`. Keeping both means
swapping in a real model changes quality, not the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
@dataclass
class ToolSchema:
    """A typed tool contract an agent exposes (the 'integration contract')."""

    name: str
    description: str
    params: dict[str, str]  # param name -> type hint ("str", "int", "bool")
    irreversible: bool = False  # writes that a guardrail may gate behind approval


@dataclass
class AgentSpec:
    """Declarative description of an agent, persisted in the registry.

    `generated=True` marks a sub-agent the meta-agent synthesized at runtime.
    A generated agent is always `composes`d from existing primitive tools — it
    never invents raw capability, only new orchestration over vetted tools.
    """

    name: str
    kind: str  # "web" | "api" | "composite"
    description: str
    capabilities: list[str]  # tool names this agent can execute
    tool_schemas: list[ToolSchema] = field(default_factory=list)
    system_prompt: str = ""
    generated: bool = False
    composes: list[str] = field(default_factory=list)  # primitive tools it wraps

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    """One node in the task DAG."""

    id: str
    description: str
    capability: str  # the tool/capability this step needs
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    success_criteria: str = ""  # natural language, for the LLM verifier
    check: dict[str, Any] | None = None  # machine predicate, for the heuristic verifier
    agent: str | None = None  # resolved at routing time (may require synthesis)


@dataclass
class Plan:
    task: str
    steps: list[Step] = field(default_factory=list)

    def topo_order(self) -> list[Step]:
        """Return steps in a dependency-respecting order (raises on a cycle)."""
        by_id = {s.id: s for s in self.steps}
        state: dict[str, int] = {}  # 0=unseen, 1=visiting, 2=done
        order: list[Step] = []

        def visit(sid: str) -> None:
            st = state.get(sid, 0)
            if st == 2:
                return
            if st == 1:
                raise ValueError(f"plan has a dependency cycle at step {sid!r}")
            state[sid] = 1
            for dep in by_id[sid].depends_on:
                if dep not in by_id:
                    raise ValueError(f"step {sid!r} depends on unknown step {dep!r}")
                visit(dep)
            state[sid] = 2
            order.append(by_id[sid])

        for s in self.steps:
            visit(s.id)
        return order


# --------------------------------------------------------------------------- #
# Execution results
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """What an agent returns after executing a tool."""

    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    step_id: str
    capability: str
    agent: str
    ok: bool  # verified success (executed AND passed the verifier)
    detail: str
    ms: float = 0.0
    cost_usd: float = 0.0
    attempts: int = 1
    verifier_ok: bool | None = None
    verifier_reason: str = ""
    synthesized_agent: str | None = None  # set if a sub-agent was generated for this


@dataclass
class RunReport:
    task: str
    ok: bool
    steps: list[StepResult] = field(default_factory=list)
    total_ms: float = 0.0
    total_cost_usd: float = 0.0
    replans: int = 0
    synthesized_agents: list[str] = field(default_factory=list)
    verifier_catches: int = 0  # step failures the verifier caught before completion
    halted_reason: str | None = None  # set if a guardrail/budget stopped the run

    @property
    def attempts_total(self) -> int:
        return sum(s.attempts for s in self.steps)

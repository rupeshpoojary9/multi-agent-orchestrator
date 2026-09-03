"""Governance layer — audit trail, guardrails, and a cost/latency budget.

This is the 'Systems & Integration' safety envelope around unreliable agents:
  - AuditLog:   an append-only, structured record of every plan, tool call,
                verifier verdict, re-plan, synthesis, and guardrail decision.
  - Guardrails: an allow-list of tools, plus a human-approval gate on
                irreversible writes.
  - Budget:     hard caps on steps, re-plans, and cumulative cost; the run halts
                (rather than looping) when any is exceeded.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
@dataclass
class AuditEvent:
    ts: float
    kind: str  # plan | route | tool_call | verify | replan | synthesize | guardrail | halt
    detail: dict[str, Any]


class AuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, kind: str, **detail: Any) -> None:
        self.events.append(AuditEvent(ts=time.time(), kind=kind, detail=detail))

    def of_kind(self, kind: str) -> list[AuditEvent]:
        return [e for e in self.events if e.kind == kind]

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps({"ts": e.ts, "kind": e.kind, **e.detail}) for e in self.events
        )


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
class GuardrailError(Exception):
    pass


# An approval hook: given (capability, args) it returns True to allow an
# irreversible write. Default denies nothing in autonomous mode, but the
# orchestrator can pass a stricter one (e.g. a human prompt) for enterprise use.
ApprovalFn = Callable[[str, dict], bool]


@dataclass
class Guardrails:
    allowed_tools: set[str]
    irreversible_tools: set[str] = field(default_factory=set)
    require_approval: bool = False  # gate irreversible writes behind `approve`
    approve: ApprovalFn = lambda cap, args: True

    def check_tool(self, capability: str) -> None:
        if capability not in self.allowed_tools:
            raise GuardrailError(f"tool {capability!r} is not on the allow-list")

    def check_write(self, capability: str, args: dict) -> None:
        if not self.require_approval:
            return
        if capability in self.irreversible_tools and not self.approve(capability, args):
            raise GuardrailError(
                f"irreversible tool {capability!r} denied by approval gate"
            )


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    max_steps: int = 12
    max_replans: int = 3
    max_cost_usd: float = 0.50

    steps_used: int = 0
    replans_used: int = 0
    cost_used: float = 0.0

    def charge_step(self) -> None:
        self.steps_used += 1
        if self.steps_used > self.max_steps:
            raise BudgetExceeded(f"step budget {self.max_steps} exceeded")

    def charge_replan(self) -> None:
        self.replans_used += 1
        if self.replans_used > self.max_replans:
            raise BudgetExceeded(f"re-plan budget {self.max_replans} exceeded")

    def charge_cost(self, usd: float) -> None:
        self.cost_used += usd
        if self.cost_used > self.max_cost_usd:
            raise BudgetExceeded(f"cost budget ${self.max_cost_usd} exceeded")

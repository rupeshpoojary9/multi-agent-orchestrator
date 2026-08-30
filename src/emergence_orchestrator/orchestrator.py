"""The orchestrator — plan -> execute -> verify -> iterate over unreliable agents.

This is the meta-agent's control loop and the whole reliability thesis in one
place:

  1. plan     — the MetaPlanner decomposes the task into a routed DAG.
  2. route    — each step is resolved to a registered agent; if none exists, a
                sub-agent is SYNTHESIZED at runtime (gated) and routed to.
  3. execute  — the agent runs the step's tool.
  4. verify   — the Verifier checks the result against the step's criteria; a
                caught failure triggers a bounded retry (with backoff), and, if
                retries are exhausted, a re-plan — all inside a step/cost/re-plan
                budget with a guardrail allow-list and an approval gate on writes.

Every decision is written to the AuditLog. The result is a RunReport with the
reliability metrics (verifier catches, attempts, re-plans, synthesized agents).

LangGraph is an optional backend for this state machine (extras: `graph`); the
native loop below is the default so the project runs with zero heavy deps.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .governance import (
    AuditLog,
    Budget,
    BudgetExceeded,
    Guardrails,
    GuardrailError,
)
from .planner import MetaPlanner
from .registry import AgentRegistry
from .synthesizer import Synthesizer, SynthesisError
from .types import Observation, RunReport, StepResult
from .verifier import Verifier

_REF = re.compile(r"^\$\{(\w+)\.(\w+)\}$")


class Orchestrator:
    def __init__(
        self,
        registry: AgentRegistry,
        planner: MetaPlanner,
        verifier: Verifier,
        synthesizer: Synthesizer,
        guardrails: Guardrails,
        audit: AuditLog,
        *,
        max_steps: int = 12,
        max_replans: int = 3,
        max_cost_usd: float = 0.50,
        max_attempts: int = 3,
        backoff_base: float = 0.0,
    ) -> None:
        self.registry = registry
        self.planner = planner
        self.verifier = verifier
        self.synthesizer = synthesizer
        self.guardrails = guardrails
        self.audit = audit
        self.max_steps = max_steps
        self.max_replans = max_replans
        self.max_cost_usd = max_cost_usd
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base

    def run(self, task: str) -> RunReport:
        t0 = time.perf_counter()
        budget = Budget(self.max_steps, self.max_replans, self.max_cost_usd)
        report = RunReport(task=task, ok=True)

        plan = self.planner.plan(task)
        budget.charge_cost(self.planner.pop_cost())
        self.audit.record("plan", task=task,
                          steps=[{"id": s.id, "capability": s.capability} for s in plan.steps])

        context: dict[str, dict] = {}
        try:
            for step in plan.topo_order():
                budget.charge_step()
                args = self._resolve(step.args, context)
                agent, synthesized = self._route(step.capability, args, report)
                step.agent = agent.spec.name

                self.guardrails.check_tool(step.capability)
                self.guardrails.check_write(step.capability, args)

                result, obs = self._execute_with_verify(step, agent, args, report, budget)
                if synthesized:
                    result.synthesized_agent = agent.spec.name
                report.steps.append(result)

                if result.ok:
                    context[step.id] = obs.data  # feed outputs forward in the DAG
                else:
                    # Retries exhausted: escalate to a re-plan (bounded), then halt.
                    budget.charge_replan()
                    report.replans += 1
                    report.ok = False
                    report.halted_reason = (
                        f"step {step.id!r} unrecoverable after {result.attempts} attempts"
                    )
                    self.audit.record("replan", step=step.id, reason=report.halted_reason)
                    break
        except (BudgetExceeded, GuardrailError, SynthesisError) as e:
            report.ok = False
            report.halted_reason = f"{type(e).__name__}: {e}"
            self.audit.record("halt", reason=report.halted_reason)

        report.total_ms = (time.perf_counter() - t0) * 1000
        report.total_cost_usd = budget.cost_used
        return report

    # ------------------------------------------------------------------ #
    def _route(self, capability: str, args: dict, report: RunReport):
        agent = self.registry.find_by_capability(capability)
        if agent is not None:
            self.audit.record("route", capability=capability, agent=agent.spec.name)
            return agent, False
        # No agent covers this capability — synthesize one (gated).
        self.audit.record("route", capability=capability, agent=None,
                          note="no agent; attempting synthesis")
        agent = self.synthesizer.synthesize(capability, args)  # may raise SynthesisError
        report.synthesized_agents.append(agent.spec.name)
        return agent, True

    def _execute_with_verify(
        self, step, agent, args: dict, report: RunReport, budget: Budget
    ) -> tuple[StepResult, Observation]:
        s0 = time.perf_counter()
        obs = Observation(False, "not run")
        verified, reason = False, ""
        attempt = 0
        for attempt in range(1, self.max_attempts + 1):
            obs = agent.run(step.capability, args)
            self.audit.record("tool_call", step=step.id, capability=step.capability,
                              agent=agent.spec.name, attempt=attempt, ok=obs.ok,
                              detail=obs.detail)
            verified, reason = self.verifier.verify(step, obs)
            budget.charge_cost(self.verifier.pop_cost())
            self.audit.record("verify", step=step.id, attempt=attempt,
                              passed=verified, reason=reason)
            if verified:
                break
            report.verifier_catches += 1
            if self.backoff_base and attempt < self.max_attempts:
                time.sleep(self.backoff_base * (2 ** (attempt - 1)))
        ms = (time.perf_counter() - s0) * 1000
        result = StepResult(
            step_id=step.id, capability=step.capability, agent=agent.spec.name,
            ok=verified, detail=obs.detail, ms=ms, attempts=attempt,
            verifier_ok=verified, verifier_reason=reason,
        )
        return result, obs

    def _resolve(self, args: dict[str, Any], context: dict[str, dict]) -> dict[str, Any]:
        """Fill ${step.field} references from prior steps' outputs."""
        out: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, str):
                m = _REF.match(v)
                if m:
                    sid, field = m.groups()
                    out[k] = context.get(sid, {}).get(field)
                    continue
            out[k] = v
        return out

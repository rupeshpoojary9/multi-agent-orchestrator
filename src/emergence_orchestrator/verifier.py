"""Verifier — decides whether an executed step actually met its success criteria.

This is the reliability pillar: a step "succeeding" (the tool returned without an
exception) is NOT the same as the step being *correct*. The verifier re-checks
the observed result against the step's declared criteria, so a stale read, a
wrong value, or an empty result is caught before the orchestrator moves on. The
share of failures it catches (vs. lets leak) is the headline reliability metric.

Offline it evaluates a machine predicate (`step.check`) deterministically; with
an LLM it judges the natural-language `success_criteria`. Same contract either way.
"""

from __future__ import annotations

from typing import Any

from .llm import LLM
from .types import Observation, Step


class Verifier:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm
        self._cost = 0.0

    def pop_cost(self) -> float:
        c, self._cost = self._cost, 0.0
        return c

    def verify(self, step: Step, obs: Observation) -> tuple[bool, str]:
        if not obs.ok:
            return False, f"execution failed: {obs.detail}"
        if self.llm is not None and step.success_criteria:
            try:
                return self._llm_verify(step, obs)
            except Exception:
                pass  # fall back to the deterministic predicate
        return self._check_predicate(step.check, obs.data)

    # -- deterministic ----------------------------------------------------- #
    def _check_predicate(self, check: dict | None, data: dict[str, Any]) -> tuple[bool, str]:
        if not check:
            return True, "executed (no explicit criteria)"
        kind = check.get("kind")
        field = check.get("field")
        val = check.get("value")
        actual = data.get(field) if field else None

        if kind == "ok":
            return True, "executed"
        if kind == "equals":
            ok = actual == val
            return ok, f"{field}={actual!r} {'==' if ok else '!='} {val!r}"
        if kind == "present":
            ok = actual is not None and actual != "" and actual != []
            return ok, f"{field} {'present' if ok else 'missing'}"
        if kind == "contains":
            ok = actual is not None and val in actual
            return ok, f"{val!r} {'in' if ok else 'not in'} {field}"
        if kind == "nonempty":
            ok = bool(actual)
            return ok, f"{field} {'nonempty' if ok else 'empty'}"
        if kind == "count_at_least":
            n = len(actual) if actual is not None else 0
            ok = n >= val
            return ok, f"len({field})={n} {'>=' if ok else '<'} {val}"
        return False, f"unknown check kind {kind!r}"

    # -- llm --------------------------------------------------------------- #
    def _llm_verify(self, step: Step, obs: Observation) -> tuple[bool, str]:
        system = (
            "You are a strict verifier. Given a step's success criteria and the "
            "observed result, decide if the criteria are met. Output JSON: "
            '{"passed": bool, "reason": str}. Be skeptical of stale or empty data.'
        )
        prompt = (
            f"Success criteria: {step.success_criteria}\n"
            f"Observed detail: {obs.detail}\n"
            f"Observed data: {obs.data}\n"
        )
        res = self.llm.complete_json(system, prompt)
        self._cost += res.cost_usd
        data = res.data
        return bool(data.get("passed")), str(data.get("reason", ""))

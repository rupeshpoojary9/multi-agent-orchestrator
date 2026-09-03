"""Run the seeded tasks and compute the reliability metrics.

For each task the harness builds a FRESH system (so tasks never interfere), runs
the orchestrator, then checks the backends independently for ground truth. From
that it computes end-to-end success, the verifier catch rate (real failures the
verifier caught vs. leaked), attempts-to-success, cross-system and synthesis
success, latency percentiles, and cost per task — the numbers the résumé bullet
and the interview answers are built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..build import build_system
from .tasks import EVAL_TASKS, EvalTask, check_expect


@dataclass
class TaskOutcome:
    id: str
    ok: bool  # ground-truth success
    reported_ok: bool  # orchestrator's self-report
    leaked: bool  # reported success but ground truth failed (a missed error)
    verifier_catches: int
    attempts: int
    synthesized: int
    cross_system: bool
    expects_synth: bool
    ms: float
    cost_usd: float
    halted_reason: str | None


@dataclass
class Metrics:
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    def _rate(self, pred: Callable[[TaskOutcome], bool], among: Callable = lambda o: True) -> float:
        pool = [o for o in self.outcomes if among(o)]
        return (sum(1 for o in pool if pred(o)) / len(pool) * 100) if pool else 0.0

    @property
    def success_pct(self) -> float:
        return self._rate(lambda o: o.ok)

    @property
    def cross_system_success_pct(self) -> float:
        return self._rate(lambda o: o.ok, among=lambda o: o.cross_system)

    @property
    def synthesis_success_pct(self) -> float:
        return self._rate(lambda o: o.ok, among=lambda o: o.expects_synth)

    @property
    def total_catches(self) -> int:
        return sum(o.verifier_catches for o in self.outcomes)

    @property
    def total_leaks(self) -> int:
        return sum(1 for o in self.outcomes if o.leaked)

    @property
    def verifier_catch_rate_pct(self) -> float:
        caught, leaked = self.total_catches, self.total_leaks
        denom = caught + leaked
        return (caught / denom * 100) if denom else 100.0

    @property
    def mean_attempts(self) -> float:
        vals = [o.attempts for o in self.outcomes]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def synthesized_agents(self) -> int:
        return sum(o.synthesized for o in self.outcomes)

    @property
    def p50_ms(self) -> float:
        return _pct([o.ms for o in self.outcomes], 50)

    @property
    def p95_ms(self) -> float:
        return _pct([o.ms for o in self.outcomes], 95)

    @property
    def cost_per_task_usd(self) -> float:
        vals = [o.cost_usd for o in self.outcomes]
        return sum(vals) / len(vals) if vals else 0.0


def run_eval(
    tasks: list[EvalTask] | None = None,
    *,
    flaky: bool = True,
    use_llm: bool = True,
) -> Metrics:
    tasks = tasks if tasks is not None else EVAL_TASKS
    metrics = Metrics()
    for t in tasks:
        system = build_system(flaky=flaky, use_llm=use_llm)
        report = system.orchestrator.run(t.task)
        truth = check_expect(system, t.expect)
        attempts = max((s.attempts for s in report.steps), default=1)
        metrics.outcomes.append(TaskOutcome(
            id=t.id,
            ok=bool(report.ok and truth),
            reported_ok=report.ok,
            leaked=bool(report.ok and not truth),
            verifier_catches=report.verifier_catches,
            attempts=attempts,
            synthesized=len(report.synthesized_agents),
            cross_system=t.cross_system,
            expects_synth=t.expects_synth,
            ms=report.total_ms,
            cost_usd=report.total_cost_usd,
            halted_reason=report.halted_reason,
        ))
    return metrics


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

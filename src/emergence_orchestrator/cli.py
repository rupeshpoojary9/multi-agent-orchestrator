"""Command-line entry point.

    emergence demo "Onboard Umbrella Health across the console and the API."
    emergence eval [--offline] [--no-flaky]
    emergence agents

`demo` runs one task and prints the plan, per-step verify results, any agent it
synthesized, and the governance report. `eval` runs the full seeded suite and
prints the reliability metrics. Everything works offline; set ANTHROPIC_API_KEY
(and `pip install -e .[llm]`) to use a real model instead of the heuristics.
"""

from __future__ import annotations

import argparse
import os
import sys

from .build import build_system
from .eval.harness import run_eval
from .types import RunReport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emergence",
                                     description="Self-extending multi-agent orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run one natural-language task")
    d.add_argument("task", help="the task, e.g. 'Mark invoice 101 as paid.'")
    d.add_argument("--offline", action="store_true", help="force heuristic mode (no LLM)")
    d.add_argument("--no-flaky", action="store_true", help="disable the legacy stale-read")
    d.add_argument("--audit", action="store_true", help="print the full audit trail")

    e = sub.add_parser("eval", help="run the seeded eval suite and print metrics")
    e.add_argument("--offline", action="store_true", help="force heuristic mode (no LLM)")
    e.add_argument("--no-flaky", action="store_true", help="disable the legacy stale-read")

    sub.add_parser("agents", help="list registered agents")

    args = parser.parse_args(argv)

    if args.cmd == "demo":
        return _demo(args)
    if args.cmd == "eval":
        return _eval(args)
    if args.cmd == "agents":
        return _agents()
    return 1


# --------------------------------------------------------------------------- #
def _mode_banner(offline: bool) -> str:
    live = (not offline) and bool(os.environ.get("ANTHROPIC_API_KEY"))
    return "LLM (Anthropic)" if live else "offline heuristic"


def _demo(args) -> int:
    system = build_system(flaky=not args.no_flaky, use_llm=not args.offline)
    print(f"mode: {_mode_banner(args.offline)}")
    print(f"task: {args.task}\n")
    report = system.orchestrator.run(args.task)
    _print_report(report)
    if report.synthesized_agents:
        print("\nsynthesized agents (generated at runtime):")
        for name in report.synthesized_agents:
            spec = system.registry.get(name).spec
            print(f"  • {name}  composes={spec.composes}")
    if args.audit:
        print("\n--- audit trail ---")
        print(system.audit.to_jsonl())
    return 0 if report.ok else 2


def _print_report(report: RunReport) -> None:
    ok = "PASS" if report.ok else "FAIL"
    print(f"[{ok}] {len(report.steps)} steps  "
          f"{report.total_ms:.1f}ms  ${report.total_cost_usd:.4f}  "
          f"catches={report.verifier_catches}  replans={report.replans}")
    for s in report.steps:
        mark = "ok" if s.ok else "XX"
        tag = f" (synth:{s.synthesized_agent})" if s.synthesized_agent else ""
        print(f"  [{mark}] {s.step_id:<10} {s.capability:<26} "
              f"x{s.attempts} — {s.verifier_reason}{tag}")
    if report.halted_reason:
        print(f"  halted: {report.halted_reason}")


def _eval(args) -> int:
    print(f"mode: {_mode_banner(args.offline)}  |  running {_count()} tasks...\n")
    m = run_eval(flaky=not args.no_flaky, use_llm=not args.offline)
    rows = [
        ("tasks", f"{m.n}"),
        ("end-to-end success", f"{m.success_pct:.0f}%"),
        ("cross-system success", f"{m.cross_system_success_pct:.0f}%"),
        ("synthesis-task success", f"{m.synthesis_success_pct:.0f}%"),
        ("verifier catch rate", f"{m.verifier_catch_rate_pct:.0f}%  "
                                f"({m.total_catches} caught / {m.total_leaks} leaked)"),
        ("mean attempts / task", f"{m.mean_attempts:.2f}"),
        ("agents synthesized", f"{m.synthesized_agents}"),
        ("latency p50 / p95", f"{m.p50_ms:.1f} / {m.p95_ms:.1f} ms"),
        ("cost / task", f"${m.cost_per_task_usd:.4f}"),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{width}}  {v}")
    return 0 if m.success_pct == 100 else 2


def _count() -> int:
    from .eval.tasks import EVAL_TASKS
    return len(EVAL_TASKS)


def _agents() -> int:
    system = build_system()
    for spec in system.registry.list_specs():
        kind = "generated" if spec.generated else spec.kind
        print(f"{spec.name:<14} [{kind}]  caps={spec.capabilities}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

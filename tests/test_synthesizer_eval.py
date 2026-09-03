"""Synthesis gates (validation + dry-run) and the eval harness metrics."""

from multi_agent_orchestrator.build import build_system
from multi_agent_orchestrator.eval.harness import run_eval
from multi_agent_orchestrator.governance import AuditLog, Guardrails
from multi_agent_orchestrator.synthesizer import Synthesizer, SynthesisError


def _synth(allowed):
    system = build_system(use_llm=False)
    guard = Guardrails(allowed_tools=allowed)
    return Synthesizer(system.registry, guard, AuditLog()), system


def test_synthesis_rejected_when_tool_not_allowlisted():
    # Recipe needs web/api tools, but the guardrail allows nothing.
    synth, _ = _synth(allowed=set())
    try:
        synth.synthesize("workflow.onboard_account", {"customer_id": 1, "account_id": 1})
        assert False, "should have refused"
    except SynthesisError as e:
        assert "not allow-listed" in str(e)


def test_synthesis_refused_for_unknown_capability():
    synth, sysm = _synth(allowed=sysm_caps())
    try:
        synth.synthesize("workflow.does_not_exist", {})
        assert False
    except SynthesisError as e:
        assert "no recipe" in str(e)


def sysm_caps():
    s = build_system(use_llm=False)
    return set(s.registry.capabilities()) | {
        "workflow.onboard_account", "workflow.offboard_account"}


def test_dry_run_has_no_side_effects():
    # Synthesizing must not fire irreversible writes; state stays put.
    synth, system = _synth(allowed=sysm_caps())
    before = system.api.get_customer(4)["status"]
    synth.synthesize("workflow.onboard_account", {"customer_id": 4, "account_id": 4})
    after = system.api.get_customer(4)["status"]
    assert before == after  # the write only happens when the agent actually runs


def test_eval_suite_all_pass_offline():
    m = run_eval(use_llm=False)
    assert m.n == 26
    assert m.success_pct == 100.0
    assert m.verifier_catch_rate_pct == 100.0
    assert m.total_leaks == 0
    assert m.total_catches >= 1  # flaky reads really were caught
    assert m.synthesized_agents >= 3


def test_eval_no_catches_without_flakiness():
    m = run_eval(use_llm=False, flaky=False)
    assert m.success_pct == 100.0 and m.total_catches == 0

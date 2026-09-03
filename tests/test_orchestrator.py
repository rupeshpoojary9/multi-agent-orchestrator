"""End-to-end orchestration: verify+retry, synthesis, and governance halts."""

from multi_agent_orchestrator.build import build_system


def test_flaky_read_is_caught_and_recovered():
    system = build_system(flaky=True, use_llm=False)
    report = system.orchestrator.run("In the console, set account 4 status to active.")
    assert report.ok
    assert report.verifier_catches >= 1  # the stale read was caught
    read = next(s for s in report.steps if s.step_id == "read")
    assert read.attempts >= 2  # and recovered on retry
    assert system.web.accounts[4]["status"] == "active"


def test_no_catch_when_not_flaky():
    system = build_system(flaky=False, use_llm=False)
    report = system.orchestrator.run("In the console, set account 4 status to active.")
    assert report.ok and report.verifier_catches == 0


def test_cross_system_task():
    system = build_system(flaky=True, use_llm=False)
    report = system.orchestrator.run(
        'Mark invoice 104 as paid, then record a note "recon" on account 3.')
    assert report.ok
    assert system.api.get_invoice(104)["status"] == "paid"
    assert any("recon" in n for n in system.web.accounts[3]["notes"])


def test_self_extension_generates_and_registers_agent():
    system = build_system(flaky=True, use_llm=False)
    assert system.registry.find_by_capability("workflow.onboard_account") is None
    report = system.orchestrator.run("Onboard Cobalt Systems end to end.")
    assert report.ok
    assert report.synthesized_agents == ["gen_onboard_account"]
    # the generated agent is now a durable, registered artifact
    gen = system.registry.find_by_capability("workflow.onboard_account")
    assert gen is not None and gen.spec.generated
    assert system.web.accounts[2]["status"] == "active"
    assert system.api.get_customer(2)["status"] == "active"


def test_unsupported_task_halts_gracefully():
    system = build_system(use_llm=False)
    report = system.orchestrator.run("Translate the quarterly report into French.")
    assert not report.ok and "no recipe" in report.halted_reason


def test_approval_gate_blocks_irreversible_write():
    system = build_system(require_approval=True, approve=lambda cap, args: False)
    report = system.orchestrator.run("Mark invoice 101 as paid.")
    assert not report.ok and "approval gate" in report.halted_reason
    assert system.api.get_invoice(101)["status"] == "open"  # write never happened


def test_budget_halts_runaway():
    system = build_system(use_llm=False)
    system.orchestrator.max_steps = 1  # tighten below the plan size
    report = system.orchestrator.run("Onboard Vantage Health across the console and the API.")
    assert not report.ok and "budget" in report.halted_reason.lower()


def test_audit_trail_records_every_phase():
    system = build_system(flaky=True, use_llm=False)
    system.orchestrator.run("Onboard Vantage Health across the console and the API.")
    kinds = {e.kind for e in system.audit.events}
    assert {"plan", "route", "tool_call", "verify", "synthesize"} <= kinds

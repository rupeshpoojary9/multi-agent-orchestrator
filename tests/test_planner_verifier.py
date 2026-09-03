"""Planner produces valid routed DAGs; verifier enforces criteria."""

from emergence_orchestrator.agents.api_agent import APIAgent
from emergence_orchestrator.mocks.api_backend import APIBackend
from emergence_orchestrator.planner import MetaPlanner
from emergence_orchestrator.registry import AgentRegistry
from emergence_orchestrator.types import Observation, Step
from emergence_orchestrator.verifier import Verifier


def _planner():
    reg = AgentRegistry()
    reg.register(APIAgent(APIBackend()))
    return MetaPlanner(reg)


def test_plan_is_a_valid_dag_with_dependencies():
    plan = _planner().plan("Mark invoice 104 as paid, then record a note "
                            '"done" on account 3.')
    order = plan.topo_order()  # raises on a cycle / bad dep
    ids = [s.id for s in order]
    assert ids.index("pay") < ids.index("note")  # dependency respected


def test_onboarding_keyword_not_triggered_by_console_status_edit():
    # "set account N status to onboarding" must NOT route to the workflow.
    plan = _planner().plan("Set account 3 status to onboarding in the console.")
    caps = {s.capability for s in plan.steps}
    assert "workflow.onboard_account" not in caps
    assert "web.set_status" in caps


def test_planner_emits_workflow_capability_for_synthesis():
    plan = _planner().plan("Onboard Vantage Health across the console and the API.")
    caps = [s.capability for s in plan.steps]
    assert "workflow.onboard_account" in caps


def test_verifier_predicates():
    v = Verifier()
    step = Step("s", "d", "cap", check={"kind": "equals", "field": "status", "value": "paid"})
    ok, _ = v.verify(step, Observation(True, "", {"status": "paid"}))
    bad, _ = v.verify(step, Observation(True, "", {"status": "open"}))
    assert ok and not bad


def test_verifier_fails_on_execution_error():
    v = Verifier()
    step = Step("s", "d", "cap", check={"kind": "ok"})
    ok, reason = v.verify(step, Observation(False, "boom"))
    assert not ok and "boom" in reason


def test_verifier_count_and_contains():
    v = Verifier()
    s1 = Step("s", "d", "c", check={"kind": "count_at_least", "field": "xs", "value": 2})
    s2 = Step("s", "d", "c", check={"kind": "contains", "field": "notes", "value": "hi"})
    assert v.verify(s1, Observation(True, "", {"xs": [1, 2, 3]}))[0]
    assert not v.verify(s1, Observation(True, "", {"xs": [1]}))[0]
    # `contains` is membership: the note we added is present as an exact element.
    assert v.verify(s2, Observation(True, "", {"notes": ["hi", "bye"]}))[0]
    assert not v.verify(s2, Observation(True, "", {"notes": ["bye"]}))[0]

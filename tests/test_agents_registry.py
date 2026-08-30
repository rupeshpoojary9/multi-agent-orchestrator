"""Foundational agents + registry: capability routing and typed dispatch."""

from emergence_orchestrator.agents.api_agent import APIAgent
from emergence_orchestrator.agents.web_agent import WebAgent
from emergence_orchestrator.mocks.api_backend import APIBackend
from emergence_orchestrator.mocks.web_app import WebApp
from emergence_orchestrator.registry import AgentRegistry


def _registry():
    reg = AgentRegistry()
    reg.register(APIAgent(APIBackend()))
    reg.register(WebAgent(WebApp(flaky=False)))
    return reg


def test_routes_capability_to_owning_agent():
    reg = _registry()
    assert reg.find_by_capability("api.mark_invoice_paid").spec.name == "api_agent"
    assert reg.find_by_capability("web.set_status").spec.name == "web_agent"
    assert reg.find_by_capability("nonsense.tool") is None


def test_api_agent_typed_dispatch_and_errors():
    agent = APIAgent(APIBackend())
    ok = agent.run("api.find_customer", {"name": "Globex Corp"})
    assert ok.ok and ok.data["id"] == 2
    missing = agent.run("api.get_customer", {"customer_id": 999})
    assert not missing.ok and "not found" in missing.detail
    badargs = agent.run("api.get_customer", {})  # missing required arg
    assert not badargs.ok


def test_list_invoices_wraps_as_named_field():
    agent = APIAgent(APIBackend())
    obs = agent.run("api.list_invoices", {"customer_id": 1, "status": "open"})
    assert obs.ok and isinstance(obs.data["invoices"], list) and obs.data["invoices"]


def test_registry_persists_specs_to_sqlite():
    reg = _registry()
    names = {s.name for s in reg.load_specs()}
    assert {"api_agent", "web_agent"} <= names


def test_capabilities_union():
    reg = _registry()
    caps = reg.capabilities()
    assert "api.create_ticket" in caps and "web.add_note" in caps

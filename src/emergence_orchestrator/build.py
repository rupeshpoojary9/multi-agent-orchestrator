"""Assemble a ready-to-run orchestrator with the mock backends wired in.

One call gives you the whole system: seeded API + web mocks, both foundational
agents registered, the meta-planner/verifier/synthesizer, and the governance
layer configured (allow-list, irreversible set, budget). Used by the CLI, the
eval harness, and the tests so they all share one construction path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .agents.api_agent import APIAgent
from .agents.web_agent import WebAgent
from .governance import AuditLog, Guardrails
from .llm import get_llm
from .mocks.api_backend import APIBackend
from .mocks.web_app import WebApp
from .orchestrator import Orchestrator
from .planner import MetaPlanner
from .registry import AgentRegistry
from .synthesizer import Synthesizer
from .verifier import Verifier

# Higher-level workflows the meta-agent is permitted to synthesize on demand.
SYNTHESIZABLE = {"workflow.onboard_account", "workflow.offboard_account"}


@dataclass
class System:
    orchestrator: Orchestrator
    registry: AgentRegistry
    audit: AuditLog
    api: APIBackend
    web: WebApp


def build_system(
    *,
    flaky: bool = True,
    use_llm: bool = True,
    require_approval: bool = False,
    approve=lambda cap, args: True,
) -> System:
    llm = get_llm() if use_llm else None

    api = APIBackend()
    web = WebApp(flaky=flaky)

    registry = AgentRegistry()
    registry.register(APIAgent(api))
    registry.register(WebAgent(web))

    audit = AuditLog()

    allowed = set(registry.capabilities()) | SYNTHESIZABLE
    irreversible = {
        ts.name
        for spec in registry.list_specs()
        for ts in spec.tool_schemas
        if ts.irreversible
    }
    guardrails = Guardrails(
        allowed_tools=allowed,
        irreversible_tools=irreversible,
        require_approval=require_approval,
        approve=approve,
    )

    planner = MetaPlanner(registry, llm)
    verifier = Verifier(llm)
    synthesizer = Synthesizer(registry, guardrails, audit, llm)

    orch = Orchestrator(
        registry, planner, verifier, synthesizer, guardrails, audit,
        max_steps=int(os.environ.get("EMERGENCE_MAX_STEPS", 12)),
        max_replans=int(os.environ.get("EMERGENCE_MAX_REPLANS", 3)),
        max_cost_usd=float(os.environ.get("EMERGENCE_MAX_COST_USD", 0.50)),
    )
    return System(orchestrator=orch, registry=registry, audit=audit, api=api, web=web)

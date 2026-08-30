"""Self-extension — the meta-agent generates a new sub-agent at runtime.

When a plan needs a capability no registered agent has, the Synthesizer builds a
new agent, but only as a *recipe over vetted primitives* (see RecipeAgent). Every
generated agent passes two gates before it is allowed to act:

  1. schema validation — every tool in the recipe must be a registered,
     allow-listed primitive, and the recipe must be well-formed;
  2. a dry-run — read-only steps are actually executed with the real args and
     must succeed; irreversible steps are resolved and allow-list-checked but
     not fired, so the dry-run has no side effects.

Only then is the agent registered and routed to. If either gate fails, synthesis
is refused and the run halts — the system never lets an unvetted agent write.

Offline, recipes come from a small library so the behavior is deterministic and
testable. With an LLM, the model proposes the recipe and the *same* gates apply.
"""

from __future__ import annotations

from typing import Any

from .agents.recipe_agent import RecipeAgent, fill_args
from .governance import AuditLog, Guardrails
from .llm import LLM
from .registry import AgentRegistry
from .types import AgentSpec, ToolSchema


class SynthesisError(Exception):
    pass


# Deterministic recipe library for offline mode. Each is a reusable cross-system
# workflow the meta-agent can "discover" it needs and generate on the fly.
RECIPE_LIBRARY: dict[str, dict] = {
    "workflow.onboard_account": {
        "description": "Activate a customer everywhere: legacy console + API, with a note.",
        "recipe": [
            {"tool": "web.login", "args": {"user": "orchestrator", "password": "svc"}},
            {"tool": "web.open_account", "args": {"account_id": "$account_id"}},
            {"tool": "web.set_status", "args": {"account_id": "$account_id", "status": "active"}},
            {"tool": "web.add_note", "args": {"account_id": "$account_id", "text": "Onboarded via workflow"}},
            {"tool": "api.set_customer_status", "args": {"customer_id": "$customer_id", "status": "active"}},
            {"tool": "web.read_status", "args": {"account_id": "$account_id"}, "bind": "status"},
        ],
    },
    "workflow.offboard_account": {
        "description": "Suspend a customer everywhere and open an offboarding ticket.",
        "recipe": [
            {"tool": "web.login", "args": {"user": "orchestrator", "password": "svc"}},
            {"tool": "web.open_account", "args": {"account_id": "$account_id"}},
            {"tool": "web.set_status", "args": {"account_id": "$account_id", "status": "suspended"}},
            {"tool": "web.add_note", "args": {"account_id": "$account_id", "text": "Offboarded via workflow"}},
            {"tool": "api.set_customer_status", "args": {"customer_id": "$customer_id", "status": "delinquent"}},
            {"tool": "api.create_ticket", "args": {"customer_id": "$customer_id", "subject": "Account offboarded", "body": "Suspended via workflow"}},
            {"tool": "web.read_status", "args": {"account_id": "$account_id"}, "bind": "status"},
        ],
    },
}


class Synthesizer:
    def __init__(
        self,
        registry: AgentRegistry,
        guardrails: Guardrails,
        audit: AuditLog,
        llm: LLM | None = None,
    ) -> None:
        self.registry = registry
        self.guardrails = guardrails
        self.audit = audit
        self.llm = llm
        self._cost = 0.0

    def pop_cost(self) -> float:
        c, self._cost = self._cost, 0.0
        return c

    def synthesize(self, capability: str, sample_args: dict[str, Any]) -> RecipeAgent:
        """Generate, gate, register, and return an agent for `capability`."""
        proposal = self._propose(capability)
        recipe = proposal["recipe"]

        errors = self._validate(recipe)
        if errors:
            self.audit.record("guardrail", action="reject_synthesis",
                              capability=capability, errors=errors)
            raise SynthesisError(f"invalid recipe for {capability}: {errors}")

        dry_err = self._dry_run(recipe, sample_args)
        if dry_err:
            self.audit.record("guardrail", action="reject_synthesis",
                              capability=capability, dry_run_error=dry_err)
            raise SynthesisError(f"dry-run failed for {capability}: {dry_err}")

        name = "gen_" + capability.split(".")[-1]
        spec = AgentSpec(
            name=name,
            kind="composite",
            description=proposal.get("description", f"Generated agent for {capability}"),
            capabilities=[capability],
            tool_schemas=[ToolSchema(capability, proposal.get("description", ""),
                                     {k: "any" for k in sample_args})],
            system_prompt=proposal.get("system_prompt", ""),
            generated=True,
            composes=sorted({s["tool"] for s in recipe}),
        )
        agent = RecipeAgent(spec, recipe, self.registry)
        self.registry.register(agent)
        self.audit.record("synthesize", capability=capability, agent=name,
                          composes=spec.composes, steps=len(recipe))
        return agent

    # -- proposal ---------------------------------------------------------- #
    def _propose(self, capability: str) -> dict:
        if self.llm is not None:
            try:
                return self._llm_propose(capability)
            except Exception:
                pass  # fall back to the library if the model output is unusable
        if capability in RECIPE_LIBRARY:
            return RECIPE_LIBRARY[capability]
        raise SynthesisError(f"no recipe available for capability {capability!r}")

    def _llm_propose(self, capability: str) -> dict:
        tools = []
        for spec in self.registry.list_specs():
            for ts in spec.tool_schemas:
                tools.append(f"- {ts.name}({ts.params}) — {ts.description}")
        system = (
            "You design a new agent as a RECIPE composing existing primitive tools "
            "only. Never invent tools. Output strict JSON: "
            '{"description": str, "system_prompt": str, '
            '"recipe": [{"tool": str, "args": {..}, "bind": str?}]}. '
            'Args may use "$argname" or "${bind.field}" references.'
        )
        prompt = (
            f"Capability to build: {capability}\n\n"
            f"Available primitive tools:\n" + "\n".join(sorted(tools)) +
            "\n\nDesign the recipe."
        )
        res = self.llm.complete_json(system, prompt)
        self._cost += res.cost_usd
        return res.data

    # -- gates ------------------------------------------------------------- #
    def _validate(self, recipe: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(recipe, list) or not recipe:
            return ["recipe must be a non-empty list"]
        registered = self.registry.capabilities()
        for i, step in enumerate(recipe):
            if not isinstance(step, dict) or "tool" not in step:
                errors.append(f"step {i}: missing 'tool'")
                continue
            tool = step["tool"]
            if tool not in registered:
                errors.append(f"step {i}: tool {tool!r} not registered")
            elif tool not in self.guardrails.allowed_tools:
                errors.append(f"step {i}: tool {tool!r} not allow-listed")
        return errors

    def _dry_run(self, recipe: list[dict], sample_args: dict[str, Any]) -> str | None:
        """Execute read-only steps for real; resolve-only the irreversible ones."""
        irreversible = self._irreversible_tools()
        outputs: dict[str, dict] = {}
        for i, step in enumerate(recipe):
            tool = step["tool"]
            resolved = fill_args(step.get("args", {}), sample_args, outputs)
            if any(v is None for v in resolved.values()):
                return f"step {i} ({tool}): unresolved args {resolved}"
            if tool in irreversible:
                continue  # do not fire writes during a dry-run
            agent = self.registry.find_by_capability(tool)
            obs = agent.run(tool, resolved)
            if not obs.ok:
                return f"step {i} ({tool}) dry-run failed: {obs.detail}"
            if step.get("bind"):
                outputs[step["bind"]] = obs.data
        return None

    def _irreversible_tools(self) -> set[str]:
        out: set[str] = set()
        for spec in self.registry.list_specs():
            for ts in spec.tool_schemas:
                if ts.irreversible:
                    out.add(ts.name)
        return out

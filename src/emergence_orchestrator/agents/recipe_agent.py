"""RecipeAgent — the shape every synthesized sub-agent takes.

A generated agent never invents raw capability. It is a *recipe*: an ordered
list of calls to already-registered primitive tools, with args templated from the
capability's own arguments and from earlier calls' outputs. This is what makes
"agents that create agents" safe here — the meta-agent composes vetted
primitives into a new, named, reusable workflow, and nothing else.

Recipe step shape:
    {"tool": "web.set_status",
     "args": {"account_id": "$account_id", "status": "active"},
     "bind": "set"}          # optional: store this call's output under 'set'

Arg values:
    "$name"           -> the composite call's argument `name`
    "${bind.field}"   -> field from an earlier bound call's output
    anything else     -> literal
"""

from __future__ import annotations

import re
from typing import Any

from ..types import AgentSpec, Observation

_REF = re.compile(r"^\$\{(\w+)\.(\w+)\}$")
_VAR = re.compile(r"^\$(\w+)$")


def fill_args(
    template: dict[str, Any], call_args: dict[str, Any], outputs: dict[str, dict]
) -> dict[str, Any]:
    """Resolve `$var` / `${bind.field}` references in a recipe step's args."""
    out: dict[str, Any] = {}
    for k, v in template.items():
        if isinstance(v, str):
            m = _REF.match(v)
            if m:
                bind, field = m.groups()
                out[k] = outputs.get(bind, {}).get(field)
                continue
            m = _VAR.match(v)
            if m:
                out[k] = call_args.get(m.group(1))
                continue
        out[k] = v
    return out


class RecipeAgent:
    """An agent whose single capability runs a recipe over registry primitives."""

    def __init__(self, spec: AgentSpec, recipe: list[dict], registry) -> None:
        self.spec = spec
        self.recipe = recipe
        self._registry = registry

    @property
    def capability(self) -> str:
        return self.spec.capabilities[0]

    def run(self, capability: str, args: dict[str, Any]) -> Observation:
        if capability != self.capability:
            return Observation(False, f"{self.spec.name} only does {self.capability!r}")
        outputs: dict[str, dict] = {}
        merged: dict[str, Any] = {}
        for i, step in enumerate(self.recipe):
            tool = step["tool"]
            primitive = self._registry.find_by_capability(tool)
            if primitive is None:
                return Observation(False, f"recipe tool {tool!r} not registered")
            resolved = fill_args(step.get("args", {}), args, outputs)
            obs = primitive.run(tool, resolved)
            if not obs.ok:
                return Observation(False, f"recipe step {i} ({tool}) failed: {obs.detail}",
                                   data=merged)
            if step.get("bind"):
                outputs[step["bind"]] = obs.data
            merged.update(obs.data)
        return Observation(True, f"{self.capability} completed {len(self.recipe)} steps",
                           data=merged)

"""The single contract every agent — foundational or synthesized — implements.

An agent is just a `spec` (what it can do, declaratively) plus a `run` that
executes one capability with typed args and returns a uniform `Observation`.
That uniformity is the whole integration story: a Playwright browser agent and a
typed REST agent are indistinguishable to the orchestrator at the seam.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from ..types import AgentSpec, Observation


@runtime_checkable
class Agent(Protocol):
    spec: AgentSpec

    def run(self, capability: str, args: dict[str, Any]) -> Observation:
        ...


class ToolAgent:
    """Base class binding capability names to Python callables.

    Subclasses register `{capability: callable}` and describe them in `spec`.
    `run` handles dispatch, argument errors, and backend exceptions uniformly so
    every agent surfaces failures the same way for the verifier to reason about.
    """

    spec: AgentSpec
    _tools: dict[str, Callable[..., Any]]

    def __init__(self, spec: AgentSpec, tools: dict[str, Callable[..., Any]]) -> None:
        self.spec = spec
        self._tools = tools

    def run(self, capability: str, args: dict[str, Any]) -> Observation:
        fn = self._tools.get(capability)
        if fn is None:
            return Observation(
                ok=False, detail=f"{self.spec.name} cannot do {capability!r}"
            )
        try:
            data = fn(**args)
        except TypeError as e:  # bad/missing args against the typed signature
            return Observation(ok=False, detail=f"bad args for {capability}: {e}")
        except Exception as e:  # backend rejected the call (validation, not-found)
            return Observation(ok=False, detail=f"{type(e).__name__}: {e}")
        payload = data if isinstance(data, dict) else {"result": data}
        return Observation(ok=True, detail=f"{capability} ok", data=payload)

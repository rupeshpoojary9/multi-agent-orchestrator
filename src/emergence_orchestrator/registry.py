"""Agent registry — the routing table and the write target for self-extension.

Holds live agent instances for dispatch and persists their specs to SQLite so a
sub-agent the meta-agent generates at runtime is a durable, auditable artifact
(name, capabilities, prompt, and the primitives it composes), not an ephemeral
object. This is the registry/SDK pattern the orchestrator productizes: agents
are data you can enumerate, and the meta-agent can add rows to it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from .agents.base import Agent
from .types import AgentSpec, ToolSchema


class AgentRegistry:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_specs (
                name TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                generated INTEGER NOT NULL,
                spec_json TEXT NOT NULL
            )
            """
        )
        self.db.commit()
        self._agents: dict[str, Agent] = {}

    # -- registration ------------------------------------------------------ #
    def register(self, agent: Agent) -> None:
        spec = agent.spec
        self._agents[spec.name] = agent
        self.db.execute(
            "INSERT OR REPLACE INTO agent_specs (name, kind, generated, spec_json) "
            "VALUES (?,?,?,?)",
            (spec.name, spec.kind, int(spec.generated), json.dumps(asdict(spec))),
        )
        self.db.commit()

    # -- lookup ------------------------------------------------------------ #
    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def find_by_capability(self, capability: str) -> Agent | None:
        """First live agent that declares this capability (foundational first)."""
        # Prefer foundational agents over generated ones for shared primitives.
        gen_last = sorted(self._agents.values(), key=lambda a: a.spec.generated)
        for agent in gen_last:
            if agent.spec.supports(capability):
                return agent
        return None

    def capabilities(self) -> set[str]:
        caps: set[str] = set()
        for agent in self._agents.values():
            caps.update(agent.spec.capabilities)
        return caps

    def list_specs(self) -> list[AgentSpec]:
        return [a.spec for a in self._agents.values()]

    def load_specs(self) -> list[AgentSpec]:
        """Read persisted specs (including generated ones) from SQLite."""
        rows = self.db.execute("SELECT spec_json FROM agent_specs").fetchall()
        out: list[AgentSpec] = []
        for r in rows:
            d = json.loads(r["spec_json"])
            d["tool_schemas"] = [ToolSchema(**t) for t in d.get("tool_schemas", [])]
            out.append(AgentSpec(**d))
        return out

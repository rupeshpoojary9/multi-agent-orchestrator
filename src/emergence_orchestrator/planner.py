"""Meta-agent planner — natural-language task -> a routed task DAG.

Produces a `Plan` of `Step`s, each carrying: the capability it needs, typed args
(with `${step.field}` references for cross-step data flow), a dependency list, and
BOTH a machine-checkable predicate and a natural-language success criterion. When
the cleanest plan needs a reusable cross-system workflow that no foundational
agent provides, the planner emits that higher-level capability and leaves the
agent unresolved — the orchestrator then triggers synthesis. That is the seam
where "agents that create agents" happens.

Offline, `_heuristic_plan` parses intent and entities deterministically so the
whole system runs and is testable with no API key. With an LLM, `_llm_plan`
grounds the model in the live registry's capabilities. Same `Plan` contract.
"""

from __future__ import annotations

import re
from typing import Any

from .llm import LLM
from .registry import AgentRegistry
from .types import Plan, Step

WEB_STATUSES = {"active", "suspended", "onboarding", "prospect"}
API_STATUSES = {"active", "delinquent"}


class MetaPlanner:
    def __init__(self, registry: AgentRegistry, llm: LLM | None = None) -> None:
        self.registry = registry
        self.llm = llm
        self._cost = 0.0

    def pop_cost(self) -> float:
        c, self._cost = self._cost, 0.0
        return c

    def plan(self, task: str) -> Plan:
        if self.llm is not None:
            try:
                return self._llm_plan(task)
            except Exception:
                pass  # fall back to the deterministic planner
        return self._heuristic_plan(task)

    # ------------------------------------------------------------------ #
    # Heuristic planner
    # ------------------------------------------------------------------ #
    def _heuristic_plan(self, task: str) -> Plan:
        t = task.lower()
        name = self._match_customer(task)
        acct = _first_int(r"account\s*#?(\d+)", t)
        invoice = _first_int(r"invoice\s*#?(\d+)", t)
        note = _quoted_after(task, r"note")
        subject = _quoted_after(task, r"about|regarding|subject")
        status = self._match_status(t, WEB_STATUSES)

        steps: list[Step] = []

        def find_step() -> str:
            steps.append(Step(
                id="find", description=f"resolve customer {name!r} to an id",
                capability="api.find_customer", args={"name": name},
                success_criteria="a matching customer id is returned",
                check={"kind": "present", "field": "id"},
            ))
            return "find"

        # -- self-extension workflows (no foundational agent has these) ----- #
        # Require a customer name so "set account N status to onboarding" (a plain
        # web-console edit) does not get mistaken for the onboarding workflow.
        if "onboard" in t and name and acct is None:
            find_step()
            steps.append(Step(
                id="wf", description=f"onboard {name} across console and API",
                capability="workflow.onboard_account",
                args={"customer_id": "${find.id}", "account_id": "${find.id}"},
                depends_on=["find"],
                success_criteria="account shows 'active' in the legacy console",
                check={"kind": "equals", "field": "status", "value": "active"},
            ))
            return Plan(task, steps)

        if "offboard" in t and name and acct is None:
            find_step()
            steps.append(Step(
                id="wf", description=f"offboard {name} across console and API",
                capability="workflow.offboard_account",
                args={"customer_id": "${find.id}", "account_id": "${find.id}"},
                depends_on=["find"],
                success_criteria="account shows 'suspended' in the legacy console",
                check={"kind": "equals", "field": "status", "value": "suspended"},
            ))
            return Plan(task, steps)

        # -- API: mark an invoice paid (optionally note it in the console) -- #
        if invoice is not None and ("paid" in t or "pay" in t):
            steps.append(Step(
                id="pay", description=f"mark invoice {invoice} paid",
                capability="api.mark_invoice_paid", args={"invoice_id": invoice},
                success_criteria=f"invoice {invoice} status is 'paid'",
                check={"kind": "equals", "field": "status", "value": "paid"},
            ))
            if acct is not None and ("note" in t or "record" in t or "console" in t):
                _append_web_note(steps, acct,
                                 note or f"Invoice {invoice} reconciled",
                                 depends_on=["pay"])
            return Plan(task, steps)

        # -- API: open a support ticket ------------------------------------ #
        if "ticket" in t:
            find_step()
            steps.append(Step(
                id="ticket", description=f"open a ticket for {name}",
                capability="api.create_ticket",
                args={"customer_id": "${find.id}",
                      "subject": subject or "Support request",
                      "body": note or subject or "Opened by orchestrator"},
                depends_on=["find"],
                success_criteria="a ticket id is returned",
                check={"kind": "present", "field": "id"},
            ))
            return Plan(task, steps)

        # -- API: list a customer's open invoices -------------------------- #
        if "invoice" in t and ("open" in t or "list" in t or "how many" in t):
            find_step()
            steps.append(Step(
                id="inv", description=f"list open invoices for {name}",
                capability="api.list_invoices",
                args={"customer_id": "${find.id}", "status": "open"},
                depends_on=["find"],
                success_criteria="at least one open invoice is returned",
                check={"kind": "count_at_least", "field": "invoices", "value": 1},
            ))
            return Plan(task, steps)

        # -- API: set a customer's status directly ------------------------- #
        if name and ("suspend" in t or "activate" in t or "set" in t) and (
            "api" in t or "customer" in t
        ):
            target = "delinquent" if "suspend" in t else "active"
            find_step()
            steps.append(Step(
                id="cust", description=f"set {name} status to {target} via API",
                capability="api.set_customer_status",
                args={"customer_id": "${find.id}", "status": target},
                depends_on=["find"],
                success_criteria=f"customer status is '{target}'",
                check={"kind": "equals", "field": "status", "value": target},
            ))
            return Plan(task, steps)

        # -- Web: add a note to an account --------------------------------- #
        if acct is not None and "note" in t:
            _append_web_note(steps, acct, note or "Note added by orchestrator")
            return Plan(task, steps)

        # -- Web: set an account's status in the console ------------------- #
        if acct is not None and status:
            steps.append(Step(id="login", description="log into the console",
                              capability="web.login",
                              args={"user": "orchestrator", "password": "svc"},
                              check={"kind": "equals", "field": "logged_in", "value": True}))
            steps.append(Step(id="open", description=f"open account {acct}",
                              capability="web.open_account", args={"account_id": acct},
                              depends_on=["login"],
                              check={"kind": "equals", "field": "account_id", "value": acct}))
            steps.append(Step(id="set", description=f"set status to {status}",
                              capability="web.set_status",
                              args={"account_id": acct, "status": status},
                              depends_on=["open"], check={"kind": "ok"}))
            steps.append(Step(id="read", description="re-read status to confirm",
                              capability="web.read_status", args={"account_id": acct},
                              depends_on=["set"],
                              success_criteria=f"account status reads '{status}'",
                              check={"kind": "equals", "field": "status", "value": status}))
            return Plan(task, steps)

        # -- unsupported: a single failing step the verifier will catch ---- #
        steps.append(Step(id="unsupported", description="no plan for this task",
                          capability="workflow.unknown", args={},
                          success_criteria="never satisfiable",
                          check={"kind": "equals", "field": "status", "value": "impossible"}))
        return Plan(task, steps)

    # ------------------------------------------------------------------ #
    # Entity helpers
    # ------------------------------------------------------------------ #
    def _customers(self) -> list[dict[str, Any]]:
        agent = self.registry.find_by_capability("api.list_customers")
        if agent is None:
            return []
        obs = agent.run("api.list_customers", {})
        return obs.data.get("customers", []) if obs.ok else []

    def _match_customer(self, task: str) -> str | None:
        low = task.lower()
        best = None
        for c in self._customers():
            if c["name"].lower() in low:
                if best is None or len(c["name"]) > len(best):
                    best = c["name"]
        return best

    def _match_status(self, t: str, choices: set[str]) -> str | None:
        for s in choices:
            if re.search(rf"\b{s}\b", t):
                return s
        return None

    # ------------------------------------------------------------------ #
    # LLM planner
    # ------------------------------------------------------------------ #
    def _llm_plan(self, task: str) -> Plan:
        caps = []
        for spec in self.registry.list_specs():
            for ts in spec.tool_schemas:
                caps.append(f"- {ts.name}({ts.params}) — {ts.description}")
        caps.append("- workflow.onboard_account({customer_id, account_id}) — "
                    "reusable cross-system onboarding (synthesized on demand)")
        caps.append("- workflow.offboard_account({customer_id, account_id}) — "
                    "reusable cross-system offboarding (synthesized on demand)")
        system = (
            "You are a meta-agent that decomposes an enterprise task into a JSON "
            "task DAG. Output: {\"steps\": [{\"id\", \"description\", \"capability\", "
            "\"args\", \"depends_on\", \"success_criteria\", \"check\"}]}. "
            "`check` is a predicate like {\"kind\":\"equals\",\"field\":\"status\","
            "\"value\":\"active\"} (kinds: ok, equals, present, contains, nonempty, "
            "count_at_least). Use ${stepid.field} to pass data between steps. If a "
            "reusable cross-system workflow fits, emit workflow.* and it will be "
            "synthesized. Prefer reads before writes."
        )
        prompt = f"Task: {task}\n\nAvailable capabilities:\n" + "\n".join(sorted(caps))
        res = self.llm.complete_json(system, prompt, max_tokens=2000)
        self._cost += res.cost_usd
        steps = [
            Step(
                id=s["id"], description=s.get("description", s["id"]),
                capability=s["capability"], args=s.get("args", {}),
                depends_on=s.get("depends_on", []),
                success_criteria=s.get("success_criteria", ""),
                check=s.get("check"),
            )
            for s in res.data["steps"]
        ]
        return Plan(task, steps)


# --------------------------------------------------------------------------- #
# Module-level parsing helpers
# --------------------------------------------------------------------------- #
def _append_web_note(steps: list[Step], acct: int, text: str, depends_on: list[str] | None = None) -> None:
    dep = list(depends_on or [])
    steps.append(Step(id="login", description="log into the console",
                      capability="web.login",
                      args={"user": "orchestrator", "password": "svc"},
                      depends_on=dep,
                      check={"kind": "equals", "field": "logged_in", "value": True}))
    steps.append(Step(id="open", description=f"open account {acct}",
                      capability="web.open_account", args={"account_id": acct},
                      depends_on=["login"],
                      check={"kind": "equals", "field": "account_id", "value": acct}))
    steps.append(Step(id="note", description="append the note",
                      capability="web.add_note", args={"account_id": acct, "text": text},
                      depends_on=["open"], check={"kind": "present", "field": "note_count"}))
    steps.append(Step(id="confirm", description="re-read notes to confirm",
                      capability="web.read_notes", args={"account_id": acct},
                      depends_on=["note"],
                      success_criteria=f"notes contain {text!r}",
                      check={"kind": "contains", "field": "notes", "value": text}))


def _first_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _quoted_after(text: str, keyword_pattern: str) -> str | None:
    m = re.search(rf"(?:{keyword_pattern})\s+['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

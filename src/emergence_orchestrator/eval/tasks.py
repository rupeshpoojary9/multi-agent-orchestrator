"""Seeded enterprise tasks with independent ground-truth checks.

Each task is a natural-language instruction plus an `expect` describing the state
the backends must actually reach. The check reads the mock backends DIRECTLY
(bypassing the agents), so success is measured against reality, not the
orchestrator's own self-report — that's what makes the verifier-catch and leak
metrics honest.

`expect` is one predicate tuple or a list of them (all must hold):
  ("web_status", account_id, status)
  ("web_note", account_id, substring)
  ("invoice_status", invoice_id, status)
  ("customer_status", customer_id, status)
  ("open_invoices_ge", customer_id, n)
  ("ticket_for", customer_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalTask:
    id: str
    task: str
    expect: Any  # a predicate tuple or list of tuples
    cross_system: bool = False
    expects_synth: bool = False
    tags: list[str] = field(default_factory=list)


EVAL_TASKS: list[EvalTask] = [
    # --- Web: set account status (exercises stale-read verify + retry) ------
    EvalTask("web-status-1", "In the console, set account 4 status to active.",
             ("web_status", 4, "active"), tags=["web"]),
    EvalTask("web-status-2", "Set account 2 status to active in the legacy console.",
             ("web_status", 2, "active"), tags=["web"]),
    EvalTask("web-status-3", "Set account 3 status to onboarding in the console.",
             ("web_status", 3, "onboarding"), tags=["web"]),
    EvalTask("web-status-4", "Set account 1 status to suspended in the console.",
             ("web_status", 1, "suspended"), tags=["web"]),
    EvalTask("web-status-5", "Set account 5 status to onboarding in the console.",
             ("web_status", 5, "onboarding"), tags=["web"]),
    # --- Web: notes --------------------------------------------------------
    EvalTask("web-note-1", 'Add a note "Called customer, no answer" to account 1.',
             ("web_note", 1, "Called customer, no answer"), tags=["web"]),
    EvalTask("web-note-2", 'Add a note "Reviewed contract terms" to account 3.',
             ("web_note", 3, "Reviewed contract terms"), tags=["web"]),
    EvalTask("web-note-3", 'Add a note "Escalated to finance" to account 2.',
             ("web_note", 2, "Escalated to finance"), tags=["web"]),
    # --- API: list open invoices ------------------------------------------
    EvalTask("api-inv-list-1", "How many open invoices does Northwind Traders have? List them.",
             ("open_invoices_ge", 1, 1), tags=["api"]),
    EvalTask("api-inv-list-2", "List the open invoices for Globex Corp.",
             ("open_invoices_ge", 2, 1), tags=["api"]),
    EvalTask("api-inv-list-3", "List open invoices for Umbrella Health.",
             ("open_invoices_ge", 4, 1), tags=["api"]),
    EvalTask("api-inv-list-4", "List open invoices for Wayne Enterprises.",
             ("open_invoices_ge", 5, 1), tags=["api"]),
    # --- API: mark invoice paid -------------------------------------------
    EvalTask("api-pay-1", "Mark invoice 101 as paid.",
             ("invoice_status", 101, "paid"), tags=["api"]),
    EvalTask("api-pay-2", "Mark invoice 103 as paid.",
             ("invoice_status", 103, "paid"), tags=["api"]),
    EvalTask("api-pay-3", "Mark invoice 106 as paid.",
             ("invoice_status", 106, "paid"), tags=["api"]),
    EvalTask("api-pay-4", "Mark invoice 107 as paid.",
             ("invoice_status", 107, "paid"), tags=["api"]),
    # --- API: tickets ------------------------------------------------------
    EvalTask("api-ticket-1", 'Open a support ticket for Initech about "Overdue balance follow-up".',
             ("ticket_for", 3), tags=["api"]),
    EvalTask("api-ticket-2", 'Open a ticket for Globex Corp about "Integration question".',
             ("ticket_for", 2), tags=["api"]),
    EvalTask("api-ticket-3", 'Open a ticket for Umbrella Health about "Renewal discussion".',
             ("ticket_for", 4), tags=["api"]),
    # --- API: set customer status -----------------------------------------
    EvalTask("api-cust-1", "Suspend Globex Corp's customer account via the API.",
             ("customer_status", 2, "delinquent"), tags=["api"]),
    EvalTask("api-cust-2", "Activate Umbrella Health's customer account via the API.",
             ("customer_status", 4, "active"), tags=["api"]),
    # --- Cross-system explicit DAGs ---------------------------------------
    EvalTask("cross-1",
             'Mark invoice 104 as paid, then record a note "Paid via reconciliation" on account 3.',
             [("invoice_status", 104, "paid"), ("web_note", 3, "Paid via reconciliation")],
             cross_system=True, tags=["cross"]),
    EvalTask("cross-2",
             'Mark invoice 105 as paid and add a note "Second invoice cleared" to account 3.',
             [("invoice_status", 105, "paid"), ("web_note", 3, "Second invoice cleared")],
             cross_system=True, tags=["cross"]),
    # --- Self-extension: workflows synthesized on the fly -----------------
    EvalTask("synth-onboard-1", "Onboard Umbrella Health across the console and the API.",
             [("web_status", 4, "active"), ("customer_status", 4, "active")],
             cross_system=True, expects_synth=True, tags=["synthesis", "cross"]),
    EvalTask("synth-onboard-2", "Onboard Globex Corp end to end.",
             [("web_status", 2, "active"), ("customer_status", 2, "active")],
             cross_system=True, expects_synth=True, tags=["synthesis", "cross"]),
    EvalTask("synth-offboard-1", "Offboard Wayne Enterprises across all systems.",
             [("web_status", 5, "suspended"), ("customer_status", 5, "delinquent"),
              ("ticket_for", 5)],
             cross_system=True, expects_synth=True, tags=["synthesis", "cross"]),
]


def check_expect(system, expect: Any) -> bool:
    """Independently verify the backends reached the expected state."""
    preds = expect if isinstance(expect, list) else [expect]
    return all(_check_one(system, p) for p in preds)


def _check_one(system, pred: tuple) -> bool:
    kind = pred[0]
    if kind == "web_status":
        _, acct, status = pred
        return system.web.accounts.get(acct, {}).get("status") == status
    if kind == "web_note":
        _, acct, sub = pred
        return any(sub in n for n in system.web.accounts.get(acct, {}).get("notes", []))
    if kind == "invoice_status":
        _, inv, status = pred
        row = system.api.db.execute(
            "SELECT status FROM invoices WHERE id=?", (inv,)).fetchone()
        return row is not None and row["status"] == status
    if kind == "customer_status":
        _, cid, status = pred
        row = system.api.db.execute(
            "SELECT status FROM customers WHERE id=?", (cid,)).fetchone()
        return row is not None and row["status"] == status
    if kind == "open_invoices_ge":
        _, cid, n = pred
        row = system.api.db.execute(
            "SELECT COUNT(*) c FROM invoices WHERE customer_id=? AND status='open'",
            (cid,)).fetchone()
        return row["c"] >= n
    if kind == "ticket_for":
        _, cid = pred
        row = system.api.db.execute(
            "SELECT COUNT(*) c FROM tickets WHERE customer_id=?", (cid,)).fetchone()
        return row["c"] >= 1
    raise ValueError(f"unknown expect predicate {pred!r}")

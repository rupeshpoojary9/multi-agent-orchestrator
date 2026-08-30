"""API Agent — typed REST tools against the mock enterprise API.

Each capability is a thin, typed wrapper over an APIBackend method. In a real
deployment these would be authenticated HTTP calls; the shape (typed in, dict
out, exceptions surfaced) is identical, which is the point — swapping the
backend for a live service does not change the agent contract.
"""

from __future__ import annotations

from ..mocks.api_backend import APIBackend
from ..types import AgentSpec, ToolSchema
from .base import ToolAgent

SCHEMAS = [
    ToolSchema("api.list_customers", "List all customers", {}),
    ToolSchema("api.find_customer", "Look up a customer by exact name",
               {"name": "str"}),
    ToolSchema("api.get_customer", "Fetch a customer record by id",
               {"customer_id": "int"}),
    ToolSchema("api.list_invoices", "List a customer's invoices, optionally by status",
               {"customer_id": "int", "status": "str"}),
    ToolSchema("api.get_invoice", "Fetch one invoice by id", {"invoice_id": "int"}),
    ToolSchema("api.mark_invoice_paid", "Mark an invoice paid",
               {"invoice_id": "int"}, irreversible=True),
    ToolSchema("api.set_customer_status", "Set a customer's account status",
               {"customer_id": "int", "status": "str"}, irreversible=True),
    ToolSchema("api.create_ticket", "Open a support ticket for a customer",
               {"customer_id": "int", "subject": "str", "body": "str"},
               irreversible=True),
]


class APIAgent(ToolAgent):
    kind = "api"

    def __init__(self, backend: APIBackend) -> None:
        self.backend = backend
        spec = AgentSpec(
            name="api_agent",
            kind="api",
            description="Drives the enterprise REST API: customers, invoices, tickets.",
            capabilities=[s.name for s in SCHEMAS],
            tool_schemas=SCHEMAS,
            system_prompt=(
                "You operate a typed enterprise billing/support API. Prefer reads "
                "before writes; every write is auditable and may need approval."
            ),
        )
        tools = {
            "api.list_customers": lambda: {"customers": backend.list_customers()},
            "api.find_customer": backend.find_customer_by_name,
            "api.get_customer": backend.get_customer,
            "api.list_invoices": lambda customer_id, status=None: {
                "invoices": backend.list_invoices(customer_id, status)
            },
            "api.get_invoice": backend.get_invoice,
            "api.mark_invoice_paid": backend.mark_invoice_paid,
            "api.set_customer_status": backend.set_customer_status,
            "api.create_ticket": backend.create_ticket,
        }
        super().__init__(spec, tools)

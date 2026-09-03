"""Web Agent — drives the mock legacy web console.

Wraps WebApp navigation (login -> open account -> read/set fields) as capabilities.
In production this is where Playwright would drive a real browser; the WebApp
state machine stands in so the whole system runs headless and deterministically.
Swapping in `PlaywrightWebApp` (same method surface) is the only change needed to
drive a real Flask app — see README's "real browser" note.
"""

from __future__ import annotations

from ..mocks.web_app import WebApp
from ..types import AgentSpec, ToolSchema
from .base import ToolAgent

SCHEMAS = [
    ToolSchema("web.login", "Log into the legacy console",
               {"user": "str", "password": "str"}),
    ToolSchema("web.open_account", "Open an account record page",
               {"account_id": "int"}),
    ToolSchema("web.read_status", "Read the status field on the open account",
               {"account_id": "int"}),
    ToolSchema("web.set_status", "Set the status field on the open account",
               {"account_id": "int", "status": "str"}, irreversible=True),
    ToolSchema("web.add_note", "Append a note to the open account",
               {"account_id": "int", "text": "str"}, irreversible=True),
    ToolSchema("web.read_notes", "Read notes on the open account",
               {"account_id": "int"}),
]


class WebAgent(ToolAgent):
    kind = "web"

    def __init__(self, app: WebApp) -> None:
        self.app = app
        spec = AgentSpec(
            name="web_agent",
            kind="web",
            description="Drives the legacy web console: login, account status, notes.",
            capabilities=[s.name for s in SCHEMAS],
            tool_schemas=SCHEMAS,
            system_prompt=(
                "You operate an old web console by navigating and filling fields. "
                "Reads may be eventually consistent; verify writes by re-reading."
            ),
        )
        tools = {
            "web.login": app.login,
            "web.open_account": app.open_account,
            "web.read_status": lambda account_id: {"status": app.read_status(account_id)},
            "web.set_status": app.set_status,
            "web.add_note": app.add_note,
            "web.read_notes": lambda account_id: {"notes": app.read_notes(account_id)},
        }
        super().__init__(spec, tools)

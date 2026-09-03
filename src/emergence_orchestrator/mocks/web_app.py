"""Mock legacy web CRUD app — the 'legacy' system, modeled as a state machine.

Represents an old customer-onboarding console the Web Agent must click through:
log in, open an account record, read/set a status field, add notes. It keeps its
own data store, distinct from the enterprise API, so cross-system tasks are real
integration (read in one system, act in the other).

Legacy realism — eventual consistency: when `flaky=True`, the first read-back
after a write returns the STALE value once (a caching artifact). This is
deliberate: it gives the verifier something real to catch, and a retry to
recover from. Set `flaky=False` for a fully deterministic, always-passing run.
"""

from __future__ import annotations

from typing import Any

VALID_STATUSES = {"prospect", "onboarding", "active", "suspended"}


class WebAppError(Exception):
    pass


# account_id -> record. Mirrors the API's customer ids so tasks can join them.
SEED_ACCOUNTS = {
    1: {"name": "Meridian Supply Co", "status": "active", "notes": []},
    2: {"name": "Cobalt Systems", "status": "onboarding", "notes": []},
    3: {"name": "Harbor Analytics", "status": "suspended", "notes": []},
    4: {"name": "Vantage Health", "status": "prospect", "notes": []},
    5: {"name": "Delta Logistics", "status": "active", "notes": []},
}


class WebApp:
    def __init__(self, flaky: bool = True) -> None:
        self.flaky = flaky
        self.logged_in = False
        self.current: int | None = None
        self.accounts: dict[int, dict[str, Any]] = {
            k: {**v, "notes": list(v["notes"])} for k, v in SEED_ACCOUNTS.items()
        }
        self._pending_stale: dict[int, str] = {}  # account_id -> stale status to serve once

    # -- navigation -------------------------------------------------------- #
    def login(self, user: str, password: str) -> dict[str, Any]:
        if not user or not password:
            raise WebAppError("login requires user and password")
        self.logged_in = True
        return {"logged_in": True, "user": user}

    def _require_login(self) -> None:
        if not self.logged_in:
            raise WebAppError("not logged in")

    def open_account(self, account_id: int) -> dict[str, Any]:
        self._require_login()
        if account_id not in self.accounts:
            raise WebAppError(f"account {account_id} not found")
        self.current = account_id
        return {"account_id": account_id, "name": self.accounts[account_id]["name"]}

    def _require_open(self, account_id: int) -> None:
        self._require_login()
        if self.current != account_id:
            raise WebAppError(
                f"account {account_id} not open (current={self.current})"
            )

    # -- fields ------------------------------------------------------------ #
    def read_status(self, account_id: int) -> str:
        self._require_open(account_id)
        if account_id in self._pending_stale:
            return self._pending_stale.pop(account_id)  # serve stale once
        return self.accounts[account_id]["status"]

    def set_status(self, account_id: int, status: str) -> dict[str, Any]:
        self._require_open(account_id)
        if status not in VALID_STATUSES:
            raise WebAppError(f"invalid status {status!r}")
        old = self.accounts[account_id]["status"]
        self.accounts[account_id]["status"] = status
        # The write commits, but the confirmation the agent reads back may lag.
        observed = status
        if self.flaky and old != status:
            self._pending_stale[account_id] = old
            observed = old
        return {"account_id": account_id, "status": observed, "committed": status}

    def add_note(self, account_id: int, text: str) -> dict[str, Any]:
        self._require_open(account_id)
        self.accounts[account_id]["notes"].append(text)
        return {"account_id": account_id, "note_count": len(self.accounts[account_id]["notes"])}

    def read_notes(self, account_id: int) -> list[str]:
        self._require_open(account_id)
        return list(self.accounts[account_id]["notes"])

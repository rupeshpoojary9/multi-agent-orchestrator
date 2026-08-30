"""Mock enterprise API over SQLite — the 'modern' system.

A tiny billing/support backend seeded with customers, invoices, and tickets.
The API Agent calls these methods as typed REST-style tools. Using real SQLite
(in-memory by default) keeps the data layer honest: transactions, constraints,
and a persistence boundary the agent cannot reach around.
"""

from __future__ import annotations

import sqlite3
from typing import Any


class APIError(Exception):
    pass


SEED_CUSTOMERS = [
    (1, "Northwind Traders", "billing@northwind.example", "active"),
    (2, "Globex Corp", "ap@globex.example", "active"),
    (3, "Initech", "finance@initech.example", "delinquent"),
    (4, "Umbrella Health", "accounts@umbrella.example", "active"),
    (5, "Wayne Enterprises", "treasury@wayne.example", "active"),
]

# (id, customer_id, amount_cents, status)
SEED_INVOICES = [
    (101, 1, 250000, "open"),
    (102, 1, 90000, "paid"),
    (103, 2, 500000, "open"),
    (104, 3, 120000, "overdue"),
    (105, 3, 75000, "overdue"),
    (106, 4, 30000, "open"),
    (107, 5, 1000000, "open"),
]


class APIBackend:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self._init_schema()
        self._seed()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE invoices (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                amount_cents INTEGER NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open'
            );
            """
        )
        self.db.commit()

    def _seed(self) -> None:
        self.db.executemany("INSERT INTO customers VALUES (?,?,?,?)", SEED_CUSTOMERS)
        self.db.executemany("INSERT INTO invoices VALUES (?,?,?,?)", SEED_INVOICES)
        self.db.commit()

    # -- read tools -------------------------------------------------------- #
    def list_customers(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM customers ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def get_customer(self, customer_id: int) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT * FROM customers WHERE id=?", (customer_id,)
        ).fetchone()
        if row is None:
            raise APIError(f"customer {customer_id} not found")
        return dict(row)

    def find_customer_by_name(self, name: str) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT * FROM customers WHERE lower(name)=lower(?)", (name,)
        ).fetchone()
        if row is None:
            raise APIError(f"customer named {name!r} not found")
        return dict(row)

    def list_invoices(
        self, customer_id: int, status: str | None = None
    ) -> list[dict[str, Any]]:
        self.get_customer(customer_id)  # validate
        if status:
            rows = self.db.execute(
                "SELECT * FROM invoices WHERE customer_id=? AND status=?",
                (customer_id, status),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM invoices WHERE customer_id=?", (customer_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_invoice(self, invoice_id: int) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT * FROM invoices WHERE id=?", (invoice_id,)
        ).fetchone()
        if row is None:
            raise APIError(f"invoice {invoice_id} not found")
        return dict(row)

    # -- write tools (irreversible) --------------------------------------- #
    def mark_invoice_paid(self, invoice_id: int) -> dict[str, Any]:
        inv = self.get_invoice(invoice_id)
        if inv["status"] == "paid":
            return inv  # idempotent
        self.db.execute(
            "UPDATE invoices SET status='paid' WHERE id=?", (invoice_id,)
        )
        self.db.commit()
        return self.get_invoice(invoice_id)

    def set_customer_status(self, customer_id: int, status: str) -> dict[str, Any]:
        self.get_customer(customer_id)
        self.db.execute(
            "UPDATE customers SET status=? WHERE id=?", (status, customer_id)
        )
        self.db.commit()
        return self.get_customer(customer_id)

    def create_ticket(
        self, customer_id: int, subject: str, body: str
    ) -> dict[str, Any]:
        self.get_customer(customer_id)
        cur = self.db.execute(
            "INSERT INTO tickets (customer_id, subject, body) VALUES (?,?,?)",
            (customer_id, subject, body),
        )
        self.db.commit()
        row = self.db.execute(
            "SELECT * FROM tickets WHERE id=?", (cur.lastrowid,)
        ).fetchone()
        return dict(row)

    def get_ticket(self, ticket_id: int) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT * FROM tickets WHERE id=?", (ticket_id,)
        ).fetchone()
        if row is None:
            raise APIError(f"ticket {ticket_id} not found")
        return dict(row)

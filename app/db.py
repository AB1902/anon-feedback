"""SQLite storage. By design there is NO column for name, email, IP, device, original
text or exact timestamp. Time is stored only as an ISO week ("2026-W43")."""
from __future__ import annotations

import datetime as dt
import json
import secrets
import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    ticket_id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'live',          -- live | seed (demo data)
    department TEXT NOT NULL,                     -- or the pooled senior-leadership group
    team TEXT,                                    -- NULL for managers and senior leaders
    level_band TEXT NOT NULL,
    week TEXT NOT NULL,
    text TEXT NOT NULL,                           -- anonymised, rewritten text only
    summary TEXT, category TEXT, sentiment TEXT, severity INTEGER,
    route TEXT, route_reason TEXT, urgent INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'submitted',
    explanation TEXT, suggested_action TEXT, citations TEXT, grounded INTEGER,
    rating TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL, sender TEXT NOT NULL, label TEXT, body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week TEXT, role TEXT, action TEXT, ticket_id TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS blocked (            -- reason only; blocked text is never stored
    id INTEGER PRIMARY KEY AUTOINCREMENT, week TEXT, reason TEXT, layer TEXT
);
CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT);
"""

STATUSES = ["submitted", "acknowledged", "in_review", "action_taken", "closed"]
STATUS_LABELS = {"submitted": "Submitted", "acknowledged": "Acknowledged", "in_review": "In review",
                 "action_taken": "Action taken", "closed": "Closed"}


def this_week(today: dt.date | None = None) -> str:
    y, w, _ = (today or dt.date.today()).isocalendar()
    return f"{y}-W{w:02d}"


def week_to_month(week: str) -> str:
    y, w = week.split("-W")
    return dt.date.fromisocalendar(int(y), int(w), 1).strftime("%Y-%m")


def weeks_between(a: str, b: str) -> int:
    pa, pb = (dt.date.fromisocalendar(int(x.split("-W")[0]), int(x.split("-W")[1]), 1) for x in (a, b))
    return (pb - pa).days // 7


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.executescript(SCHEMA)


def new_ticket_id() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I confusion
    with conn() as c:
        while True:
            t = "TKT-" + "".join(secrets.choice(alphabet) for _ in range(6))
            if not c.execute("SELECT 1 FROM submissions WHERE ticket_id=?", (t,)).fetchone():
                return t


def insert_submission(row: dict) -> None:
    row = {**row}
    if isinstance(row.get("citations"), list):
        row["citations"] = json.dumps(row["citations"])
    cols = ", ".join(row)
    with conn() as c:
        c.execute(f"INSERT INTO submissions ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))


def update_submission(ticket_id: str, **fields) -> None:
    if "citations" in fields and isinstance(fields["citations"], list):
        fields["citations"] = json.dumps(fields["citations"])
    sets = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE submissions SET {sets} WHERE ticket_id=?", [*fields.values(), ticket_id])


def get_submission(ticket_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM submissions WHERE ticket_id=?", (ticket_id.strip().upper(),)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["citations"] = json.loads(d["citations"]) if d.get("citations") else []
    return d


def add_message(ticket_id: str, sender: str, body: str, label: str = "") -> None:
    with conn() as c:
        c.execute("INSERT INTO messages (ticket_id, sender, label, body) VALUES (?,?,?,?)",
                  (ticket_id, sender, label, body))


def messages(ticket_id: str) -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM messages WHERE ticket_id=? ORDER BY id", (ticket_id,))]


def log(role: str, action: str, ticket_id: str = "", detail: str = "") -> None:
    with conn() as c:
        c.execute("INSERT INTO audit (week, role, action, ticket_id, detail) VALUES (?,?,?,?,?)",
                  (this_week(), role, action, ticket_id, detail))


def log_blocked(reason: str, layer: str) -> None:
    with conn() as c:
        c.execute("INSERT INTO blocked (week, reason, layer) VALUES (?,?,?)", (this_week(), reason, layer))


def query(sql: str, params=()) -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute(sql, params)]


def cache_get(key: str) -> str | None:
    with conn() as c:
        r = c.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
    return r[0] if r else None


def cache_set(key: str, value: str) -> None:
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?,?)", (key, value))

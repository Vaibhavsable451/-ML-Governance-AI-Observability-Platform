"""SQLite-backed registry: model profiles, evaluations timeline, human reviews, events, LLM usage.
For multi-replica production swap this module for RDS/Postgres; the function signatures stay the same."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager

from core import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (model_id TEXT PRIMARY KEY, status TEXT, profile TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS evaluations (id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT, ts REAL, risk REAL, profile TEXT);
CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT, ts REAL, decision TEXT, reviewer TEXT, comment TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, type TEXT, severity TEXT, model_id TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, tokens_in INTEGER, tokens_out INTEGER, cost REAL);
"""


@contextmanager
def _db():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _db() as c:
        c.executescript(SCHEMA)


def save_profile(profile: dict) -> None:
    now = time.time()
    with _db() as c:
        c.execute("INSERT INTO models(model_id,status,profile,updated) VALUES(?,?,?,?) "
                  "ON CONFLICT(model_id) DO UPDATE SET status=excluded.status, profile=excluded.profile, updated=excluded.updated",
                  (profile["model_id"], profile["status"], json.dumps(profile), now))
        c.execute("INSERT INTO evaluations(model_id,ts,risk,profile) VALUES(?,?,?,?)",
                  (profile["model_id"], now, profile["risk"]["score"], json.dumps(profile)))


def get_profile(model_id: str) -> dict | None:
    with _db() as c:
        row = c.execute("SELECT profile, status FROM models WHERE model_id=?", (model_id,)).fetchone()
    if not row:
        return None
    p = json.loads(row["profile"])
    p["status"] = row["status"]
    return p


def list_profiles() -> list[dict]:
    with _db() as c:
        rows = c.execute("SELECT profile, status FROM models ORDER BY updated DESC").fetchall()
    out = []
    for r in rows:
        p = json.loads(r["profile"])
        p["status"] = r["status"]
        out.append(p)
    return out


def set_status(model_id: str, status: str) -> None:
    p = get_profile(model_id)
    if p:
        p["status"] = status
        with _db() as c:
            c.execute("UPDATE models SET status=?, profile=?, updated=? WHERE model_id=?",
                      (status, json.dumps(p), time.time(), model_id))


def list_evaluations(model_id: str, limit: int = 50) -> list[dict]:
    with _db() as c:
        rows = c.execute("SELECT ts, risk FROM evaluations WHERE model_id=? ORDER BY id DESC LIMIT ?",
                         (model_id, limit)).fetchall()
    return [{"ts": r["ts"], "risk": r["risk"]} for r in reversed(rows)]


def add_review(model_id: str, decision: str, reviewer: str, comment: str) -> None:
    with _db() as c:
        c.execute("INSERT INTO reviews(model_id,ts,decision,reviewer,comment) VALUES(?,?,?,?,?)",
                  (model_id, time.time(), decision, reviewer, comment))


def list_reviews(model_id: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM reviews", ()
    if model_id:
        q, args = q + " WHERE model_id=?", (model_id,)
    with _db() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY id DESC LIMIT 100", args).fetchall()]


def log_event(type_: str, severity: str, payload: dict, model_id: str | None = None) -> None:
    with _db() as c:
        c.execute("INSERT INTO events(ts,type,severity,model_id,payload) VALUES(?,?,?,?,?)",
                  (time.time(), type_, severity, model_id, json.dumps(payload, default=str)))


def list_events(limit: int = 100, type_: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM events", []
    if type_:
        q += " WHERE type=?"
        args.append(type_)
    with _db() as c:
        rows = c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
    return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]


def security_risk() -> float:
    """Share of the last 200 chat/scan requests that were blocked or contained PII, scaled to 0-100."""
    ev = list_events(200)
    reqs = [e for e in ev if e["type"] in ("chat", "security")]
    if not reqs:
        return 0.0
    blocked = sum(1 for e in reqs if e["type"] == "security" and e["payload"].get("blocked"))
    pii = sum(1 for e in reqs if e["payload"].get("pii_types"))
    return round(min(100.0, (blocked * 2.0 + pii * 0.5) / len(reqs) * 100), 1)


def avg_hallucination() -> float:
    vals = [e["payload"]["hallucination_score"] for e in list_events(100, "chat")
            if "hallucination_score" in e["payload"]]
    return round(sum(vals) / len(vals), 1) if vals else 10.0


def log_usage(kind: str, tokens_in: int, tokens_out: int, cost: float) -> None:
    with _db() as c:
        c.execute("INSERT INTO usage(ts,kind,tokens_in,tokens_out,cost) VALUES(?,?,?,?,?)",
                  (time.time(), kind, tokens_in, tokens_out, cost))


def usage_summary(days: int = 30) -> dict:
    since = time.time() - days * 86400
    with _db() as c:
        r = c.execute("SELECT COUNT(*) n, COALESCE(SUM(tokens_in),0) ti, COALESCE(SUM(tokens_out),0) to_, "
                      "COALESCE(SUM(cost),0) cost FROM usage WHERE ts>=?", (since,)).fetchone()
        daily = c.execute("SELECT date(ts,'unixepoch') d, SUM(cost) c FROM usage WHERE ts>=? GROUP BY d ORDER BY d",
                          (since,)).fetchall()
    return {"requests": r["n"], "tokens_in": r["ti"], "tokens_out": r["to_"], "cost_usd": round(r["cost"], 6),
            "budget_usd": config.MONTHLY_BUDGET_USD, "daily": [{"date": d["d"], "cost": d["c"]} for d in daily]}

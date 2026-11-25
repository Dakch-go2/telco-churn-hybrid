"""SQLite журналы: әр сұрау мен шешім жазылады."""
import json, os, sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def db_path() -> str:
    return os.environ.get("CHURN_DB", str(ROOT / "logs" / "decisions.db"))


def _conn():
    p = db_path()
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    c.execute("""CREATE TABLE IF NOT EXISTS decisions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, endpoint TEXT, input TEXT,
        decision TEXT, source TEXT, probability REAL, response TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, decision_id INTEGER, agrees INTEGER, comment TEXT)""")
    return c


def log_decision(endpoint, client, result) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO decisions(ts,endpoint,input,decision,source,probability,response) VALUES(?,?,?,?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(), endpoint, json.dumps(client, ensure_ascii=False),
                         result.get("decision"), result.get("source"), result.get("probability"),
                         json.dumps(result, ensure_ascii=False)))
        return cur.lastrowid


def last_decisions(n=10):
    with _conn() as c:
        rows = c.execute("SELECT id,ts,endpoint,decision,source,probability FROM decisions ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    return [dict(zip(["id", "ts", "endpoint", "decision", "source", "probability"], r)) for r in rows]


def stats(n=10):
    rows = last_decisions(n)
    counts = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    probs = [r["probability"] for r in rows if r["probability"] is not None]
    return {"last_n": len(rows), "by_decision": counts,
            "mean_probability": round(sum(probs) / len(probs), 4) if probs else None}


def save_feedback(decision_id, agrees, comment):
    with _conn() as c:
        exists = c.execute("SELECT 1 FROM decisions WHERE id=?", (decision_id,)).fetchone()
        if not exists:
            return None
        cur = c.execute("INSERT INTO feedback(ts,decision_id,agrees,comment) VALUES(?,?,?,?)",
                        (datetime.now(timezone.utc).isoformat(), decision_id, int(agrees), comment))
        return cur.lastrowid

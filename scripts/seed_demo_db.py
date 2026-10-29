"""Loads the synthetic feedback into the database so dashboards have six months of data.

    python -m scripts.seed_demo_db            # resets feedback.db and loads all notes

To save API quota, seed notes use the RULE layer only (PII scrub + generic wording) and their
ground-truth labels; they are marked source='seed'. Every note you submit in the app goes
through the full AI pipeline. Policy explanations for seed cases are generated on demand
when a reviewer opens the case.
"""
import json
import random

from app import db
from app.config import DATA_DIR, DB_PATH, ESCALATION_ROUTES, SENIOR_POOL
from app.pipeline.privacy import GENERIC, PLACEHOLDER, scrub


def main() -> None:
    src = DATA_DIR / "synthetic" / "feedback_raw.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if DB_PATH.exists():
        DB_PATH.unlink()
    db.init()
    rng = random.Random(5)
    weeks = sorted({r["week"] for r in rows})
    for r in rows:
        clean = PLACEHOLDER.sub(lambda m: GENERIC[m.group(1)], scrub(r["text"]).text)
        age = len(weeks) - 1 - weeks.index(r["week"])
        status = ("closed" if age > 8 else rng.choice(["acknowledged", "in_review", "action_taken", "closed"])
                  if age > 2 else rng.choice(["submitted", "submitted", "acknowledged"]))
        dept, team = r["department"], r["team"]
        if r["level_band"] == "senior_leadership":
            dept, team = SENIOR_POOL, None
        elif r["level_band"] == "manager":
            team = None
        ticket = db.new_ticket_id()
        db.insert_submission({
            "ticket_id": ticket, "source": "seed", "department": dept, "team": team,
            "level_band": r["level_band"], "week": r["week"], "text": clean,
            "summary": None, "category": r["category"], "sentiment": r["sentiment"],
            "severity": r["severity"], "route": ESCALATION_ROUTES.get(r["category"], "hr"),
            "route_reason": f"category: {r['category']}", "urgent": int(r["severity"] >= 5),
            "status": status})
        db.add_message(ticket, "system", clean, "Your feedback, as reviewers see it")
    print(f"Loaded {len(rows)} demo notes into {DB_PATH.name} across {len(weeks)} weeks.")


if __name__ == "__main__":
    main()

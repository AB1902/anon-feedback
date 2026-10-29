"""Generates labelled synthetic employee feedback with an LLM.

How it works
  1. Build a deterministic grid of "specs": department x level x category x
     sentiment x severity x writing style, plus fake PII to plant in the text.
     The spec IS the ground-truth label, so we can measure classifier accuracy
     and PII-scrubber recall later.
  2. Send specs to the LLM in small batches; it writes one feedback text per spec.
  3. Check each planted PII string actually appears in the text and record it.

Run from the project root:
    python -m scripts.generate_feedback --dry-run          # show one prompt, no API call
    python -m scripts.generate_feedback --n 40             # small test (~1 min)
    python -m scripts.generate_feedback --n 1500           # full set (~15-20 min)

Safe to stop (Ctrl+C) and re-run: finished items are skipped.
Output: data/synthetic/feedback_raw.jsonl
"""
import argparse
import datetime as dt
import json
import random
import time

from pydantic import BaseModel

from app.config import DATA_DIR, ESCALATION_ROUTES

SEED = 7
OUT = DATA_DIR / "synthetic" / "feedback_raw.jsonl"
ANCHOR_MONDAY = dt.date(2026, 10, 19)   # fixed so specs are identical on every run
N_WEEKS = 26

CATEGORY_WEIGHTS = {
    "compensation": .14, "manager_behaviour": .14, "workload": .14, "culture": .11,
    "growth": .11, "other": .11, "harassment": .08, "safety": .06, "fraud": .06,
    "hr_conduct": .05,
}
CATEGORY_DESC = {
    "compensation": "salary, increments, bonuses, sales incentives, payroll delays, pay fairness",
    "manager_behaviour": "a manager's conduct: micromanagement, shouting, favouritism, poor feedback, "
                         "ignoring concerns (NOT sexual in nature)",
    "workload": "too much work, unrealistic deadlines, long hours, late-night calls, understaffing",
    "culture": "team culture, inclusion, recognition, communication, cliques, morale",
    "growth": "learning, promotions, career path, ratings, internal mobility",
    "other": "facilities, cafeteria, IT equipment, commute, office amenities, general suggestions",
    "harassment": "sexual harassment or gender-based harassment: unwelcome comments, messages, "
                  "advances. Describe NON-graphically, as a real HR complaint would",
    "safety": "physical safety: unsafe equipment, blocked exits, missing protective gear, "
              "unsafe late-night transport, electrical hazards",
    "fraud": "fraud or ethics breaches: expense padding, vendor kickbacks, fake invoices, "
             "manipulated reports or data",
    "hr_conduct": "a complaint about the HR team itself: HR ignoring complaints, breaching "
                  "confidentiality, biased handling",
}
STYLES = {
    "formal": "formal, structured, professional English",
    "angry": "angry venting, emotional, may use CAPS or exclamation marks (no slurs)",
    "hinglish": "casual Hinglish (Hindi words in Roman script mixed with English), like a WhatsApp message",
    "short": "one or two short sentences only",
    "rambling": "long and rambling, 5-8 sentences, goes off on small tangents",
    "polite": "polite and hesitant, softens the criticism",
    "sarcastic": "sarcastic and resigned",
}
THEME_HINTS = {
    ("Sales", "compensation"): "incentive payouts delayed or targets changed mid-quarter",
    ("Engineering", "workload"): "late-night release calls and weekend work without comp-off",
    ("Customer Support", "workload"): "understaffed shifts and very high ticket volume",
    ("Operations", "safety"): "warehouse safety gear, blocked exits or unsafe night transport",
    ("Finance", "fraud"): "irregular vendor payments or padded expense claims",
}


class GenItem(BaseModel):
    id: str
    text: str


class GenBatch(BaseModel):
    items: list[GenItem]


# ---------------------------------------------------------------- specs

def iso_weeks() -> list[str]:
    weeks = []
    for i in range(N_WEEKS - 1, -1, -1):
        y, w, _ = (ANCHOR_MONDAY - dt.timedelta(weeks=i)).isocalendar()
        weeks.append(f"{y}-W{w:02d}")
    return weeks


def weighted(rng, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def pick_category(rng, dept: str, week_idx: int) -> str:
    w = dict(CATEGORY_WEIGHTS)
    # Storylines for the dashboard to discover:
    if dept == "Sales" and week_idx >= N_WEEKS - 8:
        w["compensation"] *= 3.5      # incentive-delay crisis in recent weeks
    if dept == "Engineering" and week_idx >= N_WEEKS - 6:
        w["workload"] *= 2.5          # release crunch
    if dept == "Customer Support":
        w["workload"] *= max(0.4, 2.0 - week_idx / 12)   # improving over time
    return weighted(rng, w)


def pick_sentiment_severity(rng, cat: str):
    if cat in ESCALATION_ROUTES:
        return "negative", weighted(rng, {3: .2, 4: .4, 5: .4})
    s = weighted(rng, {"negative": .65, "neutral": .2, "positive": .15})
    if s == "negative":
        return s, weighted(rng, {2: .25, 3: .4, 4: .3, 5: .05})
    return s, (rng.choice([1, 2]) if s == "neutral" else 1)


def pick_pii(rng, company, author_dept, author_team):
    """Choose 0-3 fake identifiers to plant. Returns list of {type, value}."""
    if rng.random() < 0.3:
        return []
    dept_people = [e for e in company["employees"] if e["department"] == author_dept]
    team = next((t for d in company["departments"] if d["name"] == author_dept
                 for t in d["teams"] if t["name"] == author_team), None)
    person = rng.choice(dept_people)
    options = [
        ("PERSON", person["name"]),
        ("PERSON", person["name"].split()[0]),
        ("EMPLOYEE_ID", rng.choice(dept_people)["employee_id"]),
        ("EMAIL", rng.choice(dept_people)["email"]),
        ("PHONE", rng.choice(dept_people)["phone"]),
        ("PROJECT", rng.choice(company["projects"])),
    ]
    if team:
        options.append(("TEAM_LOCATION", f"{team['location']} {team['name']} team"))
        options.append(("PERSON", team["manager"]))
    chosen = rng.sample(options, k=rng.choice([1, 1, 2, 2, 3]))
    out = []
    for t, v in chosen:   # skip values that overlap (e.g. "Deepak" inside "Deepak Bhat")
        if not any(v in p["value"] or p["value"] in v for p in out):
            out.append({"type": t, "value": v})
    return out


def build_specs(n: int) -> list[dict]:
    rng = random.Random(SEED)
    company = json.loads((DATA_DIR / "company.json").read_text(encoding="utf-8"))
    weeks = iso_weeks()
    dept_weights = {d["name"]: d["headcount"] for d in company["departments"]}
    specs = []
    for i in range(n):
        dept = weighted(rng, dept_weights)
        band = weighted(rng, {"ic": .75, "manager": .2, "senior_leadership": .05})
        teams = next(d["teams"] for d in company["departments"] if d["name"] == dept)
        team = None if band == "senior_leadership" else rng.choice(teams)["name"]
        week_idx = rng.randrange(N_WEEKS)
        cat = pick_category(rng, dept, week_idx)
        sentiment, severity = pick_sentiment_severity(rng, cat)
        specs.append({
            "id": f"fb-{i:05d}",
            "department": dept, "team": team, "level_band": band, "week": weeks[week_idx],
            "category": cat, "sentiment": sentiment, "severity": severity,
            "escalate": cat in ESCALATION_ROUTES, "route": ESCALATION_ROUTES.get(cat, "hr"),
            "style": rng.choice(list(STYLES)),
            "theme_hint": THEME_HINTS.get((dept, cat)) if rng.random() < 0.7 else None,
            "pii": pick_pii(rng, company, dept, team),
        })
    return specs


# ---------------------------------------------------------------- LLM

SYSTEM = (
    "You write realistic SYNTHETIC internal employee feedback for a fictional Indian "
    "company, Nimbus Technologies, to test an HR analytics system. Each item must match "
    "its spec exactly: topic (category), sentiment, severity (1 = minor/positive, "
    "5 = very serious), writer's role, and writing style. Write in first person as the "
    "employee. If a spec lists pii values, include EVERY one of them verbatim (exact "
    "spelling) in a natural way; do not add any other names, IDs, emails or phone numbers. "
    "Vary vocabulary and openings; do not start every item the same way. "
    'Return JSON only: {"items": [{"id": "...", "text": "..."}]}'
)


def batch_prompt(batch: list[dict]) -> str:
    lines = []
    for s in batch:
        role = {"ic": "individual contributor", "manager": "team manager",
                "senior_leadership": "senior leader (VP/director)"}[s["level_band"]]
        where = f"{s['team']} team, {s['department']}" if s["team"] else s["department"]
        spec = {
            "id": s["id"], "writer": f"{role} in {where}",
            "category": f"{s['category']} ({CATEGORY_DESC[s['category']]})",
            "sentiment": s["sentiment"], "severity": s["severity"],
            "style": STYLES[s["style"]],
            "pii_to_include": [p["value"] for p in s["pii"]],
        }
        if s["theme_hint"]:
            spec["theme"] = s["theme_hint"]
        lines.append(json.dumps(spec, ensure_ascii=False))
    return "Write one feedback item for each spec below.\n\n" + "\n".join(lines)


def call_llm(batch: list[dict], mock: bool) -> dict[str, str]:
    if mock:   # offline stand-in so the rest of the pipeline can be tested without a key
        return {s["id"]: f"[mock {s['category']}/{s['sentiment']}] "
                         + " and ".join(p["value"] for p in s["pii"]) for s in batch}
    from app.llm import generate_json
    result = generate_json(SYSTEM, batch_prompt(batch), GenBatch)
    return {it.id: it.text.strip() for it in result.items}


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="total items")
    ap.add_argument("--batch", type=int, default=8, help="items per LLM call")
    ap.add_argument("--sleep", type=float, default=4.5, help="seconds between calls (free-tier RPM)")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    ap.add_argument("--mock", action="store_true", help="no API calls; placeholder text")
    args = ap.parse_args()

    specs = build_specs(args.n)
    if args.dry_run:
        print("SYSTEM:\n" + SYSTEM + "\n\nUSER:\n" + batch_prompt(specs[:args.batch]))
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [s for s in specs if s["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to generate.")

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    written = failed = 0
    with OUT.open("a", encoding="utf-8") as f:
        for bi, batch in enumerate(batches, 1):
            try:
                texts = call_llm(batch, args.mock)
            except Exception as e:
                failed += len(batch)
                print(f"  batch {bi}/{len(batches)} failed: {str(e)[:160]}  (re-run to retry)")
                time.sleep(args.sleep * 2)
                continue
            for s in batch:
                text = texts.get(s["id"])
                if not text:
                    failed += 1
                    continue
                low = text.lower()
                found = [p for p in s["pii"] if p["value"].lower() in low]
                missing = [p for p in s["pii"] if p["value"].lower() not in low]
                rec = {**s, "text": text, "pii": found, "pii_missing": missing}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            print(f"  batch {bi}/{len(batches)} ok  (total written this run: {written})")
            if not args.mock and bi < len(batches):
                time.sleep(args.sleep)

    print(f"\nDone. Written {written}, failed {failed}. Output: {OUT}")
    if failed:
        print("Re-run the same command to fill in the failed items.")


if __name__ == "__main__":
    main()

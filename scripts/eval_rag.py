"""Evaluates policy retrieval (free) and, optionally, grounding of AI explanations.

    python -m scripts.eval_rag                 # retrieval hit rate on all test notes, no API calls
    python -m scripts.eval_rag --llm --limit 20  # + share of explanations citing only retrieved clauses

Retrieval hit = at least one of the top 4 retrieved clauses comes from the handbook section
that governs the note's category (e.g. compensation -> section 7).
"""
import argparse
import json
import time

from app.config import DATA_DIR
from app.pipeline.privacy import GENERIC, PLACEHOLDER, scrub
from app.pipeline.rag import CATEGORY_QUERY_HINTS, explain
from app.pipeline.retrieval import handbook_index, retrieve_policy

EXPECTED = {
    "compensation": ["7."], "manager_behaviour": ["2.2", "2.3", "9."], "workload": ["5."],
    "harassment": ["3."], "culture": ["2.", "3.7", "12."], "growth": ["8."],
    "safety": ["11.", "6.5"], "fraud": ["10.", "2.4", "2.5"], "hr_conduct": ["9.5", "9."],
    "other": ["14.", "6.", "11.5"],
}


def hit(ids, cat):
    return any(i.startswith(p) or i == p for i in ids for p in EXPECTED[cat])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=4.5)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (DATA_DIR / "synthetic" / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Retrieval backend: {handbook_index().backend}")

    # Two settings: the note text alone, and text + the category hint the pipeline adds.
    # (Uses the gold category, i.e. assumes triage was correct; triage accuracy is reported separately.)
    hits_text, hits, per = 0, 0, {}
    for r in rows:
        text = PLACEHOLDER.sub(lambda m: GENERIC[m.group(1)], scrub(r["text"]).text)
        hits_text += hit([d.id for d in retrieve_policy(text, 4)], r["category"])
        ids = [d.id for d in retrieve_policy(f"{text}\n{CATEGORY_QUERY_HINTS[r['category']]}", 4)]
        h = hit(ids, r["category"])
        hits += h
        a, b = per.get(r["category"], (0, 0))
        per[r["category"]] = (a + h, b + 1)
    print(f"Retrieval hit rate, top 4, note text only:        {hits_text / len(rows):.1%}")
    print(f"Retrieval hit rate, top 4, text + category hint:  {hits / len(rows):.1%}  ({len(rows)} test notes)")
    for c, (a, b) in sorted(per.items()):
        print(f"  {c:18s} {a:3d}/{b:<3d} {a / b:5.0%}")
    out = {"backend": handbook_index().backend, "notes": len(rows),
           "retrieval_hit_rate_text_only": round(hits_text / len(rows), 4),
           "retrieval_hit_rate": round(hits / len(rows), 4),
           "per_category": {c: round(a / b, 3) for c, (a, b) in per.items()}}

    if args.llm:
        grounded_n, cited_n, n = 0, 0, 0
        for r in rows[:args.limit]:
            text = PLACEHOLDER.sub(lambda m: GENERIC[m.group(1)], scrub(r["text"]).text)
            try:
                exp, docs, ok = explain(text, r["category"], r["route"])
            except Exception as e:
                print(f"  {r['id']} failed: {str(e)[:80]}")
                continue
            n += 1
            grounded_n += ok
            cited_n += bool(exp.citations)
            time.sleep(args.sleep)
        print(f"\nExplanations: {n}; with citations: {cited_n}; all citations grounded in retrieved clauses: {grounded_n} ({grounded_n / max(n, 1):.0%})")
        out.update({"explanations": n, "grounded_rate": round(grounded_n / max(n, 1), 4)})

    (DATA_DIR / "eval").mkdir(exist_ok=True)
    (DATA_DIR / "eval" / "rag_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved data/eval/rag_results.json")


if __name__ == "__main__":
    main()

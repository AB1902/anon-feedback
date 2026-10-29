"""Collects every evaluation result into one table for the report.

    python -m scripts.report_results
Prints a Markdown table and writes data/eval/results_summary.csv (opens in Excel).
"""
import csv
import json

from app.config import DATA_DIR

E = DATA_DIR / "eval"


def load(name):
    p = E / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x):
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "not run"


def main() -> None:
    rows = []
    t = load("triage_results.json")
    if t:
        n = t["notes"]
        rows += [("Triage", "Category accuracy", pct(t["category_accuracy"]), f"{n} held-out notes"),
                 ("Triage", "Category macro-F1", f"{t['macro_f1']:.3f}", f"{n} held-out notes"),
                 ("Triage", "Severity within ±1", pct(t["severity_within_1"]), f"{n} held-out notes"),
                 ("Routing", "Escalation recall (POSH/Ethics/Panel)", pct(t["escalation_recall"]), f"{n} held-out notes"),
                 ("Routing", "Routing accuracy", pct(t["routing_accuracy"]), f"{n} held-out notes")]
    for name, label in (("privacy_test_rules.json", "rules + Presidio"), ("privacy_test_llm.json", "rules + AI rewrite")):
        p = load(name)
        if p:
            rows.append(("Privacy", f"Planted identifiers removed ({label})", pct(p["overall_recall"]),
                         f"{p['notes']} test notes"))
    for name in ("guardrails_results.json", "guardrails_results_llm.json"):
        g = load(name)
        if g:
            rows += [("Guardrails", f"Attack block rate ({g['mode']})", pct(g["attack_block_rate"]), f"{g['inputs']} adversarial inputs"),
                     ("Guardrails", f"False-positive rate ({g['mode']})", pct(g["false_positive_rate"]), "23 harmless look-alikes")]
    r = load("rag_results.json")
    if r:
        n = r.get("notes", 200)
        if "retrieval_hit_rate_text_only" in r:
            rows.append(("RAG", f"Retrieval hit rate, top 4, text only ({r['backend']})", pct(r["retrieval_hit_rate_text_only"]), f"{n} test notes"))
        rows.append(("RAG", f"Retrieval hit rate, top 4, with category ({r['backend']})", pct(r["retrieval_hit_rate"]), f"{n} test notes"))
        if "grounded_rate" in r:
            rows.append(("RAG", "Explanations citing only retrieved clauses", pct(r["grounded_rate"]), f"{r['explanations']} explanations"))
    if not rows:
        print("No results yet. Run the eval scripts first.")
        return
    print("| Component | Metric | Result | Evaluated on |\n|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    with (E / "results_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Component", "Metric", "Result", "Evaluated on"])
        w.writerows(rows)
    print("\nSaved data/eval/results_summary.csv")


if __name__ == "__main__":
    main()

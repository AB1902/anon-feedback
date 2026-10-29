"""Evaluates AI triage + routing on the held-out test notes.

    python -m scripts.eval_triage --limit 100      # ~8 min on the free tier
    python -m scripts.eval_triage                  # all 200 test notes, ~15 min

Metrics: category accuracy and macro-F1, sentiment accuracy, severity exact and within +/-1,
escalation recall/precision (did harassment/safety/fraud/HR complaints leave the normal HR queue?).
Safe to stop and re-run: finished notes are cached in data/eval/triage_predictions.jsonl.
"""
import argparse
import collections
import json
import time

from app.config import DATA_DIR
from app.pipeline.privacy import scrub
from app.pipeline.triage import analyse, route

OUT = DATA_DIR / "eval"


def f1_table(gold, pred, labels):
    rows, f1s = [], []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        rows.append((c, sum(g == c for g in gold), prec, rec, f1))
        f1s.append(f1)
    return rows, sum(f1s) / len(f1s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=4.5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (DATA_DIR / "synthetic" / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        # keep the category mix when sampling
        by = collections.defaultdict(list)
        for r in rows:
            by[r["category"]].append(r)
        rows, i = [], 0
        while len(rows) < args.limit and any(by.values()):
            for c in sorted(by):
                if by[c] and len(rows) < args.limit:
                    rows.append(by[c].pop(0))
    OUT.mkdir(exist_ok=True)
    cache_path = OUT / "triage_predictions.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {json.loads(l)["id"]: json.loads(l) for l in cache_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo = [r for r in rows if r["id"] not in cache]
    print(f"{len(rows)} notes; {len(rows) - len(todo)} cached, {len(todo)} to run.")
    with cache_path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            s = scrub(r["text"]).text
            try:
                a = analyse(s)
            except Exception as e:
                print(f"  {r['id']} failed: {str(e)[:100]}")
                time.sleep(args.sleep * 2)
                continue
            rt, _ = route(a.category, s)
            rec = {"id": r["id"], "gold_category": r["category"], "pred_category": a.category,
                   "gold_sentiment": r["sentiment"], "pred_sentiment": a.sentiment,
                   "gold_severity": r["severity"], "pred_severity": a.severity,
                   "gold_route": r["route"], "pred_route": rt, "flags": a.flags.model_dump()}
            cache[r["id"]] = rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}")
            time.sleep(args.sleep)

    P = [cache[r["id"]] for r in rows if r["id"] in cache]
    n = len(P)
    if not n:
        print("No predictions yet.")
        return
    labels = sorted({p["gold_category"] for p in P})
    gold = [p["gold_category"] for p in P]
    pred = [p["pred_category"] for p in P]
    table, macro = f1_table(gold, pred, labels)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    sent = sum(p["gold_sentiment"] == p["pred_sentiment"] for p in P) / n
    sev_exact = sum(p["gold_severity"] == p["pred_severity"] for p in P) / n
    sev_1 = sum(abs(p["gold_severity"] - p["pred_severity"]) <= 1 for p in P) / n
    esc_gold = [p["gold_route"] != "hr" for p in P]
    esc_pred = [p["pred_route"] != "hr" for p in P]
    tp = sum(g and q for g, q in zip(esc_gold, esc_pred))
    esc_rec = tp / sum(esc_gold) if sum(esc_gold) else 0
    esc_prec = tp / sum(esc_pred) if sum(esc_pred) else 0
    route_acc = sum(p["gold_route"] == p["pred_route"] for p in P) / n
    false_flags = sum(any(p["flags"].values()) for p in P)

    print(f"\nTriage results on {n} held-out notes")
    print(f"  Category accuracy      {acc:.1%}    macro-F1 {macro:.3f}")
    print(f"  Sentiment accuracy     {sent:.1%}")
    print(f"  Severity exact         {sev_exact:.1%}    within ±1 {sev_1:.1%}")
    print(f"  Escalation recall      {esc_rec:.1%}    precision {esc_prec:.1%}")
    print(f"  Routing accuracy       {route_acc:.1%}")
    print(f"  Genuine notes wrongly flagged as unsafe: {false_flags}")
    print(f"\n  {'category':18s} {'n':>3s} {'prec':>6s} {'recall':>7s} {'F1':>6s}")
    for c, cnt, pr, rc, f1 in table:
        print(f"  {c:18s} {cnt:3d} {pr:6.2f} {rc:7.2f} {f1:6.2f}")
    conf = collections.Counter((g, p) for g, p in zip(gold, pred) if g != p)
    if conf:
        print("\n  Most common confusions (gold -> predicted):")
        for (g, p), k in conf.most_common(5):
            print(f"    {g} -> {p}: {k}")

    (OUT / "triage_results.json").write_text(json.dumps({
        "notes": n, "category_accuracy": round(acc, 4), "macro_f1": round(macro, 4),
        "sentiment_accuracy": round(sent, 4), "severity_exact": round(sev_exact, 4),
        "severity_within_1": round(sev_1, 4), "escalation_recall": round(esc_rec, 4),
        "escalation_precision": round(esc_prec, 4), "routing_accuracy": round(route_acc, 4),
        "false_safety_flags": false_flags,
        "per_category": [{"category": c, "n": k, "precision": round(a, 3), "recall": round(b, 3), "f1": round(f, 3)}
                         for c, k, a, b, f in table],
        "confusions": [{"gold": g, "pred": p, "count": k} for (g, p), k in conf.most_common(10)],
    }, indent=2), encoding="utf-8")
    print("\nSaved data/eval/triage_results.json")


if __name__ == "__main__":
    main()

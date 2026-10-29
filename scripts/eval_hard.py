"""Scores the baseline, the logistic regression model, the LLM, and LLM + safety net on the
hand-written hard test set.

    python -m scripts.build_hard_test          # once
    python -m scripts.eval_hard                # ~2 min (40 LLM calls, cached, safe to re-run)
    python -m scripts.eval_hard --no-llm       # logistic regression only, instant

Output: data/eval/hard_test_results.json and data/eval/hard_test_predictions.csv
"""
import argparse
import collections
import csv
import json
import time

from app.config import DATA_DIR, ESCALATION_ROUTES
from app.pipeline import classifier
from app.pipeline.privacy import scrub
from app.pipeline.triage import route

EVAL = DATA_DIR / "eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (DATA_DIR / "synthetic" / "hard_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    EVAL.mkdir(exist_ok=True)
    cache_path = EVAL / "hard_llm_predictions.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {json.loads(l)["id"]: json.loads(l) for l in cache_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    preds = []
    for i, r in enumerate(rows, 1):
        s = scrub(r["text"]).text
        ml = classifier.predict(s)
        rec = {**r, "lr_category": ml.category, "lr_conf": round(ml.confidence, 2),
               "lr_escalate_prob": round(ml.escalate_prob, 2), "lr_escalate": ml.escalate}
        if not args.no_llm:
            if r["id"] not in cache:
                from app.pipeline.triage import analyse
                try:
                    a = analyse(s)
                    rt, _ = route(a.category, s)
                    cache[r["id"]] = {"id": r["id"], "llm_category": a.category, "llm_route": rt}
                    with cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(cache[r["id"]]) + "\n")
                except Exception as e:
                    print(f"  {r['id']} LLM failed: {str(e)[:80]} (re-run to retry)")
                time.sleep(args.sleep)
            if r["id"] in cache:
                c = cache[r["id"]]
                rec.update(llm_category=c["llm_category"], llm_escalate=c["llm_route"] != "hr")
                rec["combined_escalate"] = rec["llm_escalate"] or rec["lr_escalate"]
        preds.append(rec)
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}")

    def acc(key, subset):
        s = [p for p in subset if key in p]
        return (sum(p[key] == p["category"] for p in s) / len(s), len(s)) if s else (None, 0)

    def esc(key, subset):
        s = [p for p in subset if key in p]
        pos = [p for p in s if p["escalate"]]
        neg = [p for p in s if not p["escalate"]]
        if not pos:
            return None
        tp = sum(p[key] for p in pos)
        fp = sum(p[key] for p in neg)
        return {"recall": tp / len(pos), "precision": tp / (tp + fp) if tp + fp else 0.0,
                "missed": len(pos) - tp, "false_alarms": fp, "positives": len(pos)}

    base_acc = sum(p["category"] == "compensation" for p in preds) / len(preds)   # majority class in training
    lr_acc, n = acc("lr_category", preds)
    llm_acc, n_llm = acc("llm_category", preds)
    res = {"notes": len(preds), "baseline_accuracy": base_acc, "lr_accuracy": lr_acc, "llm_accuracy": llm_acc,
           "llm_notes": n_llm, "lr_escalation": esc("lr_escalate", preds),
           "llm_escalation": esc("llm_escalate", preds), "combined_escalation": esc("combined_escalate", preds)}

    print(f"\nHARD TEST SET: {len(preds)} hand-written notes ({sum(p['escalate'] for p in preds)} committee cases)")
    print(f"  {'Model':32s} {'Accuracy':>9s} {'Esc. recall':>12s} {'Esc. precision':>15s} {'Missed':>7s}")
    print(f"  {'Majority-class baseline':32s} {base_acc:9.1%} {0:12.1%} {'n/a':>15s} {sum(p['escalate'] for p in preds):7d}")
    for name, a, e in (("TF-IDF + Logistic Regression", lr_acc, res["lr_escalation"]),
                       ("LLM (zero-shot)", llm_acc, res["llm_escalation"]),
                       ("LLM + LR safety net", llm_acc, res["combined_escalation"])):
        if a is None or e is None:
            continue
        print(f"  {name:32s} {a:9.1%} {e['recall']:12.1%} {e['precision']:15.1%} {e['missed']:7d}")

    # Sensitivity analysis only (NOT used to pick the threshold: that would be tuning on the test set)
    pos = [p["lr_escalate_prob"] for p in preds if p["escalate"]]
    neg = [p["lr_escalate_prob"] for p in preds if not p["escalate"]]
    print("\n  LR escalation threshold sensitivity on hard notes (analysis only):")
    res["threshold_sensitivity"] = []
    for th in (0.48, 0.30, 0.20, 0.10):
        rec = sum(x >= th for x in pos) / len(pos)
        fa = sum(x >= th for x in neg)
        res["threshold_sensitivity"].append({"threshold": th, "recall": rec, "false_alarms": fa})
        print(f"    threshold {th:.2f}: recall {rec:5.0%}, false alarms {fa}/{len(neg)}")

    print("\n  Accuracy by difficulty type:")
    by = collections.defaultdict(list)
    for p in preds:
        by[p["difficulty"]].append(p)
    res["by_difficulty"] = {}
    for d, items in sorted(by.items()):
        la, _ = acc("lr_category", items)
        ma, mn = acc("llm_category", items)
        res["by_difficulty"][d] = {"n": len(items), "lr": la, "llm": ma}
        print(f"    {d:12s} n={len(items):2d}   LR {la:5.0%}" + (f"   LLM {ma:5.0%}" if mn else ""))

    print("\n  Notes the models got wrong:")
    for p in preds:
        wrong = [m for m in ("lr", "llm") if f"{m}_category" in p and p[f"{m}_category"] != p["category"]]
        if wrong:
            got = ", ".join(f"{m.upper()}={p[m + '_category']}" for m in wrong)
            print(f"    [{p['category']}] {got} | {p['text'][:70]}")

    (EVAL / "hard_test_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with (EVAL / "hard_test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["id", "difficulty", "category", "lr_category", "lr_conf", "lr_escalate_prob", "llm_category",
                "escalate", "lr_escalate", "llm_escalate", "combined_escalate", "text"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(preds)
    print("\nSaved data/eval/hard_test_results.json and hard_test_predictions.csv")


if __name__ == "__main__":
    main()

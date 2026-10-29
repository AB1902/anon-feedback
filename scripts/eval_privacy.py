"""Measures how well the privacy layer removes the PII we planted in the synthetic data.

    python -m scripts.eval_privacy                    # rules + Presidio only, all test notes, no API calls
    python -m scripts.eval_privacy --llm --limit 60   # full layer incl. AI rewrite (uses API quota)
    python -m scripts.eval_privacy --split train      # run on the development set instead

Recall = share of planted identifiers that no longer appear in the final text.
Results are saved to data/eval/ for the report.
"""
import argparse
import collections
import json
import time

from app.config import DATA_DIR
from app.pipeline.privacy import anonymise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--llm", action="store_true", help="include the AI style rewrite")
    ap.add_argument("--limit", type=int, default=0, help="max notes (0 = all)")
    ap.add_argument("--sleep", type=float, default=4.5, help="seconds between LLM calls")
    ap.add_argument("--no-presidio", action="store_true")
    args = ap.parse_args()

    path = DATA_DIR / "synthetic" / f"{args.split}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["pii"]]          # only notes that contain planted PII
    if args.limit:
        rows = rows[:args.limit]
    mode = "rules+rewrite" if args.llm else "rules only"
    print(f"Evaluating {len(rows)} notes with planted PII ({args.split} split, {mode})\n")

    planted, leaked = collections.Counter(), collections.Counter()
    rewrite_catches = 0
    examples, failures = [], []
    for i, r in enumerate(rows, 1):
        try:
            res = anonymise(r["text"], use_llm=args.llm, use_presidio=not args.no_presidio)
        except Exception as e:
            failures.append({"id": r["id"], "error": str(e)[:200]})
            print(f"  {r['id']} failed: {str(e)[:100]}")
            continue
        low = res.final_text.lower()
        note_leaks = []
        for p in r["pii"]:
            planted[p["type"]] += 1
            if p["value"].lower() in low:
                leaked[p["type"]] += 1
                note_leaks.append(p)
        rewrite_catches += len(res.leaks_after_rewrite)
        if note_leaks or len(examples) < 5:
            examples.append({"id": r["id"], "original": r["text"], "scrubbed": res.scrubbed.text,
                             "final": res.final_text, "leaked": note_leaks})
        if args.llm:
            if i % 10 == 0:
                print(f"  {i}/{len(rows)} done")
            time.sleep(args.sleep)

    total_p, total_l = sum(planted.values()), sum(leaked.values())
    print(f"{'Type':15s} {'Planted':>8s} {'Leaked':>7s} {'Recall':>7s}")
    by_type = {}
    for t in sorted(planted):
        rec = 1 - leaked[t] / planted[t]
        by_type[t] = {"planted": planted[t], "leaked": leaked[t], "recall": round(rec, 4)}
        print(f"{t:15s} {planted[t]:8d} {leaked[t]:7d} {rec:7.1%}")
    overall = 1 - total_l / total_p if total_p else 0
    print(f"{'OVERALL':15s} {total_p:8d} {total_l:7d} {overall:7.1%}")
    if args.llm:
        print(f"\nIdentifiers re-caught after the rewrite (safety net): {rewrite_catches}")
    if failures:
        print(f"Failed notes: {len(failures)} (re-run to retry)")

    out_dir = DATA_DIR / "eval"
    out_dir.mkdir(exist_ok=True)
    name = f"privacy_{args.split}_{'llm' if args.llm else 'rules'}.json"
    (out_dir / name).write_text(json.dumps({
        "mode": mode, "split": args.split, "notes": len(rows), "overall_recall": round(overall, 4),
        "by_type": by_type, "rewrite_catches": rewrite_catches, "failures": failures,
        "examples": examples[:40],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved details and before/after examples to data/eval/{name}")


if __name__ == "__main__":
    main()

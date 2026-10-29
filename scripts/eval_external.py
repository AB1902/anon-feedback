"""Runs the pipeline on a sample of a downloaded Kaggle dataset (real-world check).

    python -m scripts.eval_external --csv data/raw/glassdoor.csv --col cons --mode triage --n 50
    python -m scripts.eval_external --csv data/raw/sexist.csv --col statement --mode harassment --n 50
    python -m scripts.eval_external --csv data/raw/injections.csv --col text --mode guard --label-col label --n 200

Modes
  triage      : category/severity for real reviews -> CSV for manual spot-check
  harassment  : share routed to the POSH committee (or blocked as abuse)
  guard       : guardrail block rate; with --label-col, also false positives on benign rows
If --col is wrong, the script prints the available columns.
"""
import argparse
import collections
import csv
import json
import sys
import time

import pandas as pd

from app.config import DATA_DIR
from app.pipeline.guardrails import check_input, check_llm_flags
from app.pipeline.privacy import scrub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", required=True)
    ap.add_argument("--mode", choices=["triage", "harassment", "guard"], required=True)
    ap.add_argument("--label-col", default=None, help="guard mode: column marking attacks (1/true/injection/jailbreak)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--llm", action="store_true", help="guard mode: also use AI safety flags")
    ap.add_argument("--sleep", type=float, default=4.5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, on_bad_lines="skip", low_memory=False)
    if args.col not in df.columns:
        print(f"Column '{args.col}' not found. Columns: {list(df.columns)}")
        sys.exit(1)
    df = df[df[args.col].astype(str).str.len().between(20, 3000)]
    df = df.sample(min(args.n, len(df)), random_state=1)
    texts = df[args.col].astype(str).tolist()
    out_rows, stats = [], collections.Counter()

    if args.mode == "guard":
        labels = None
        if args.label_col:
            raw = df[args.label_col].astype(str).str.lower()
            labels = raw.isin(["1", "true", "injection", "jailbreak", "malicious", "yes"]).tolist()
        for i, t in enumerate(texts):
            g = check_input(t)
            blocked = not g.allowed
            if not blocked and args.llm:
                from app.pipeline.triage import analyse
                try:
                    blocked = not check_llm_flags(analyse(scrub(t).text).flags.model_dump()).allowed
                except Exception:
                    pass
                time.sleep(args.sleep)
            is_attack = labels[i] if labels is not None else True
            stats[("attack" if is_attack else "benign", blocked)] += 1
            out_rows.append({"text": t[:300], "attack": is_attack, "blocked": blocked, "reasons": ",".join(g.reasons)})
        att = stats[("attack", True)] + stats[("attack", False)]
        ben = stats[("benign", True)] + stats[("benign", False)]
        print(f"Attack block rate: {stats[('attack', True)]}/{att} = {stats[('attack', True)] / max(att, 1):.1%}")
        if ben:
            print(f"False-positive rate: {stats[('benign', True)]}/{ben} = {stats[('benign', True)] / ben:.1%}")
    else:
        from app.pipeline.triage import analyse, route
        for i, t in enumerate(texts, 1):
            g = check_input(t)
            row = {"text": t[:400], "blocked": not g.allowed, "reasons": ",".join(g.reasons)}
            if g.allowed:
                try:
                    s = scrub(t).text
                    a = analyse(s)
                    rt, _ = route(a.category, s)
                    row.update({"category": a.category, "sentiment": a.sentiment, "severity": a.severity,
                                "route": rt, "summary": a.summary,
                                "flagged": ",".join(k for k, v in a.flags.model_dump().items() if v)})
                    stats[a.category] += 1
                    stats["route:" + rt] += 1
                except Exception as e:
                    row["error"] = str(e)[:100]
                time.sleep(args.sleep)
            else:
                stats["blocked"] += 1
            out_rows.append(row)
            if i % 10 == 0:
                print(f"  {i}/{len(texts)}")
        n = len(texts)
        if args.mode == "harassment":
            posh = stats["route:posh_committee"]
            print(f"Routed to POSH committee: {posh}/{n} = {posh / n:.1%};  blocked as abusive: {stats['blocked']}")
        print("Category counts:", {k: v for k, v in stats.items() if not k.startswith("route:")})

    name = f"external_{args.mode}"
    (DATA_DIR / "eval").mkdir(exist_ok=True)
    pd.DataFrame(out_rows).to_csv(DATA_DIR / "eval" / f"{name}.csv", index=False)
    (DATA_DIR / "eval" / f"{name}.json").write_text(json.dumps({str(k): v for k, v in stats.items()}, indent=2))
    print(f"Saved data/eval/{name}.csv (open it to spot-check by hand)")


if __name__ == "__main__":
    main()

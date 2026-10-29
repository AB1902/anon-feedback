"""Red-team evaluation of the guardrails on the 110-input adversarial set.

    python -m scripts.eval_guardrails            # rule layer only (free, instant)
    python -m scripts.eval_guardrails --llm      # + AI safety flags on inputs the rules let through

Writes data/eval/guardrails_results.json and data/eval/red_team_table.csv (attack, expected,
result) for the report.
"""
import argparse
import collections
import csv
import json
import time

from app.config import DATA_DIR
from app.pipeline.guardrails import check_input, check_llm_flags
from app.pipeline.privacy import scrub

OUT = DATA_DIR / "eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--sleep", type=float, default=4.5)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (DATA_DIR / "synthetic" / "adversarial.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    results = []
    for r in rows:
        g = check_input(r["text"])
        blocked, layer, reason = not g.allowed, "rules" if not g.allowed else "", ", ".join(g.reasons)
        if not blocked and args.llm:
            from app.pipeline.triage import analyse
            try:
                a = analyse(scrub(r["text"]).text)
                fg = check_llm_flags(a.flags.model_dump())
                if not fg.allowed:
                    blocked, layer, reason = True, "AI safety flags", ", ".join(fg.reasons)
            except Exception as e:
                reason = f"error: {str(e)[:60]}"
            time.sleep(args.sleep)
        correct = blocked == (r["expected"] == "block")
        results.append({**r, "blocked": blocked, "layer": layer, "reason": reason, "correct": correct})

    by = collections.defaultdict(lambda: [0, 0])
    for x in results:
        by[x["attack_type"]][0] += x["correct"]
        by[x["attack_type"]][1] += 1
    attacks = [x for x in results if x["expected"] == "block"]
    benign = [x for x in results if x["expected"] == "allow"]
    block_rate = sum(x["blocked"] for x in attacks) / len(attacks)
    fp_rate = sum(x["blocked"] for x in benign) / len(benign)
    mode = "rules + AI safety flags" if args.llm else "rules only"
    print(f"Guardrail evaluation ({mode}) on {len(results)} inputs\n")
    print(f"  {'attack type':22s} {'correct':>8s}")
    for k, (c, n) in sorted(by.items()):
        print(f"  {k:22s} {c:3d}/{n:<3d} {c / n:6.0%}")
    print(f"\n  Attack block rate:     {block_rate:.1%}  ({sum(x['blocked'] for x in attacks)}/{len(attacks)})")
    print(f"  False-positive rate:   {fp_rate:.1%}  ({sum(x['blocked'] for x in benign)}/{len(benign)} harmless look-alikes blocked)")
    missed = [x for x in attacks if not x["blocked"]]
    if missed:
        print("\n  Attacks that got through:")
        for x in missed:
            print(f"    [{x['attack_type']}] {x['text'][:90]}")

    OUT.mkdir(exist_ok=True)
    name = "guardrails_results_llm.json" if args.llm else "guardrails_results.json"
    (OUT / name).write_text(json.dumps({
        "mode": mode, "inputs": len(results), "attack_block_rate": round(block_rate, 4),
        "false_positive_rate": round(fp_rate, 4),
        "by_type": {k: {"correct": c, "total": n} for k, (c, n) in by.items()}}, indent=2), encoding="utf-8")
    with (OUT / "red_team_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Attack type", "Input", "Expected", "Result", "Blocked by", "Reason", "Pass"])
        for x in results:
            if x["notes"] == "variant":
                continue
            w.writerow([x["attack_type"], x["text"], x["expected"], "blocked" if x["blocked"] else "allowed",
                        x["layer"], x["reason"], "yes" if x["correct"] else "NO"])
    print(f"\nSaved data/eval/{name} and data/eval/red_team_table.csv")


if __name__ == "__main__":
    main()

"""Stratified 80/20 split of the generated feedback (by category).

    python -m scripts.split_dataset
Outputs: data/synthetic/train.jsonl, data/synthetic/test.jsonl, and prints stats.
The test set is held out: never use it to tune prompts.
"""
import collections
import json
import random

from app.config import DATA_DIR

SRC = DATA_DIR / "synthetic" / "feedback_raw.jsonl"
TEST_FRACTION = 0.2


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(3)
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    train, test = [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        k = max(1, round(len(items) * TEST_FRACTION))
        test += items[:k]
        train += items[k:]

    for name, data in (("train", train), ("test", test)):
        data.sort(key=lambda r: r["id"])
        path = DATA_DIR / "synthetic" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")

    planted = sum(len(r["pii"]) for r in rows)
    missing = sum(len(r.get("pii_missing", [])) for r in rows)
    print(f"Total {len(rows)} -> train {len(train)}, test {len(test)}")
    print("Per category (train/test):")
    for cat in sorted(by_cat):
        print(f"  {cat:18s} {sum(r['category']==cat for r in train):4d} / {sum(r['category']==cat for r in test):3d}")
    print(f"PII planted and present: {planted}   (LLM dropped: {missing})")
    print("PII by type:", dict(collections.Counter(p["type"] for r in rows for p in r["pii"])))


if __name__ == "__main__":
    main()

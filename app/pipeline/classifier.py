"""Trained classifier: TF-IDF + logistic regression (scikit-learn).

Runs on every submission alongside the LLM:
- predicts the category with a confidence score,
- predicts the probability that the case needs a committee; if that probability is at or
  above the trained threshold, the case is routed to a committee even when the LLM missed it
  (a recall safety net),
- classifies on its own if the LLM service is unavailable.

Train with:  python -m scripts.train_classifier
If the saved model can't be loaded (e.g. a different scikit-learn version on the cloud), it is
retrained automatically from data/synthetic/train.jsonl in a few seconds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from app.config import DATA_DIR, ESCALATION_ROUTES, ROOT

MODEL_PATH = ROOT / "models" / "feedback_clf.joblib"
DEFAULT_THRESHOLD = 0.48


def features():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    return FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)),
    ])


def _quick_train() -> dict | None:
    """Fast retrain with the tuned settings (no grid search). Used only as a fallback."""
    path = DATA_DIR / "synthetic" / "train.jsonl"
    if not path.exists():
        return None
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from app.pipeline.privacy import scrub
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    X = [scrub(r["text"], use_presidio=False).text for r in rows]
    cat = Pipeline([("tfidf", features()), ("lr", LogisticRegression(C=30, max_iter=3000, class_weight="balanced"))])
    esc = Pipeline([("tfidf", features()), ("lr", LogisticRegression(C=30, max_iter=3000, class_weight="balanced"))])
    cat.fit(X, [r["category"] for r in rows])
    esc.fit(X, [int(r["category"] in ESCALATION_ROUTES) for r in rows])
    bundle = {"category_model": cat, "escalation_model": esc, "threshold": DEFAULT_THRESHOLD,
              "trained_on": len(rows), "trained_at": "auto-retrained", "test_metrics": {}}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    return bundle


@lru_cache(maxsize=1)
def load() -> dict | None:
    try:
        import joblib
        b = joblib.load(MODEL_PATH)
        b["category_model"].predict(["test"])          # fails fast on a version mismatch
        return b
    except Exception as e:
        print(f"[classifier] saved model unavailable ({str(e)[:80]}); retraining quickly")
        try:
            return _quick_train()
        except Exception as e2:
            print(f"[classifier] disabled: {str(e2)[:120]}")
            return None


@dataclass
class MLPrediction:
    category: str
    confidence: float          # probability of the predicted category
    escalate_prob: float       # probability the case needs a committee
    threshold: float

    @property
    def escalate(self) -> bool:
        return self.escalate_prob >= self.threshold

    @property
    def committee_category(self) -> str:
        """Most likely committee category, used when the safety net escalates a case."""
        return self._committee

    _committee: str = "harassment"


def predict(scrubbed_text: str) -> MLPrediction | None:
    b = load()
    if not b:
        return None
    cm, em = b["category_model"], b["escalation_model"]
    probs = cm.predict_proba([scrubbed_text])[0]
    classes = list(cm.classes_)
    i = int(probs.argmax())
    esc_classes = [c for c in classes if c in ESCALATION_ROUTES]
    committee = max(esc_classes, key=lambda c: probs[classes.index(c)]) if esc_classes else "harassment"
    p = MLPrediction(classes[i], float(probs[i]), float(em.predict_proba([scrubbed_text])[0][1]),
                     float(b.get("threshold", DEFAULT_THRESHOLD)))
    p._committee = committee
    return p


def model_card() -> dict:
    b = load()
    if not b:
        return {"available": False}
    m = b.get("test_metrics", {})
    return {"available": True, "threshold": b.get("threshold"), "trained_on": b.get("trained_on"),
            "trained_at": b.get("trained_at"),
            "test_accuracy": m.get("category", {}).get("test_accuracy"),
            "test_macro_f1": m.get("category", {}).get("test_macro_f1"),
            "escalation_recall": m.get("escalation", {}).get("at_chosen_threshold", {}).get("recall"),
            "roc_auc": m.get("escalation", {}).get("roc_auc")}

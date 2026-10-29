"""Trains the logistic regression classifiers and picks the escalation threshold.

    python -m scripts.train_classifier

Two models, both TF-IDF (word 1-2 grams + character 3-5 grams, so Hinglish and typos still
match) + logistic regression, trained on the 798 development notes only:

  1. Category model   : 10 classes (compensation, workload, harassment, ...)
  2. Escalation model : yes/no, "does this case need a committee?" (harassment, safety,
                        fraud, complaint about HR). Outputs a probability.

Threshold: the escalation probability is turned into a decision with a threshold chosen on
the TRAINING data only (5-fold out-of-fold predictions), as the highest threshold that still
catches >= 95% of real committee cases. Missing a harassment case is worse than an extra
review, so recall is prioritised over precision. The default 0.5 is reported for comparison.

Everything is then scored once on the 200 held-out test notes against:
  - a majority-class baseline (the floor any model must beat),
  - the LLM's labels, if scripts.eval_triage has been run.

Outputs: models/feedback_clf.joblib, data/eval/classifier_results.json,
data/eval/threshold_table.csv, data/eval/fig_confusion_matrix.png, data/eval/fig_threshold_curve.png
"""
import json
import time

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from app.config import DATA_DIR, ESCALATION_ROUTES, ROOT
from app.pipeline.classifier import features
from app.pipeline.privacy import scrub

TARGET_RECALL = 0.95
MODEL_PATH = ROOT / "models" / "feedback_clf.joblib"
EVAL = DATA_DIR / "eval"
SEED = 42


def load(split):
    rows = [json.loads(l) for l in (DATA_DIR / "synthetic" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # Train on scrubbed text, exactly what the model sees inside the app
    X = [scrub(r["text"], use_presidio=False).text for r in rows]
    y_cat = [r["category"] for r in rows]
    y_esc = np.array([int(r["category"] in ESCALATION_ROUTES) for r in rows])
    return rows, X, y_cat, y_esc


def pick_threshold(y_true, proba, target_recall):
    """Highest threshold whose recall is still >= target (fewest false alarms at that recall)."""
    best = 0.05
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        if recall_score(y_true, (proba >= t).astype(int), zero_division=0) >= target_recall:
            best = t
    return float(best)


def main() -> None:
    t0 = time.time()
    tr_rows, Xtr, ytr_cat, ytr_esc = load("train")
    te_rows, Xte, yte_cat, yte_esc = load("test")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    print(f"Training on {len(Xtr)} notes, testing on {len(Xte)} held-out notes\n")

    # ---------------- 1. category model
    cat_pipe = Pipeline([("tfidf", features()),
                         ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    grid = GridSearchCV(cat_pipe, {"lr__C": [1, 3, 10, 30]}, cv=cv, scoring="f1_macro", n_jobs=-1)
    grid.fit(Xtr, ytr_cat)
    cat_model = grid.best_estimator_
    pred_cat = cat_model.predict(Xte)
    base_cat = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr_cat).predict(Xte)
    cat_res = {
        "best_C": grid.best_params_["lr__C"], "cv_macro_f1": round(grid.best_score_, 4),
        "test_accuracy": round(accuracy_score(yte_cat, pred_cat), 4),
        "test_macro_f1": round(f1_score(yte_cat, pred_cat, average="macro"), 4),
        "baseline_accuracy": round(accuracy_score(yte_cat, base_cat), 4),
        "baseline_macro_f1": round(f1_score(yte_cat, base_cat, average="macro", zero_division=0), 4),
    }
    print("CATEGORY MODEL (10 classes)")
    print(f"  best C = {cat_res['best_C']}   5-fold CV macro-F1 on training data = {cat_res['cv_macro_f1']}")
    print(f"  Test accuracy {cat_res['test_accuracy']:.1%}   macro-F1 {cat_res['test_macro_f1']:.3f}")
    print(f"  Majority baseline: accuracy {cat_res['baseline_accuracy']:.1%}   macro-F1 {cat_res['baseline_macro_f1']:.3f}")
    print(classification_report(yte_cat, pred_cat, digits=2, zero_division=0))

    # ---------------- 2. escalation model + threshold
    esc_pipe = Pipeline([("tfidf", features()),
                         ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    eg = GridSearchCV(esc_pipe, {"lr__C": [1, 3, 10, 30]}, cv=cv, scoring="average_precision", n_jobs=-1)
    eg.fit(Xtr, ytr_esc)
    esc_model = eg.best_estimator_
    oof = cross_val_predict(eg.best_estimator_, Xtr, ytr_esc, cv=cv, method="predict_proba")[:, 1]
    thr = pick_threshold(ytr_esc, oof, TARGET_RECALL)
    p_te = esc_model.predict_proba(Xte)[:, 1]

    def scores(t):
        yp = (p_te >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte_esc, yp, labels=[0, 1]).ravel()
        return {"threshold": t, "precision": round(precision_score(yte_esc, yp, zero_division=0), 4),
                "recall": round(recall_score(yte_esc, yp, zero_division=0), 4),
                "f1": round(f1_score(yte_esc, yp, zero_division=0), 4),
                "missed_committee_cases": int(fn), "false_alarms": int(fp), "tp": int(tp), "tn": int(tn)}

    chosen, default = scores(thr), scores(0.5)
    base_esc = scores(1.01)          # "never escalate" baseline
    esc_res = {"best_C": eg.best_params_["lr__C"], "target_recall": TARGET_RECALL, "threshold": thr,
               "roc_auc": round(roc_auc_score(yte_esc, p_te), 4),
               "pr_auc": round(average_precision_score(yte_esc, p_te), 4),
               "at_chosen_threshold": chosen, "at_default_0_5": default, "baseline_never_escalate": base_esc,
               "test_positives": int(yte_esc.sum()), "test_total": len(yte_esc)}
    print("ESCALATION MODEL (needs a committee: yes/no)")
    print(f"  Threshold chosen on training data for recall >= {TARGET_RECALL:.0%}: {thr:.2f}")
    print(f"  Test ROC-AUC {esc_res['roc_auc']:.3f}   PR-AUC {esc_res['pr_auc']:.3f}   "
          f"({esc_res['test_positives']} committee cases in {len(yte_esc)} test notes)")
    print(f"  {'':22s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'missed':>7s} {'false alarms':>13s}")
    for name, s in (("chosen threshold", chosen), ("default 0.50", default), ("never escalate", base_esc)):
        print(f"  {name:22s} {s['precision']:9.2f} {s['recall']:7.2f} {s['f1']:6.2f} {s['missed_committee_cases']:7d} {s['false_alarms']:13d}")

    # threshold table for the report
    EVAL.mkdir(exist_ok=True)
    with (EVAL / "threshold_table.csv").open("w", encoding="utf-8") as f:
        f.write("threshold,precision,recall,f1,missed_committee_cases,false_alarms\n")
        for t in np.round(np.arange(0.1, 0.95, 0.05), 2):
            s = scores(float(t))
            f.write(f"{t:.2f},{s['precision']},{s['recall']},{s['f1']},{s['missed_committee_cases']},{s['false_alarms']}\n")

    # ---------------- 3. compare with the LLM on the same notes (if available)
    llm_res = None
    pp = EVAL / "triage_predictions.jsonl"
    if pp.exists():
        preds = {json.loads(l)["id"]: json.loads(l) for l in pp.read_text(encoding="utf-8").splitlines() if l.strip()}
        idx = [i for i, r in enumerate(te_rows) if r["id"] in preds]
        if idx:
            g = [yte_cat[i] for i in idx]
            llm = [preds[te_rows[i]["id"]]["pred_category"] for i in idx]
            lr_sub = [pred_cat[i] for i in idx]
            ge = [int(c in ESCALATION_ROUTES) for c in g]
            le = [int(preds[te_rows[i]["id"]]["pred_route"] != "hr") for i in idx]
            me = [int(p_te[i] >= thr) for i in idx]
            llm_res = {"notes": len(idx),
                       "llm_accuracy": round(accuracy_score(g, llm), 4), "llm_macro_f1": round(f1_score(g, llm, average="macro"), 4),
                       "lr_accuracy": round(accuracy_score(g, lr_sub), 4), "lr_macro_f1": round(f1_score(g, lr_sub, average="macro"), 4),
                       "llm_escalation_recall": round(recall_score(ge, le, zero_division=0), 4),
                       "lr_escalation_recall": round(recall_score(ge, me, zero_division=0), 4),
                       "combined_escalation_recall": round(recall_score(ge, [max(a, b) for a, b in zip(le, me)], zero_division=0), 4)}
            print(f"\nLLM vs LOGISTIC REGRESSION on the same {len(idx)} test notes")
            print(f"  Category accuracy   LLM {llm_res['llm_accuracy']:.1%}   LR {llm_res['lr_accuracy']:.1%}")
            print(f"  Category macro-F1   LLM {llm_res['llm_macro_f1']:.3f}   LR {llm_res['lr_macro_f1']:.3f}")
            print(f"  Escalation recall   LLM {llm_res['llm_escalation_recall']:.1%}   LR {llm_res['lr_escalation_recall']:.1%}   "
                  f"LLM + LR safety net {llm_res['combined_escalation_recall']:.1%}")
    else:
        print("\n(Run `python -m scripts.eval_triage --limit 100` to add the LLM comparison.)")

    # ---------------- 4. interpretability: strongest words per category
    vec = cat_model.named_steps["tfidf"].transformer_list[0][1]
    names = np.array(vec.get_feature_names_out())
    n_words = len(names)
    coefs = cat_model.named_steps["lr"].coef_[:, :n_words]
    top = {c: list(names[np.argsort(coefs[i])[::-1][:6]]) for i, c in enumerate(cat_model.classes_)}
    print("\nTop words per category (what the model learned):")
    for c, w in top.items():
        print(f"  {c:18s} {', '.join(w)}")

    # ---------------- 5. figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = list(cat_model.classes_)
        cm = confusion_matrix(yte_cat, pred_cat, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Logistic regression: category confusion matrix (test, n={len(yte_cat)})")
        fig.tight_layout(); fig.savefig(EVAL / "fig_confusion_matrix.png", dpi=160); plt.close(fig)

        prec, rec, th = precision_recall_curve(yte_esc, p_te)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(th, prec[:-1], label="Precision")
        ax.plot(th, rec[:-1], label="Recall")
        ax.axvline(thr, ls="--", color="crimson", label=f"Chosen threshold = {thr:.2f}")
        ax.axvline(0.5, ls=":", color="grey", label="Default 0.50")
        ax.set_xlabel("Escalation probability threshold"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
        ax.set_title("Escalation model: precision and recall vs threshold (test set)")
        ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(EVAL / "fig_threshold_curve.png", dpi=160); plt.close(fig)
    except Exception as e:
        print(f"(figures skipped: {e})")

    # ---------------- save
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"category_model": cat_model, "escalation_model": esc_model, "threshold": thr,
                 "target_recall": TARGET_RECALL, "trained_on": len(Xtr),
                 "trained_at": time.strftime("%Y-%m-%d"), "test_metrics": {"category": cat_res, "escalation": esc_res}},
                MODEL_PATH, compress=3)
    (EVAL / "classifier_results.json").write_text(json.dumps(
        {"category": cat_res, "escalation": esc_res, "llm_comparison": llm_res, "top_words": top}, indent=2), encoding="utf-8")
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB), "
          f"data/eval/classifier_results.json, threshold_table.csv and 2 figures  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()

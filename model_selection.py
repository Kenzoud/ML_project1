"""Cross-validated ablation and hyperparameter selection, then holdout scoring.

Usage: python model_selection.py   (~20 minutes)
Writes outputs/model_selection.json and prints the result tables.

Protocol:
  1. Hold out 20% of the labeled rows (stratified, seed 42). Nothing below
     looks at them until step 4.
  2. Ablation: 5-fold stratified CV on the other 80% for every preprocessing
     config, with logistic and ridge regression.
  3. Selection: on the "full" config, cross-validate the regularization
     strength lambda of logistic regression. The model with the best mean
     fold F1 is kept, with the threshold chosen on its pooled out-of-fold scores.
  4. Refit the selected model on the 80% and score the holdout once at the
     selected threshold: an unbiased estimate of test performance.
Folds are identical across configs and models, so fold-wise differences are
paired comparisons.
"""

import json

import numpy as np

from cross_validation import cross_validate, logistic_model, ridge_model, summarize
from evaluation import classification_metrics, predict_labels, roc_auc, stratified_split
from preprocessing import CONFIGS, fit_preprocessor, load_data, transform

SEED = 42
K = 5
LAMBDAS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
RIDGE_LAMBDA = 1e-5


def print_row(label, s, ref=None):
    """Print one CV summary line, with the paired F1 difference to ref."""
    delta = ""
    if ref is not None:
        d = np.array(s["f1_folds"]) - np.array(ref["f1_folds"])
        delta = f"  dF1={d.mean():+.4f}±{d.std():.4f}"
    print(
        f"{label:<36} F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}"
        f"  P={s['precision_mean']:.3f} R={s['recall_mean']:.3f}"
        f"  acc={s['accuracy_mean']:.4f}  AUC={s['auc_mean']:.4f}"
        f"  t={s['threshold']:.2f}{delta}",
        flush=True,
    )


def main():
    """Run the ablation, select lambda, and score the holdout."""
    x, _, y, _, _, names = load_data()
    tr, hold = stratified_split(y, val_ratio=0.2, seed=SEED)
    x_tr, y_tr = x[tr], y[tr]
    results = {"ablation": [], "lambda": [], "holdout": []}

    print(f"== Ablation ({K}-fold CV on {len(tr)} rows)")
    previous = None
    for config in CONFIGS:
        models = [logistic_model(0), ridge_model(RIDGE_LAMBDA)]
        oof, fold_of = cross_validate(x_tr, y_tr, names, config, models, K, SEED)
        for m in models:
            s = summarize(y_tr, oof[m["name"]], fold_of)
            s.update({"config": config, "model": m["name"]})
            results["ablation"].append(s)
            is_logistic = m["name"].startswith("logistic")
            print_row(f"{config} / {m['name']}", s, previous if is_logistic else None)
            if is_logistic:
                previous = s

    print(f"== Regularization (full config, {K}-fold CV)")
    reference = next(
        s
        for s in results["ablation"]
        if s["config"] == "full" and s["model"] == "logistic lambda=0"
    )
    models = [logistic_model(l) for l in LAMBDAS if l > 0]
    oof, fold_of = cross_validate(x_tr, y_tr, names, "full", models, K, SEED)
    results["lambda"].append({**reference, "lambda": 0})
    print_row("logistic lambda=0", reference)
    for m, l in zip(models, [l for l in LAMBDAS if l > 0]):
        s = summarize(y_tr, oof[m["name"]], fold_of)
        s.update({"config": "full", "model": m["name"], "lambda": l})
        results["lambda"].append(s)
        print_row(m["name"], s, reference)

    best = max(results["lambda"], key=lambda s: s["f1_mean"])
    print(f"selected: lambda={best['lambda']:g}, threshold={best['threshold']:.2f}")

    print("== Holdout (scored once)")
    for config, lambda_, t in [
        (
            "baseline",
            0,
            next(
                s["threshold"]
                for s in results["ablation"]
                if s["config"] == "baseline" and s["model"] == "logistic lambda=0"
            ),
        ),
        ("full", best["lambda"], best["threshold"]),
    ]:

        params = fit_preprocessor(x_tr, names, config)
        model = logistic_model(lambda_)
        w = model["fit"](y_tr, transform(x_tr, params))
        scores = model["link"](transform(x[hold], params) @ w)
        m = classification_metrics(y[hold], predict_labels(scores, t))
        m.update(
            {
                "config": config,
                "lambda": lambda_,
                "threshold": t,
                "auc": float(roc_auc(y[hold], scores)),
            }
        )
        results["holdout"].append(m)
        print(
            f"{config} lambda={lambda_:g} t={t:.2f}: F1={m['f1']:.4f} P={m['precision']:.3f}"
            f" R={m['recall']:.3f} acc={m['accuracy']:.4f} AUC={m['auc']:.4f}"
        )
    # fmt: skip

    with open("outputs/model_selection.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

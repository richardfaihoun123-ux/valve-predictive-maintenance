"""Entraîne le modèle final et écrit les artefacts.

Usage :  python -m valve.train --data-dir data --out-dir .
(nécessite PS2.txt, FS1.txt et profile.txt dans --data-dir)
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .data import N_TRAIN, binarize_valve, load_raw
from .features import build_features
from .pipeline import RANDOM_STATE, make_pipeline, selected_feature_names


def main(data_dir: str = "data", out_dir: str = ".") -> dict:
    out = Path(out_dir)
    ps2, fs1, profile = load_raw(data_dir)
    feats = build_features(ps2, fs1)
    y = binarize_valve(profile["valve"])
    X_tr, X_te, y_tr, y_te = feats.iloc[:N_TRAIN], feats.iloc[N_TRAIN:], y[:N_TRAIN], y[N_TRAIN:]

    pipe = make_pipeline()
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    r = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    pipe.fit(X_tr, y_tr)
    pred, score = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    names = selected_feature_names(pipe, feats.columns)
    metrics = {
        "cv_f1_mean": float(np.mean(r["test_f1"])), "cv_accuracy_mean": float(np.mean(r["test_accuracy"])),
        "cv_roc_auc_mean": float(np.mean(r["test_roc_auc"])),
        "test_accuracy": accuracy_score(y_te, pred), "test_precision": precision_score(y_te, pred),
        "test_recall": recall_score(y_te, pred), "test_f1": f1_score(y_te, pred),
        "test_roc_auc": roc_auc_score(y_te, score), "n_train": int(N_TRAIN), "n_test": int(len(y_te)),
        "selected_features": {n: float(c) for n, c in zip(names, pipe.named_steps["clf"].coef_[0])},
        "n_features_total": int(feats.shape[1]),
        "n_features_after_corr_filter": int(len(pipe.named_steps["corr"].keep_)),
    }
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "feature_names": list(feats.columns)}, out / "models" / "best_valve_model.joblib")
    feats.to_csv(out / "data" / "features_all_cycles.csv", index=False)
    (out / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    main(a.data_dir, a.out_dir)

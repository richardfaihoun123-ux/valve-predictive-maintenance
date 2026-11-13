"""Extraction de variables à partir des signaux bruts (un cycle = une série temporelle)."""
import numpy as np
import pandas as pd
from scipy import stats

N_SEGMENTS = 10


def extract_features(X: np.ndarray, name: str, n_seg: int = N_SEGMENTS) -> pd.DataFrame:
    """Résume chaque cycle (ligne de X) par 40 statistiques.

    - statistiques globales : tendance centrale, dispersion, quantiles, forme, énergie, pente ;
    - dynamique : variations d'un point à l'autre ;
    - moyenne et écart-type sur `n_seg` segments temporels consécutifs.
    """
    X = np.asarray(X, dtype=float)
    q = np.percentile(X, [5, 25, 50, 75, 95], axis=1)
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2
    d = np.diff(X, axis=1)
    f = {
        "mean": X.mean(1), "std": X.std(1), "min": X.min(1), "max": X.max(1), "ptp": np.ptp(X, axis=1),
        "median": q[2], "p05": q[0], "p25": q[1], "p75": q[3], "p95": q[4], "iqr": q[3] - q[1],
        "skew": stats.skew(X, axis=1), "kurt": stats.kurtosis(X, axis=1),
        "energy": (X ** 2).mean(1),
        "slope": ((t * (X - X.mean(1, keepdims=True))).sum(1)) / (t ** 2).sum(),
        "argmax": X.argmax(1) / X.shape[1], "argmin": X.argmin(1) / X.shape[1],
        "diff_mean_abs": np.abs(d).mean(1), "diff_max": d.max(1), "diff_min": d.min(1),
    }
    for i, seg in enumerate(np.array_split(X, n_seg, axis=1)):
        f[f"seg{i}_mean"] = seg.mean(1)
        f[f"seg{i}_std"] = seg.std(1)
    return pd.DataFrame({f"{name}_{k}": v for k, v in f.items()})


def build_features(ps2: np.ndarray, fs1: np.ndarray) -> pd.DataFrame:
    """Concatène les variables de PS2 (pression) et FS1 (débit). Les NaN (skew/kurtosis d'un
    signal constant) sont remplacés par 0."""
    feats = pd.concat([extract_features(ps2, "PS2"), extract_features(fs1, "FS1")], axis=1)
    return feats.fillna(0.0)

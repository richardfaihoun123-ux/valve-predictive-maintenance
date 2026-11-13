"""Transformeurs scikit-learn personnalisés."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrFilter(BaseEstimator, TransformerMixin):
    """Supprime, pour chaque paire de variables dont |corrélation| > `threshold`, la seconde.

    S'ajuste uniquement sur les données d'apprentissage reçues par `fit` (pas de fuite en CV).
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        c = np.nan_to_num(np.abs(np.corrcoef(np.asarray(X, dtype=float), rowvar=False)))
        up = np.triu(c, k=1)
        self.keep_ = np.array([j for j in range(c.shape[0]) if not (up[:, j] > self.threshold).any()])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]

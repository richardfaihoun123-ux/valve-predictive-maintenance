"""Pipeline final : variance -> filtre de corrélation -> mise à l'échelle -> sélection -> classifieur.

Toute la sélection de variables est DANS le pipeline : elle est donc réajustée dans chaque pli
de validation croisée, ce qui évite toute fuite d'information vers les données de validation.
"""
import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .transformers import CorrFilter

RANDOM_STATE = 42
# Hyperparamètres retenus par Grid Search (F1, validation croisée 5-fold) — voir le notebook.
BEST_PARAMS = {"k": 10, "C": 10.0}


def make_pipeline(k: int = BEST_PARAMS["k"], C: float = BEST_PARAMS["C"]) -> Pipeline:
    return Pipeline([
        ("var", VarianceThreshold(1e-10)),
        ("corr", CorrFilter(0.95)),
        ("scale", RobustScaler()),
        ("sel", SelectKBest(f_classif, k=k)),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE)),
    ])


def selected_feature_names(pipe: Pipeline, feature_names):
    """Noms des variables réellement utilisées par le classifieur."""
    names = np.array(list(feature_names))
    names = names[pipe.named_steps["var"].get_support()]
    names = names[pipe.named_steps["corr"].keep_]
    return names[pipe.named_steps["sel"].get_support()]

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from valve.pipeline import make_pipeline, selected_feature_names


def _jeu_synthetique(n=400, p=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 3] + 0.1 * rng.normal(size=n) > 0).astype(int)   # seule la variable 3 est informative
    return pd.DataFrame(X, columns=[f"v{i}" for i in range(p)]), y


def test_pipeline_apprend_et_selectionne_la_bonne_variable():
    X, y = _jeu_synthetique()
    pipe = make_pipeline(k=3).fit(X, y)
    assert pipe.score(X, y) > 0.95
    assert "v3" in selected_feature_names(pipe, X.columns)


def test_pipeline_sans_fuite_en_validation_croisee():
    X, y = _jeu_synthetique()
    assert cross_val_score(make_pipeline(k=3), X, y, cv=5).mean() > 0.9


def test_bruit_pur_reste_au_niveau_du_hasard():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(300, 20)))
    y = rng.integers(0, 2, 300)
    assert cross_val_score(make_pipeline(k=5), X, y, cv=5).mean() < 0.65   # la sélection n'« invente » rien


def test_probabilites_valides():
    X, y = _jeu_synthetique()
    p = make_pipeline(k=3).fit(X, y).predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(1), 1)

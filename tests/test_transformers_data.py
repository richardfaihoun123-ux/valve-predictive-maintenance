import numpy as np

from valve.data import binarize_valve
from valve.transformers import CorrFilter


def test_corrfilter_supprime_la_colonne_redondante():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    X = np.column_stack([a, 2 * a + 1e-6 * rng.normal(size=200), b])
    cf = CorrFilter(0.95).fit(X)
    assert list(cf.keep_) == [0, 2]
    assert cf.transform(X).shape == (200, 2)


def test_corrfilter_garde_les_colonnes_independantes():
    X = np.random.default_rng(1).normal(size=(300, 5))
    assert CorrFilter(0.95).fit(X).transform(X).shape[1] == 5


def test_binarisation_de_la_valve():
    assert list(binarize_valve([100, 90, 80, 73, 100])) == [1, 0, 0, 0, 1]

import numpy as np
import pytest

from valve.features import build_features, extract_features


def test_nombre_de_variables_par_capteur():
    X = np.random.default_rng(0).normal(size=(5, 600))
    f = extract_features(X, "S")
    assert f.shape == (5, 40)            # 20 globales + 10 segments x (moyenne, écart-type)
    assert f.columns[0] == "S_mean" and "S_seg9_std" in f.columns


def test_valeurs_connues_sur_signal_constant():
    X = np.full((3, 100), 5.0)
    f = extract_features(X, "S")
    assert np.allclose(f["S_mean"], 5.0) and np.allclose(f["S_std"], 0.0)
    assert np.allclose(f["S_slope"], 0.0) and np.allclose(f["S_ptp"], 0.0)


def test_pente_d_une_rampe():
    X = np.tile(np.arange(100) * 2.0, (2, 1))     # pente = 2 par échantillon
    assert np.allclose(extract_features(X, "S")["S_slope"], 2.0)


def test_segments_decoupent_le_signal():
    X = np.concatenate([np.zeros((1, 50)), np.ones((1, 50))], axis=1)
    f = extract_features(X, "S", n_seg=2)
    assert f["S_seg0_mean"].iloc[0] == 0.0 and f["S_seg1_mean"].iloc[0] == 1.0


def test_deterministe_et_sans_nan():
    rng = np.random.default_rng(1)
    ps2, fs1 = rng.normal(size=(4, 6000)), rng.normal(size=(4, 600))
    a, b = build_features(ps2, fs1), build_features(ps2, fs1)
    assert a.shape == (4, 80) and a.equals(b) and not a.isna().any().any()


def test_signal_constant_ne_produit_pas_de_nan():
    f = build_features(np.ones((2, 6000)), np.ones((2, 600)))
    assert not f.isna().any().any()

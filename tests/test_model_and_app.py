"""Tests sur les artefacts livrés (modèle, variables, profil) et sur l'application web."""
from pathlib import Path

import pytest

from valve.data import N_TRAIN, binarize_valve, load_profile
from valve.predict import ValvePredictor

ROOT = Path(__file__).resolve().parent.parent
MODEL, FEATS, PROFILE = (ROOT / "models" / "best_valve_model.joblib", ROOT / "data" / "features_all_cycles.csv",
                         ROOT / "data" / "profile.txt")
pytestmark = pytest.mark.skipif(not (MODEL.exists() and FEATS.exists() and PROFILE.exists()),
                                reason="artefacts absents : lancer `python -m valve.train`")


@pytest.fixture(scope="module")
def predictor():
    return ValvePredictor.load(MODEL, FEATS, PROFILE)


def test_performance_sur_le_test_final(predictor):
    y = binarize_valve(load_profile(PROFILE)["valve"])
    X_te = predictor.features.iloc[N_TRAIN:][predictor.feature_names]
    acc = (predictor.model.predict(X_te) == y[N_TRAIN:]).mean()
    assert acc >= 0.98


def test_predict_cycle_structure(predictor):
    r = predictor.predict_cycle(1)
    assert {"cycle", "optimal", "probability_optimal", "label", "contributions", "correct"} <= r.keys()
    assert 0 <= r["probability_optimal"] <= 1 and len(r["contributions"]) == 6


def test_predict_cycle_hors_bornes(predictor):
    for bad in (0, predictor.n_cycles + 1, -3):
        with pytest.raises(IndexError):
            predictor.predict_cycle(bad)


@pytest.fixture(scope="module")
def client():
    from app.app import create_app
    return create_app().test_client()


def test_app_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_app_page_accueil(client):
    r = client.get("/")
    assert r.status_code == 200 and "Condition de la valve" in r.get_data(as_text=True)


def test_app_api_predict(client):
    r = client.get("/api/predict/2001")
    d = r.get_json()
    assert r.status_code == 200 and d["cycle"] == 2001 and d["label"] in {"OPTIMAL", "NON OPTIMAL"}


def test_app_api_cycle_invalide(client):
    assert client.get("/api/predict/0").status_code == 404
    assert client.get("/api/predict/99999").status_code == 404
    assert client.get("/api/predict/abc").status_code == 404

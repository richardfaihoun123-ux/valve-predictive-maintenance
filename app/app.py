"""Petite application web : prédiction de la condition de la valve pour un numéro de cycle.

Lancement local :  python -m flask --app app.app run      (ou : gunicorn app.app:app)
API :  GET /api/predict/<n>   ·   GET /health
"""
import os
from pathlib import Path

from flask import Flask, jsonify, render_template

from valve.predict import ValvePredictor

ROOT = Path(__file__).resolve().parent.parent


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["predictor"] = ValvePredictor.load(
        os.environ.get("MODEL_PATH", ROOT / "models" / "best_valve_model.joblib"),
        os.environ.get("FEATURES_PATH", ROOT / "data" / "features_all_cycles.csv"),
        os.environ.get("PROFILE_PATH", ROOT / "data" / "profile.txt"))

    @app.get("/")
    def index():
        return render_template("index.html", n_cycles=app.config["predictor"].n_cycles)

    @app.get("/health")
    def health():
        return jsonify(status="ok", n_cycles=app.config["predictor"].n_cycles)

    @app.get("/api/predict/<int:n>")
    def predict(n: int):
        try:
            return jsonify(app.config["predictor"].predict_cycle(n))
        except IndexError as e:
            return jsonify(error=str(e)), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

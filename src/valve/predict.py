"""Prédiction pour un numéro de cycle (1 = premier cycle)."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import joblib

from .data import N_TRAIN, binarize_valve, load_profile
from .pipeline import selected_feature_names


@dataclass
class ValvePredictor:
    model: object
    feature_names: list
    features: pd.DataFrame
    profile: pd.DataFrame | None = None

    @classmethod
    def load(cls, model_path, features_path, profile_path=None):
        art = joblib.load(model_path)
        feats = pd.read_csv(features_path)
        prof = load_profile(profile_path) if profile_path and Path(profile_path).exists() else None
        return cls(art["model"], art["feature_names"], feats, prof)

    @property
    def n_cycles(self) -> int:
        return len(self.features)

    def predict_cycle(self, n: int) -> dict:
        if not 1 <= n <= self.n_cycles:
            raise IndexError(f"Le cycle doit être compris entre 1 et {self.n_cycles}.")
        x = self.features.iloc[[n - 1]][self.feature_names]
        proba = float(self.model.predict_proba(x)[0, 1])
        optimal = bool(self.model.predict(x)[0])

        # Contribution de chaque variable sélectionnée (coefficient x valeur mise à l'échelle).
        names = selected_feature_names(self.model, self.feature_names)
        z = self.model[:-1].transform(x)[0]
        contrib = self.model.named_steps["clf"].coef_[0] * z
        top = sorted(zip(names, contrib, x[list(names)].iloc[0]), key=lambda t: -abs(t[1]))[:6]

        out = {"cycle": n, "optimal": optimal, "probability_optimal": round(proba, 4),
               "label": "OPTIMAL" if optimal else "NON OPTIMAL",
               "split": "apprentissage (cycles 1-2000)" if n <= N_TRAIN else "test final (cycles 2001+)",
               "contributions": [{"feature": f, "value": round(float(v), 4), "contribution": round(float(c), 4)}
                                 for f, c, v in top]}
        if self.profile is not None:
            level = int(self.profile["valve"].iloc[n - 1])
            true_opt = bool(binarize_valve(level))
            out.update(true_valve_condition=level, true_optimal=true_opt, correct=(true_opt == optimal))
        return out

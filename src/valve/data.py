"""Chargement des données et construction de la cible."""
from pathlib import Path
import numpy as np
import pandas as pd

N_TRAIN = 2000  # les 2000 premiers cycles construisent le modèle, les suivants forment le test final
PROFILE_COLUMNS = ["cooler", "valve", "pump_leakage", "accumulator", "stable_flag"]


def load_profile(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=PROFILE_COLUMNS)


def load_signal(path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None).values


def load_raw(data_dir):
    d = Path(data_dir)
    return load_signal(d / "PS2.txt"), load_signal(d / "FS1.txt"), load_profile(d / "profile.txt")


def binarize_valve(valve) -> np.ndarray:
    """1 = valve optimale (100 %), 0 = non optimale (90, 80 ou 73 %)."""
    return (np.asarray(valve) == 100).astype(int)

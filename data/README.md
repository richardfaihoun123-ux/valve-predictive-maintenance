# Données

| Fichier | Contenu | Dans Git ? |
|---|---|---|
| `PS2.txt` | Pression (bar), 100 Hz, 6000 points par cycle, 2205 cycles | non (≈ 80 Mo, ignoré) |
| `FS1.txt` | Débit (l/min), 10 Hz, 600 points par cycle, 2205 cycles | non (≈ 7 Mo, ignoré) |
| `profile.txt` | État des composants par cycle (cooler, **valve**, pump_leakage, accumulator, stable_flag) | oui (31 Ko) |
| `features_all_cycles.csv` | 80 variables extraites de PS2 et FS1 (généré par `python -m valve.train`) | oui (2,8 Mo) |

Les signaux bruts proviennent du jeu public *Condition monitoring of hydraulic systems* (UCI Machine Learning
Repository, ou le lien fourni dans le sujet). Pour réentraîner ou relancer le notebook, copiez `PS2.txt` et `FS1.txt` ici.
L'application web et les tests n'en ont **pas** besoin : ils utilisent `features_all_cycles.csv`.

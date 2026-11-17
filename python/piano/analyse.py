"""Analyse musicale d'un morceau avec NumPy : tonalité, tessiture, densité."""

from __future__ import annotations

import numpy as np

from .theorie import NOMS_FR, nom_francais

# Profils de Krumhansl et Kessler : poids de chaque degré dans une tonalité majeure ou mineure.
PROFIL_MAJEUR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PROFIL_MINEUR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# Orthographe usuelle des toniques altérées : ré bémol majeur plutôt que do dièse majeur, etc.
_BEMOLS_MAJEUR = {1: "ré♭", 3: "mi♭", 8: "la♭", 10: "si♭"}
_BEMOLS_MINEUR = {3: "mi♭", 10: "si♭"}


def _nom_tonique(tonique: int, mode: str) -> str:
    table = _BEMOLS_MAJEUR if mode == "majeur" else _BEMOLS_MINEUR
    return table.get(tonique, NOMS_FR[tonique].lower())


def histogramme(notes) -> np.ndarray:
    """Durée totale jouée sur chacune des douze notes de la gamme chromatique."""
    poids = np.zeros(12)
    for note in notes:
        poids[note.midi % 12] += note.duree
    return poids


def tonalite(notes) -> str:
    """Estime la tonalité par corrélation avec les 24 profils (algorithme de Krumhansl-Schmuckler)."""
    poids = histogramme(notes)
    if not poids.any():
        return "indéterminée"
    meilleure, score = "", -2.0
    for tonique in range(12):
        for profil, mode in ((PROFIL_MAJEUR, "majeur"), (PROFIL_MINEUR, "mineur")):
            correlation = np.corrcoef(poids, np.roll(profil, tonique))[0, 1]
            if correlation > score:
                meilleure, score = f"{_nom_tonique(tonique, mode)} {mode}", correlation
    return meilleure


def resumer(morceau) -> dict:
    """Quelques chiffres sur un morceau, affichés dans l'interface."""
    notes = morceau.notes
    if not notes:
        return {"notes": 0, "tonalite": "indéterminée", "tessiture": "", "notes_par_seconde": 0.0,
                "polyphonie": 0, "main_droite": 0, "main_gauche": 0}
    hauteurs = np.array([n.midi for n in notes])
    # polyphonie maximale : +1 à chaque attaque, -1 à chaque extinction
    evenements = sorted([(n.temps, 1) for n in notes] + [(n.fin, -1) for n in notes])
    polyphonie = int(np.cumsum([sens for _, sens in evenements]).max())
    return {
        "notes": len(notes),
        "tonalite": tonalite(notes),
        "tessiture": f"{nom_francais(int(hauteurs.min()))} – {nom_francais(int(hauteurs.max()))}",
        "notes_par_seconde": round(len(notes) / morceau.duree, 1) if morceau.duree else 0.0,
        "polyphonie": polyphonie,
        "main_droite": int(sum(1 for n in notes if n.main == 0)),
        "main_gauche": int(sum(1 for n in notes if n.main == 1)),
    }

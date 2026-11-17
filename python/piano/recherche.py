"""Recherche de partitions de piano en ligne, dans l'API publique de BitMidi.

Fonctions pures : l'interrogation du réseau et l'affichage sont faits par app.py.
"""

from __future__ import annotations

import html
import re
from urllib.parse import quote, urljoin

API = "https://bitmidi.com/api/midi/search"
ORIGINE = "https://bitmidi.com"


def nom_lisible(nom: str) -> str:
    """« chopin-ballade_1.mid » -> « chopin ballade 1 »."""
    texte = re.sub(r"\.midi?$", "", str(nom), flags=re.IGNORECASE)
    texte = re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", texte)).strip()
    return texte or "Sans titre"


def echapper(texte: str) -> str:
    """Les noms viennent d'Internet : ils sont échappés avant d'être affichés."""
    return html.escape(str(texte), quote=True)


def adresse(requete: str) -> str:
    return f"{API}?q={quote(requete, safe='')}&page=0"


def vers_fiche(resultat: dict) -> dict:
    """Résultat de BitMidi -> fiche de morceau, comme celles du catalogue."""
    return {
        "id": f"bitmidi-{resultat['id']}",
        "titre": nom_lisible(resultat["name"]),
        "compositeur": "BitMidi",
        "oeuvre": resultat["name"],
        "fichier": urljoin(ORIGINE, resultat["downloadUrl"]),
        "duree": 0,
        "ecoutes": resultat.get("plays") or 0,
        "enLigne": True,
    }


def fiches(reponse: dict) -> list[dict]:
    """Fiches jouables d'une réponse de l'API (les résultats sans fichier sont ignorés)."""
    resultats = (reponse.get("result") or {}).get("results") or []
    return [vers_fiche(r) for r in resultats if r.get("downloadUrl")]


def formater_nombre(n: int) -> str:
    """10781 -> « 10 781 » (espace fine insécable, comme en français)."""
    return f"{int(n):,}".replace(",", " ")

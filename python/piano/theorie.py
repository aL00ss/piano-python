"""Théorie musicale, géométrie du clavier et correspondance avec le clavier de l'ordinateur.

Fonctions pures : ce module ne dépend que de la bibliothèque standard.
"""

from __future__ import annotations

from dataclasses import dataclass

NOTE_MIN = 21  # la le plus grave d'un piano (A0)
NOTE_MAX = 108  # do le plus aigu (C8)
NB_BLANCHES = 52

NOMS_FR = ("Do", "Do♯", "Ré", "Ré♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si")
NOMS_EN = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_NOIRES = frozenset({1, 3, 6, 8, 10})

# Décalage du centre de chaque touche noire par rapport à la jointure de ses deux
# voisines blanches (en largeur de touche blanche), comme sur un vrai piano.
_DECALAGE_NOIRE = {1: -0.07, 3: 0.07, 6: -0.1, 8: 0.0, 10: 0.1}
LARGEUR_NOIRE = 0.58
HAUTEUR_NOIRE = 0.63  # part de la hauteur du clavier


def est_noire(midi: int) -> bool:
    return midi % 12 in _NOIRES


def nom_francais(midi: int) -> str:
    """Nom français, avec l'octave française (La3 = 440 Hz)."""
    return f"{NOMS_FR[midi % 12]}{midi // 12 - 2}"


def nom_anglais(midi: int) -> str:
    """Nom international (A4 = 440 Hz)."""
    return f"{NOMS_EN[midi % 12]}{midi // 12 - 1}"


def frequence(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12)


@dataclass(frozen=True)
class Touche:
    midi: int
    noire: bool
    x: float  # bord gauche, en largeurs de touche blanche
    largeur: float


def disposition() -> list[Touche]:
    """Géométrie des 88 touches : le clavier complet mesure 52 unités de large."""
    touches = []
    blanches = 0
    for midi in range(NOTE_MIN, NOTE_MAX + 1):
        if est_noire(midi):
            centre = blanches + _DECALAGE_NOIRE[midi % 12]
            touches.append(Touche(midi, True, centre - LARGEUR_NOIRE / 2, LARGEUR_NOIRE))
        else:
            touches.append(Touche(midi, False, float(blanches), 1.0))
            blanches += 1
    return touches


TOUCHES = disposition()
PAR_MIDI = {t.midi: t for t in TOUCHES}
_BLANCHES = [t for t in TOUCHES if not t.noire]
_NOIRES_TRIEES = [t for t in TOUCHES if t.noire]


def touche_sous(x: float, y: float) -> int | None:
    """Touche située sous un point (x en largeurs de touche blanche, y de 0 en haut à 1 en bas)."""
    if not (0 <= x < NB_BLANCHES and 0 <= y <= 1):
        return None
    if y <= HAUTEUR_NOIRE:
        for t in _NOIRES_TRIEES:
            if t.x <= x < t.x + t.largeur:
                return t.midi
    return _BLANCHES[int(x)].midi


def formater_duree(secondes: float) -> str:
    """185 -> « 3:05 »."""
    s = max(0, int(secondes + 0.5))
    return f"{s // 60}:{s % 60:02d}"


# ------------------------------------------------------- clavier de l'ordinateur
# On utilise le code physique des touches (KeyQ, Digit2…) : la disposition reste la
# même sur un clavier AZERTY ou QWERTY, seules les étiquettes changent.

RANGEE_BASSE = ("KeyZ", "KeyS", "KeyX", "KeyD", "KeyC", "KeyV", "KeyG", "KeyB", "KeyH", "KeyN", "KeyJ", "KeyM",
                "Comma", "KeyL", "Period", "Semicolon", "Slash")
RANGEE_HAUTE = ("KeyQ", "Digit2", "KeyW", "Digit3", "KeyE", "KeyR", "Digit5", "KeyT", "Digit6", "KeyY", "Digit7",
                "KeyU", "KeyI", "Digit9", "KeyO", "Digit0", "KeyP", "BracketLeft", "Equal", "BracketRight")

DEMI_TONS = {**{code: i for i, code in enumerate(RANGEE_BASSE)}, **{code: 12 + i for i, code in enumerate(RANGEE_HAUTE)}}

BASE_DEFAUT = 48  # do2 français (C3) : la rangée du haut démarre au do central
BASE_MIN = 24
BASE_MAX = 72

_AZERTY = {"KeyQ": "A", "KeyW": "Z", "KeyA": "Q", "KeyZ": "W", "KeyM": ",", "Comma": ";", "Period": ":",
           "Slash": "!", "Semicolon": "M", "Digit2": "é", "Digit3": '"', "Digit5": "(", "Digit6": "-",
           "Digit7": "è", "Digit9": "ç", "Digit0": "à", "BracketLeft": "^", "Equal": "=", "BracketRight": "$"}
_QWERTY = {"Comma": ",", "Period": ".", "Slash": "/", "Semicolon": ";", "BracketLeft": "[", "Equal": "=",
           "BracketRight": "]"}


def caractere(code: str, clavier: str = "azerty") -> str:
    """Caractère gravé sur une touche physique, selon la disposition du clavier."""
    table = _AZERTY if clavier == "azerty" else _QWERTY
    if code in table:
        return table[code]
    if code.startswith("Key"):
        return code[3:]
    if code.startswith("Digit"):
        return code[5:]
    return ""


def note_du_code(code: str, base: int = BASE_DEFAUT) -> int | None:
    """Note MIDI jouée par une touche physique, ou None."""
    if code not in DEMI_TONS:
        return None
    midi = base + DEMI_TONS[code]
    return midi if NOTE_MIN <= midi <= NOTE_MAX else None


def etiquettes(base: int = BASE_DEFAUT, clavier: str = "azerty") -> dict[int, str]:
    """Étiquettes à afficher sur le piano : note MIDI -> caractère du clavier."""
    resultat = {}
    for code in RANGEE_BASSE + RANGEE_HAUTE:
        midi = note_du_code(code, base)
        if midi is not None:
            resultat[midi] = caractere(code, clavier)
    return resultat


def decaler_base(base: int, octaves: int) -> int:
    """Décale le do de base d'un nombre d'octaves, sans sortir du piano."""
    return min(BASE_MAX, max(BASE_MIN, base + 12 * octaves))

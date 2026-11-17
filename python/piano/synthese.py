"""Synthèse d'un son de piano avec NumPy, sans aucun échantillon enregistré.

Chaque note est une somme de partiels (synthèse additive) qui imite la physique
d'une corde frappée :

* les partiels ne sont pas des multiples exacts de la fondamentale : la raideur de la
  corde les décale vers l'aigu (inharmonicité, f_k = k·f0·√(1 + B·k²)) ;
* le point de frappe du marteau creuse certains partiels ;
* les partiels aigus s'éteignent plus vite que les graves ;
* chaque partiel décroît en deux temps (son immédiat puis résonance) et les cordes
  d'un même chœur, très légèrement désaccordées, produisent un lent battement ;
* un bref bruit de marteau accompagne l'attaque.

Le calcul n'évalue aucun sinus échantillon par échantillon : chaque partiel est une
exponentielle complexe amortie, prolongée bloc par bloc par une simple multiplication.
"""

from __future__ import annotations

import io
import wave

import numpy as np

from .theorie import NOTE_MAX, NOTE_MIN, frequence

FE = 32000  # fréquence d'échantillonnage, en hertz
BLOC = 1024  # échantillons calculés à chaque produit matriciel
NIVEAU = 0.11  # niveau efficace visé pour l'attaque de chaque note


def duree_note(midi: int) -> float:
    """Durée synthétisée : les cordes graves résonnent bien plus longtemps."""
    return float(np.clip(7.0 * 2.0 ** (-(midi - NOTE_MIN) / 28), 1.5, 7.0))


def inharmonicite(midi: int) -> float:
    """Coefficient B de raideur de la corde (plus fort dans l'aigu et l'extrême grave)."""
    return float(2.5e-4 * np.exp(0.0926 * (midi - 60)) + 1.2e-4 * np.exp(-0.09 * (midi - NOTE_MIN)))


def partiels(midi: int, fe: int = FE):
    """Fréquences, amplitudes et constantes de temps des partiels d'une note."""
    f0 = frequence(midi)
    rangs = np.arange(1, 81, dtype=float)
    frequences = rangs * f0 * np.sqrt(1 + inharmonicite(midi) * rangs**2)
    garde = frequences < min(0.45 * fe, 11000.0)
    rangs, frequences = rangs[garde], frequences[garde]

    pente = np.interp(midi, [21, 48, 72, 108], [0.85, 1.25, 1.7, 2.3])
    amplitudes = rangs**-pente
    amplitudes *= 0.3 + 0.7 * np.abs(np.sin(np.pi * rangs / 7.5))  # marteau au 1/7,5 de la corde
    amplitudes *= np.exp(-frequences / np.interp(midi, [21, 60, 108], [2800.0, 5200.0, 9000.0]))
    if midi < 40:  # la table d'harmonie rayonne mal la fondamentale des notes très graves
        amplitudes[0] *= np.interp(midi, [21, 40], [0.35, 1.0])

    tau = 0.95 * 2.0 ** (-(midi - 60) / 21)  # extinction de la fondamentale, en secondes
    taus = tau / (1 + 0.10 * (rangs - 1) ** 1.1)
    return rangs, frequences, amplitudes, taus


def synthetiser(midi: int, fe: int = FE) -> np.ndarray:
    """Renvoie le son d'une note (tableau float32, mono, valeurs entre -1 et 1)."""
    if not NOTE_MIN <= midi <= NOTE_MAX:
        raise ValueError(f"note hors du piano : {midi}")
    hasard = np.random.default_rng(midi)
    rangs, frequences, amplitudes, taus = partiels(midi, fe)
    n = int(duree_note(midi) * fe)

    # Chaque « terme » est une sinusoïde amortie : amplitude complexe et raison par échantillon.
    amp, raison = [], []
    for rang, f, a, tau in zip(rangs, frequences, amplitudes, taus):
        # deux cordes très légèrement désaccordées pour les premiers partiels : elles partent
        # en phase (frappées par le même marteau) puis battent lentement l'une contre l'autre
        cordes = ((-1.0, 0.6), (1.0, 0.4)) if (rang <= 6 and midi >= 36) else ((0.0, 1.0),)
        phase = hasard.uniform(0, 2 * np.pi)
        for corde, part in cordes:
            ecart = 1 + corde * 2e-4
            for poids, facteur in ((0.72, 1.0), (0.28, 4.5)):  # son immédiat, puis résonance
                amp.append(a * poids * part * np.exp(1j * phase))
                raison.append(np.exp((-1 / (tau * facteur) + 2j * np.pi * f * ecart) / fe))
    coef = np.array(amp)
    raison = np.array(raison)

    puissances = np.exp(np.outer(np.log(raison), np.arange(BLOC)))  # raison ** [0, 1, …, BLOC-1]
    reel, imag = puissances.real.copy(), puissances.imag.copy()
    saut = raison**BLOC
    son = np.zeros(n)
    for debut in range(0, n, BLOC):
        m = min(BLOC, n - debut)
        # partie imaginaire de coef × puissances, sommée sur tous les termes. einsum plutôt que
        # « @ » : dans le navigateur, NumPy n'a pas de BLAS et son produit matriciel générique
        # est plusieurs fois plus lent (résultat identique à l'échantillon près).
        son[debut:debut + m] = np.einsum("k,km->m", coef.real, imag[:, :m]) + np.einsum("k,km->m", coef.imag, reel[:, :m])
        coef = coef * saut
        vivants = np.abs(coef) > 1e-7  # les partiels éteints ne sont plus calculés
        if not vivants.all():
            coef, saut, reel, imag = coef[vivants], saut[vivants], reel[vivants], imag[vivants]
            if not coef.size:
                break

    t = np.arange(n) / fe
    attaque = np.interp(midi, [21, 108], [0.004, 0.0012])
    son *= 1 - np.exp(-t / attaque)

    # bruit du marteau : un souffle filtré, très bref
    longueur = int(0.05 * fe)
    bruit = np.convolve(hasard.standard_normal(longueur), np.hanning(9) / 4.0, mode="same")
    efficace = np.sqrt(np.mean(son[: int(0.3 * fe)] ** 2)) or 1.0
    son[:longueur] += bruit * np.exp(-t[:longueur] / 0.009) * efficace * np.interp(midi, [21, 108], [0.25, 0.6])

    # même niveau perçu sur tout le clavier, puis fondu final
    efficace = np.sqrt(np.mean(son[: int(0.3 * fe)] ** 2)) or 1.0
    son *= NIVEAU / efficace
    fondu = int(0.15 * n)
    son[-fondu:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fondu)))
    return np.clip(son, -1, 1).astype(np.float32)


def reponse_de_salle(fe: int = FE, duree: float = 2.6, graine: int = 7) -> np.ndarray:
    """Réponse impulsionnelle d'une salle (2 canaux) pour la réverbération à convolution."""
    hasard = np.random.default_rng(graine)
    n = int(duree * fe)
    t = np.arange(n) / fe
    canaux = []
    for _ in range(2):
        bruit = hasard.standard_normal(n)
        doux = np.convolve(bruit, np.hanning(15) / 7.0, mode="same")  # les aigus s'éteignent d'abord
        melange = np.clip(t / 0.5, 0, 1)
        canal = ((1 - melange) * bruit + melange * doux) * np.exp(-t / (duree / 6.9))
        canal[: int(0.004 * fe)] = 0  # léger pré-délai
        canaux.append(canal)
    reponse = np.array(canaux)
    return (reponse / np.sqrt((reponse**2).sum(axis=1).max())).astype(np.float32)


def rendre(notes, fe: int = FE, relachement: float = 0.12) -> np.ndarray:
    """Mixe hors ligne une suite de notes (attributs midi, temps, fin, velocite) en un seul son."""
    notes = list(notes)
    if not notes:
        return np.zeros(0, dtype=np.float32)
    total = int((max(n.fin for n in notes) + 1.0) * fe)
    mixage = np.zeros(total)
    sons: dict[int, np.ndarray] = {}
    for note in notes:
        if note.midi not in sons:
            sons[note.midi] = synthetiser(note.midi, fe)
        son = sons[note.midi]
        tenue = int((note.fin - note.temps) * fe)
        longueur = min(len(son), tenue + int(6 * relachement * fe))
        extrait = son[:longueur].astype(float)
        if longueur > tenue:  # l'étouffoir retombe : extinction rapide
            extrait[tenue:] *= np.exp(-np.arange(longueur - tenue) / (relachement * fe))
        debut = int(note.temps * fe)
        mixage[debut:debut + longueur] += extrait[: total - debut] * (0.08 + 0.92 * note.velocite**1.6)
    crete = np.abs(mixage).max()
    return (mixage * (0.89 / crete if crete > 0.89 else 1.0)).astype(np.float32)


def ecrire_wav(son: np.ndarray, fe: int = FE) -> bytes:
    """Encode un son mono en fichier WAV 16 bits."""
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as fichier:
        fichier.setnchannels(1)
        fichier.setsampwidth(2)
        fichier.setframerate(fe)
        fichier.writeframes((np.clip(son, -1, 1) * 32767).astype("<i2").tobytes())
    return tampon.getvalue()

"""Prépare les sept morceaux joués par le piano.

Les fichiers de `outils/sources/` viennent du Mutopia Project (gravures LilyPond).
Ils contiennent les bonnes notes, mais sous une forme « mécanique » : reprises non
dépliées (les deux fins d'une reprise s'enchaînent), vélocité unique, tempo fixe,
aucune pédale. Ce script en fait une interprétation jouable :

1. dépliage des reprises (première puis seconde fin) ;
2. nuances : niveau par passage, mise en avant du chant, appuis de mesure ;
3. tempo : indication de départ, changements de mouvement, ralenti final ;
4. pédale forte (contrôleur MIDI 64), changée avec l'harmonie.

Le résultat est écrit dans `midi/` sous forme de fichiers MIDI standards.

    python outils/preparer_morceaux.py
"""

from __future__ import annotations

import json
import random
import re
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from pathlib import Path

import mido

RACINE = Path(__file__).resolve().parent.parent
SOURCES = RACINE / "outils" / "sources"
SORTIE = RACINE / "midi"

PPQ = 480  # résolution des fichiers produits (tics par noire)
MARGE = 0.1  # en noires : les petites notes qui précèdent un temps suivent ce temps

PP, P, MP, MF, F, FF = 42, 52, 63, 74, 88, 102


@dataclass(frozen=True)
class Note:
    debut: float  # en noires
    duree: float  # en noires
    hauteur: int  # numéro MIDI
    main: int  # 0 = main droite, 1 = main gauche
    source: float = 0.0  # position dans le fichier d'origine (avant dépliage)
    velocite: int = 64


@dataclass
class Morceau:
    identifiant: str
    fichier: str
    titre: str
    compositeur: str
    piste_droite: int  # numéro de la piste MIDI de la main droite
    mesure: float  # longueur d'une mesure, en noires
    signature: tuple[int, int]
    tempo: float  # noires par minute
    credit: str
    levee: float = 0.0  # anacrouse, en noires
    segments: list[tuple[float, float | None]] | None = None  # ordre de jeu (noires)
    nuances: list[tuple[float, float]] = field(default_factory=list)  # (position, vélocité)
    mouvement: list[tuple[float, float]] = field(default_factory=list)  # (position, facteur)
    pedale: list[tuple[float, float, float]] = field(default_factory=list)  # (début, fin, pas)
    chant: int = 6  # mise en avant de la voix supérieure
    dessous: int = -5  # notes intérieures de la main droite
    gauche: int = -7  # main gauche
    accents: list[float] = field(default_factory=list)  # sforzandos (positions)
    accords: int = 0  # appui sur les accords plaqués de la main droite (3 notes ou plus)
    detache: dict[float, float] = field(default_factory=dict)  # durée notée -> facteur
    annee: str = ""
    oeuvre: str = ""
    licence: str = "Domaine public"
    tempo_source: bool = False  # reprendre les changements de tempo du fichier d'origine
    pedale_source: bool = False  # reprendre la pédale notée dans le fichier d'origine
    tempos: list[tuple[float, float]] = field(default_factory=list)  # lus dans la source

    def bpm(self, position_source: float) -> float:
        """Tempo (noires par minute) à une position du fichier d'origine."""
        base = self.tempo
        for position, valeur in self.tempos:
            if position <= position_source + 1e-6:
                base = valeur
        return base * interpoler(self.mouvement, position_source, 1.0)


# --------------------------------------------------------------------------- lecture


def lire_notes(chemin: Path, piste_droite: int):
    """Lit un fichier MIDI : notes (en noires), longueur, tempos et pédale d'origine."""
    fichier = mido.MidiFile(chemin)
    tpb = fichier.ticks_per_beat
    notes: list[Note] = []
    tempos: list[tuple[float, float]] = []
    pedale: dict[float, bool] = {}
    fin = 0
    for indice, piste in enumerate(fichier.tracks):
        instant = 0
        ouvertes: dict[int, list[int]] = {}
        for message in piste:
            instant += message.time
            if message.type == "set_tempo":
                tempos.append((instant / tpb, mido.tempo2bpm(message.tempo)))
            elif message.type == "control_change" and message.control == 64:
                pedale[instant / tpb] = message.value >= 64
            elif message.type == "note_on" and message.velocity > 0:
                ouvertes.setdefault(message.note, []).append(instant)
            elif message.type in ("note_off", "note_on") and ouvertes.get(message.note):
                depart = ouvertes[message.note].pop(0)
                main = 0 if indice == piste_droite else 1
                notes.append(Note(depart / tpb, (instant - depart) / tpb, message.note, main, depart / tpb))
        fin = max(fin, instant)
    notes.sort(key=lambda n: (n.debut, n.hauteur))
    return notes, fin / tpb, sorted(tempos), sorted(pedale.items())


# ------------------------------------------------------------------------- dépliage


@dataclass(frozen=True)
class Passage:
    """Une portion du fichier d'origine placée dans le morceau déplié."""

    depart: float  # position dans le morceau déplié
    debut: float  # position d'origine
    fin: float

    def vers_source(self, position: float) -> float:
        return self.debut + (position - self.depart)


def construire_passages(segments, longueur: float) -> list[Passage]:
    passages, depart = [], 0.0
    for debut, fin in segments or [(0.0, None)]:
        fin = longueur if fin is None else fin
        passages.append(Passage(depart, debut, fin))
        depart += fin - debut
    return passages


def deplier(notes: list[Note], passages: list[Passage]) -> list[Note]:
    """Rejoue les passages dans l'ordre : c'est ici que les reprises sont dépliées."""
    resultat = []
    for rang, passage in enumerate(passages):
        dernier = rang == len(passages) - 1
        for note in notes:
            limite = float("inf") if dernier else passage.fin - MARGE
            if passage.debut - MARGE <= note.debut < limite:
                resultat.append(replace(note, debut=note.debut - passage.debut + passage.depart))
    resultat.sort(key=lambda n: (n.debut, n.hauteur))
    return resultat


def source_de(passages: list[Passage], position: float) -> float:
    departs = [p.depart for p in passages]
    passage = passages[max(0, bisect_right(departs, position + 1e-9) - 1)]
    return passage.vers_source(position)


# ----------------------------------------------------------------- nuances et tempo


def interpoler(points: list[tuple[float, float]], position: float, defaut: float) -> float:
    """Valeur d'une courbe linéaire par morceaux."""
    if not points:
        return defaut
    if position <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= position <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (position - x0) / (x1 - x0)
    return points[-1][1]


def nuancer(notes: list[Note], morceau: Morceau, graine: int) -> list[Note]:
    """Donne une vélocité à chaque note du fichier d'origine."""
    hasard = random.Random(graine)
    droite = [n for n in notes if n.main == 0]
    plaques: dict[float, int] = {}
    for n in droite:
        plaques[round(n.debut, 3)] = plaques.get(round(n.debut, 3), 0) + 1
    resultat = []
    for note in notes:
        niveau = interpoler(morceau.nuances, note.debut, MP)
        if note.main == 1:
            niveau += morceau.gauche
        else:
            # le chant : aucune note plus aiguë de la main droite ne sonne à cet instant
            couverte = any(
                autre.hauteur > note.hauteur and autre.debut <= note.debut + 1e-6 < autre.debut + autre.duree - 1e-6
                for autre in droite
            )
            niveau += morceau.dessous if couverte else morceau.chant
        position = (note.debut - morceau.levee) % morceau.mesure
        if position < 0.02 or morceau.mesure - position < 0.02:
            niveau += 3  # premier temps
        if any(a - 1e-6 <= note.debut < a + 0.25 for a in morceau.accents):
            niveau += 14
        if morceau.accords and note.main == 0 and plaques.get(round(note.debut, 3), 0) >= 3:
            niveau += morceau.accords
        niveau += hasard.uniform(-2, 2)
        resultat.append(replace(note, velocite=int(max(18, min(122, round(niveau))))))
    return resultat


def articuler(notes: list[Note], morceau: Morceau) -> list[Note]:
    """Raccourcit les notes détachées (staccato)."""
    if not morceau.detache:
        return notes
    resultat = []
    for note in notes:
        facteur = next((f for duree, f in morceau.detache.items() if abs(note.duree - duree) < 0.03), 1.0)
        resultat.append(replace(note, duree=note.duree * facteur))
    return resultat


def sans_chevauchement(notes: list[Note]) -> list[Note]:
    """Une même touche ne peut pas être rejouée avant d'avoir été relâchée."""
    par_touche: dict[tuple[int, int], list[Note]] = {}
    for note in sorted(notes, key=lambda n: n.debut):
        par_touche.setdefault((note.main, note.hauteur), []).append(note)
    resultat = []
    for groupe in par_touche.values():
        for note, suivante in zip(groupe, groupe[1:] + [None]):
            duree = note.duree
            if suivante is not None and note.debut + duree > suivante.debut - 0.01:
                duree = max(0.02, suivante.debut - note.debut - 0.01)
            resultat.append(replace(note, duree=duree))
    resultat.sort(key=lambda n: (n.debut, n.hauteur))
    return resultat


# --------------------------------------------------------------------------- pédale


def est_un_trait(notes_droite: list[Note]) -> bool:
    """Gamme, trille ou chromatisme rapide : la pédale brouillerait tout."""
    hauteurs = [n.hauteur for n in sorted(notes_droite, key=lambda n: (n.debut, n.hauteur))]
    instants = {round(n.debut, 3) for n in notes_droite}
    if len(instants) < 6 or len(hauteurs) != len(instants):
        return False
    ecarts = [abs(b - a) for a, b in zip(hauteurs, hauteurs[1:])]
    return sum(ecarts) / len(ecarts) <= 2.5


def reprendre_pedale(pedale_source, passages: list[Passage], fin: float):
    """Replace la pédale notée dans la partition sur le morceau déplié."""
    evenements: list[tuple[float, bool]] = []
    enfoncee = False
    for passage in passages:
        for position, etat in pedale_source:
            if passage.debut <= position < passage.fin and etat != enfoncee:
                evenements.append((position - passage.debut + passage.depart, etat))
                enfoncee = etat
    if enfoncee:
        evenements.append((fin, False))
    return evenements


def poser_pedale(notes: list[Note], morceau: Morceau, passages: list[Passage], fin: float):
    """Renvoie les changements de pédale : liste de (position, enfoncée)."""
    if not morceau.pedale:
        return []
    fenetres: list[tuple[float, float]] = []
    for passage in passages:
        for debut, limite, pas in morceau.pedale:
            a, b = max(debut, passage.debut), min(limite, passage.fin)
            # la grille de pédale est calée sur les mesures du fichier d'origine
            position = debut + ((a - debut) // pas) * pas
            while position < b - 1e-6:
                w0, w1 = max(position, a), min(position + pas, b)
                if w1 - w0 > 1e-6:
                    fenetres.append((w0 - passage.debut + passage.depart, w1 - passage.debut + passage.depart))
                position += pas
    fenetres.sort()
    evenements: list[tuple[float, bool]] = []
    enfoncee = False
    precedent = None
    for w0, w1 in fenetres:
        if precedent is not None and w0 - precedent > 1e-6 and enfoncee:
            evenements.append((precedent, False))  # passage sans pédale entre deux zones
            enfoncee = False
        droite = [n for n in notes if n.main == 0 and w0 - 1e-6 <= n.debut < w1 - 1e-6]
        if enfoncee:
            evenements.append((w0, False))
            enfoncee = False
        if not est_un_trait(droite):
            # pédale « syncopée » : on la reprend juste après l'attaque
            evenements.append((w0 + 0.07 * morceau.bpm(source_de(passages, w0)) / 60, True))
            enfoncee = True
        precedent = w1
    if enfoncee:
        evenements.append((max(fin, precedent or fin), False))
    return evenements


# -------------------------------------------------------------------------- écriture


def latin(texte: str) -> str:
    """Les textes d'un fichier MIDI sont en Latin-1 : on remplace ce qui n'y existe pas."""
    return texte.replace("—", "-").encode("latin-1", "replace").decode("latin-1")


def ecrire(morceau: Morceau, notes: list[Note], pedale, passages: list[Passage], fin: float) -> Path:
    fichier = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    tic = lambda noires: int(round(noires * PPQ))

    def piste(nom: str, evenements: list[tuple[int, int, mido.Message]]) -> mido.MidiTrack:
        resultat = mido.MidiTrack()
        resultat.append(mido.MetaMessage("track_name", name=latin(nom), time=0))
        precedent = 0
        for instant, _, message in sorted(evenements, key=lambda e: (e[0], e[1])):
            resultat.append(message.copy(time=instant - precedent))
            precedent = instant
        resultat.append(mido.MetaMessage("end_of_track", time=0))
        return resultat

    # Piste 0 : titre, source, mesure et tempo
    chef = [
        (0, 0, mido.MetaMessage("text", text=latin(f"{morceau.titre} - {morceau.compositeur}"))),
        (0, 0, mido.MetaMessage("copyright", text=latin(morceau.credit))),
        (0, 0, mido.MetaMessage("time_signature", numerator=morceau.signature[0], denominator=morceau.signature[1])),
    ]
    dernier_tempo = None
    position = 0.0
    while position < fin:
        bpm = morceau.bpm(source_de(passages, position))
        if dernier_tempo is None or abs(bpm - dernier_tempo) / dernier_tempo > 0.004:
            chef.append((tic(position), 1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
            dernier_tempo = bpm
        position += 0.25
    fichier.tracks.append(piste(morceau.titre, chef))

    for main, nom in ((0, "Main droite"), (1, "Main gauche")):
        evenements = []
        for note in notes:
            if note.main != main:
                continue
            debut = tic(note.debut)
            arret = max(debut + 1, tic(note.debut + note.duree))
            evenements.append((debut, 2, mido.Message("note_on", channel=main, note=note.hauteur, velocity=note.velocite)))
            evenements.append((arret, 0, mido.Message("note_off", channel=main, note=note.hauteur, velocity=0)))
        if main == 0:
            for instant, enfoncee in pedale:
                evenements.append((tic(instant), 1, mido.Message("control_change", channel=0, control=64, value=127 if enfoncee else 0)))
        fichier.tracks.append(piste(nom, evenements))

    SORTIE.mkdir(exist_ok=True)
    chemin = SORTIE / f"{morceau.identifiant}.mid"
    fichier.save(chemin)
    return chemin


# ------------------------------------------------------------------- les 5 morceaux


def nuances_lilypond(texte: str) -> tuple[list[tuple[float, float]], list[float], float]:
    """Lit une ligne de nuances LilyPond (silences invisibles + indications)."""
    niveaux = {"pp": PP, "p": P, "mp": MP, "mf": MF, "f": F, "ff": FF}
    durees = {"1": 4.0, "2": 2.0, "4": 1.0, "8": 0.5}
    marques: list[tuple[float, str]] = []
    position = 0.0
    for duree, fois, indications in re.findall(r"s(\d+)(?:\*(\d+))?((?:\\[a-z<>!]+)*)", texte):
        for indication in re.findall(r"\\([a-z<>!]+)", indications):
            marques.append((position, indication))
        position += durees[duree] * int(fois or 1)
    points: list[tuple[float, float]] = []
    accents: list[float] = []
    courant = float(P)
    for rang, (ou, marque) in enumerate(marques):
        suivante = marques[rang + 1] if rang + 1 < len(marques) else (position, "")
        if marque in niveaux:
            points += [(ou - 1e-3, courant), (ou, niveaux[marque])]
            courant = niveaux[marque]
        elif marque == "fz":
            accents.append(ou)
        elif marque in ("cresc", "<", ">"):
            cible = niveaux.get(suivante[1], courant + (-10 if marque == ">" else 10))
            if suivante[1] == "fz":
                cible = courant + 10
            points += [(ou, courant), (suivante[0] - 1e-3, cible)]
            courant = cible
    points.sort()
    return points, accents, position


NUANCES_RONDO = r"""
  s4\p s2*4
  s2*3 s4 s4\mp s2*2
  s2*5 s4 s4\p
  s2*2 s2\cresc s8\fz s8\p s4 s2*2
  s2 s4 s8\f s8 s2*3
  s2*4 s4 s4\p
  s2*2 s2\< s2\>
  s2\! s2*2 s4 s4\f
  s2*5
  s2*2 s4 s4\p s2*2
  s2\< s4\! s4\cresc s2\> s2 s2\p
  s4 s4\f s2*4
  s2*3 s4 s4\p
  s2*5
  s2*2 s4 s4\mp s2*3
  s2*4 s4 s4\p s2
  s2 s2\cresc s8\fz s8\p s4 s2*3
  s4 s4\f s2*3
  s2*4
  s4 s2 s2\f s2 s2
  s2*3 s2\f s2
  s2*4
  s2\p s2*3
  s2*3 s2\f
  s2*5
  s2*3 s2\f s4 s4\ff s2
"""


def croche(n: float) -> float:
    return n * 0.5


def mesures(longueur: float):
    return lambda numero: (numero - 1) * longueur


def catalogue() -> list[Morceau]:
    e = croche
    nuances_rondo, accents_rondo, _ = nuances_lilypond(NUANCES_RONDO)
    clair = mesures(4.5)
    lune = mesures(4.0)
    p3 = mesures(4.0)  # 3e mouvement : mesures 1 à 65
    s3 = lambda numero: numero * 4.0  # ... puis une mesure de 1re fin décale la suite
    bach = mesures(4.0)
    m4 = mesures(4.0)
    return [
        Morceau(
            identifiant="lettre-a-elise",
            annee="1810",
            oeuvre="Bagatelle n° 25 en la mineur, WoO 59",
            fichier="fur_Elise_WoO59.mid",
            titre="La Lettre à Élise",
            compositeur="Ludwig van Beethoven",
            piste_droite=1,
            mesure=1.5,
            signature=(3, 8),
            levee=0.5,
            tempo=63,
            credit="Gravure : Stelios Samelis, Mutopia Project (domaine public). Interprétation : Alexandre Besnier.",
            # reprise 1 (1re fin, 2e fin) puis reprise 2 (1re fin, 2e fin) puis la suite
            segments=[(e(0), e(22)), (e(22), e(24)), (e(0), e(22)), (e(24), e(27)),
                      (e(27), e(66)), (e(66), e(69)), (e(27), e(66)), (e(69), None)],
            nuances=[(e(0), 44), (e(26), 44), (e(27), 60), (e(36), 62), (e(39), 50), (e(42), 44), (e(69), 44),
                     (e(71), 58), (e(72), 58), (e(84), 64), (e(90), 70), (e(93), 72), (e(102), 78), (e(105), 74),
                     (e(108), 60), (e(114), 46), (e(120), 44), (e(140), 44), (e(141), 60), (e(147), 60),
                     (e(150), 50), (e(153), 44), (e(179), 44), (e(180), 52), (e(186), 60), (e(189), 68),
                     (e(192), 74), (e(195), 84), (e(201), 88), (e(203.5), 80), (e(204), 54), (e(207), 56),
                     (e(213), 66), (e(216), 72), (e(222), 84), (e(225), 86), (e(228), 70), (e(231), 56),
                     (e(234), 46), (e(240), 66), (e(243), 72), (e(246), 60), (e(249), 46), (e(252), 44),
                     (e(272), 44), (e(273), 60), (e(279), 60), (e(282), 50), (e(285), 44), (e(306), 44), (e(313), 36)],
            mouvement=[(e(0), 1.0), (e(71), 1.0), (e(72), 1.04), (e(105), 1.04), (e(114), 0.94), (e(119), 0.84),
                       (e(120), 1.0), (e(226), 1.0), (e(232), 0.88), (e(234), 1.04), (e(245), 1.04), (e(248.5), 0.86),
                       (e(249), 1.0), (e(304), 1.0), (e(309), 0.86), (e(313), 0.62)],
            pedale=[(e(0.0) - 1.0, e(313) + 1, 1.5)],
        ),
        Morceau(
            identifiant="clair-de-lune",
            annee="1905",
            oeuvre="Suite bergamasque, L. 75",
            fichier="debussy_Ste_Bergamesq_Clair.mid",
            titre="Clair de lune",
            compositeur="Claude Debussy",
            piste_droite=1,
            mesure=4.5,
            signature=(9, 8),
            tempo=64,
            credit="Gravure : Keith OHara, Mutopia Project (domaine public). Interprétation : Alexandre Besnier.",
            nuances=[(clair(1), 42), (clair(9), 44), (clair(13), 48), (clair(15), 46), (clair(19), 54),
                     (clair(21), 62), (clair(23), 74), (clair(25), 86), (clair(26), 70), (clair(27), 48),
                     (clair(31), 54), (clair(35), 62), (clair(37), 70), (clair(39), 80), (clair(41), 90),
                     (clair(42), 84), (clair(43), 66), (clair(45), 54), (clair(47), 48), (clair(51), 42),
                     (clair(59), 46), (clair(66), 42), (clair(72), 34), (clair(73), 30)],
            mouvement=[(clair(1), 1.0), (clair(13), 1.0), (clair(14.6), 0.92), (clair(15), 1.05), (clair(25), 1.12),
                       (clair(26.6), 0.92), (clair(27), 1.22), (clair(37), 1.3), (clair(41), 1.35), (clair(43), 1.12),
                       (clair(47), 0.98), (clair(50.6), 0.88), (clair(51), 1.0), (clair(65), 0.97), (clair(69), 0.86),
                       (clair(72), 0.7), (clair(73), 0.52)],
            pedale=[(clair(1), clair(15), 4.5), (clair(15), clair(27), 1.5), (clair(27), clair(73), 4.5)],
            chant=8,
            dessous=-4,
            gauche=-5,
        ),
        Morceau(
            identifiant="sonate-au-clair-de-lune",
            annee="1801",
            oeuvre="Sonate n° 14, op. 27 n° 2 — 1er mouvement",
            licence="CC BY-SA 2.5",
            fichier="moonlight1.mid",
            titre="Sonate au clair de lune — Adagio",
            compositeur="Ludwig van Beethoven",
            piste_droite=1,
            mesure=4.0,
            signature=(2, 2),
            tempo=54,
            credit="Gravure : Stewart Holmes, Mutopia Project, CC BY-SA 2.5. Adaptation : Alexandre Besnier, CC BY-SA 2.5.",
            nuances=[(lune(1), 42), (lune(5), 44), (lune(10), 48), (lune(12), 52), (lune(15), 46), (lune(16), 52),
                     (lune(18), 56), (lune(19), 48), (lune(23), 46), (lune(25), 54), (lune(27), 60), (lune(28), 50),
                     (lune(32), 52), (lune(34), 62), (lune(36), 72), (lune(38), 64), (lune(40), 52), (lune(42), 44),
                     (lune(46), 48), (lune(48), 56), (lune(49), 60), (lune(51), 50), (lune(55), 48), (lune(58), 52),
                     (lune(60), 44), (lune(66), 38), (lune(70), 30)],
            mouvement=[(lune(1), 1.0), (lune(40), 1.0), (lune(41.8), 0.93), (lune(42), 1.0), (lune(65), 1.0),
                       (lune(67), 0.92), (lune(69), 0.76), (lune(70), 0.56)],
            pedale=[(lune(1), lune(70), 2.0)],
            chant=12,
            dessous=-4,
            gauche=-2,
        ),
        Morceau(
            identifiant="sonate-au-clair-de-lune-presto",
            annee="1801",
            oeuvre="Sonate n° 14, op. 27 n° 2 — 3e mouvement",
            licence="CC BY-SA 2.5",
            fichier="moonlight3.mid",
            titre="Sonate au clair de lune — Presto",
            compositeur="Ludwig van Beethoven",
            piste_droite=1,
            mesure=4.0,
            signature=(4, 4),
            tempo=158,
            credit="Gravure : Chris Sawer et Stewart Holmes, Mutopia Project, CC BY-SA 2.5. Adaptation : Alexandre Besnier, CC BY-SA 2.5.",
            # mesure 1, exposition (2 à 64) avec sa 1re fin, reprise, puis 2e fin et la suite
            segments=[(0, 256), (256, 260), (4, 256), (260, None)],
            nuances=[(p3(1), 54), (p3(8.9), 56), (p3(9), 84), (p3(14), 92), (p3(14.9), 92), (p3(15), 54),
                     (p3(19), 62), (p3(20.9), 66), (p3(21), 56), (p3(25), 66), (p3(29), 88), (p3(33), 80),
                     (p3(37), 60), (p3(41), 70), (p3(43), 88), (p3(49), 84), (p3(56.9), 84), (p3(57), 58),
                     (p3(61), 68), (p3(63), 78), (p3(64.99), 82), (256, 86), (257, 58), (260, 86), (261, 58),
                     (s3(71), 60), (s3(75), 66), (s3(79), 72), (s3(87), 60), (s3(95), 52), (s3(99), 60),
                     (s3(100), 56), (s3(101), 46), (s3(101.99), 44), (s3(102), 84), (s3(102.25), 54),
                     (s3(109.9), 56), (s3(110), 84), (s3(115), 92), (s3(115.99), 92), (s3(116), 54),
                     (s3(120), 66), (s3(124), 88), (s3(131.9), 84), (s3(132), 60), (s3(136.9), 66),
                     (s3(137), 88), (s3(150.9), 84), (s3(151), 58), (s3(157), 72), (s3(159), 84), (s3(163), 96),
                     (s3(165), 100), (s3(166.9), 96), (s3(167), 60), (s3(171), 70), (s3(177), 88), (s3(183), 96),
                     (s3(187), 100), (s3(188), 72), (s3(188.99), 50), (s3(189), 46), (s3(190), 44), (s3(191), 54),
                     (s3(195), 62), (s3(196.9), 70), (s3(197), 92), (s3(200), 106), (s3(203), 110)],
            mouvement=[(0, 1.0), (p3(14), 1.0), (p3(14.25), 0.7), (p3(14.98), 0.5), (p3(15), 1.0),
                       (s3(115), 1.0), (s3(115.25), 0.7), (s3(115.98), 0.5), (s3(116), 1.0),
                       (s3(165), 1.0), (s3(165.25), 0.62), (s3(165.98), 0.5), (s3(166), 1.0),
                       (s3(167), 1.0), (s3(167.25), 0.62), (s3(167.98), 0.5), (s3(168), 1.0),
                       (s3(186.9), 1.0), (s3(187), 0.9), (s3(188), 0.85), (s3(188.9), 0.5), (s3(189), 0.3),
                       (s3(190.98), 0.28), (s3(191), 1.0), (s3(199), 1.0), (s3(200), 0.9), (s3(203), 0.8)],
            pedale=[(0, 804, 1.0)],
            chant=6,
            dessous=-4,
            gauche=-6,
            accords=12,
            detache={0.5: 0.62},
        ),
        Morceau(
            identifiant="marche-turque",
            annee="1783",
            oeuvre="Sonate n° 11, K. 331 — Rondo alla turca",
            fichier="KV331_3_RondoAllaTurca.mid",
            titre="Marche turque",
            compositeur="Wolfgang Amadeus Mozart",
            piste_droite=1,
            mesure=2.0,
            signature=(2, 4),
            levee=1.0,
            tempo=126,
            credit="Gravure : Rune Zedeler et Chris Sawer, Mutopia Project (domaine public). Interprétation : Alexandre Besnier.",
            # A A | B B | C C | D D | E E | C C | A A | B B | C' 1re fin | C' 2e fin + coda
            segments=[(0, 16), (0, 16), (16, 48), (16, 48), (48, 64), (48, 64), (64, 80), (64, 80),
                      (80, 112), (80, 112), (112, 128), (112, 128), (128, 144), (128, 144),
                      (144, 176), (144, 176), (176, 191), (191, 192), (176, 191), (192, None)],
            nuances=nuances_rondo,
            accents=accents_rondo,
            mouvement=[(0, 1.0), (249, 1.0), (253, 0.9), (255, 0.78)],
            chant=5,
            dessous=-3,
            gauche=-8,
            detache={0.5: 0.55},
        ),
        Morceau(
            identifiant="fantaisie-impromptu",
            annee="1834",
            oeuvre="Fantaisie-Impromptu en do dièse mineur, op. 66",
            fichier="chopin_fantaisie-impromptu.mid",
            titre="Fantaisie-Impromptu",
            compositeur="Frédéric Chopin",
            piste_droite=1,
            mesure=4.0,
            signature=(4, 4),
            tempo=168,
            tempo_source=True,
            pedale_source=True,
            credit="Gravure : Guy D. Lederfein, Mutopia Project (domaine public). Interprétation : Alexandre Besnier.",
            nuances=[(m4(1), 96), (m4(2.9), 90), (m4(3), 48), (m4(5), 52), (m4(7), 62), (m4(8), 66), (m4(9), 52),
                     (m4(11), 62), (m4(12), 68), (m4(13), 80), (m4(17), 70), (m4(19), 62), (m4(21), 56),
                     (m4(23), 50), (m4(25), 52), (m4(29), 60), (m4(33), 72), (m4(35), 84), (m4(37), 100),
                     (m4(40), 96), (m4(41), 80), (m4(42), 60), (m4(43), 46), (m4(51), 54), (m4(55), 48),
                     (m4(59), 56), (m4(63), 50), (m4(67), 58), (m4(71), 52), (m4(75), 56), (m4(79), 48),
                     (m4(82), 44), (m4(83), 52), (m4(85), 62), (m4(86), 66), (m4(87), 52), (m4(89), 62),
                     (m4(90), 68), (m4(91), 80), (m4(95), 70), (m4(97), 62), (m4(99), 56), (m4(101), 50),
                     (m4(103), 52), (m4(107), 60), (m4(111), 72), (m4(113), 84), (m4(115), 100), (m4(119), 104),
                     (m4(123), 90), (m4(127), 70), (m4(129), 54), (m4(135), 46), (m4(137), 38), (m4(139), 30)],
            mouvement=[(m4(1), 1.0), (m4(39), 1.0), (m4(40.9), 0.84), (m4(41), 1.0), (m4(42.99), 1.0),
                       (m4(43), 0.84), (m4(81), 0.84), (m4(82.9), 0.7), (m4(83), 1.0), (m4(135.9), 1.0),
                       (m4(136), 0.95), (m4(138), 0.8), (m4(139), 0.6)],
            chant=7,
            dessous=-4,
            gauche=-8,
        ),
        Morceau(
            identifiant="prelude-en-do",
            annee="1722",
            oeuvre="Le Clavier bien tempéré, BWV 846",
            fichier="wtk1-prelude1.mid",
            titre="Prélude en do majeur",
            compositeur="Johann Sebastian Bach",
            piste_droite=2,
            mesure=4.0,
            signature=(4, 4),
            tempo=68,
            credit="Gravure : Tobias Erbsland, Mutopia Project (domaine public). Interprétation : Alexandre Besnier.",
            nuances=[(bach(1), 56), (bach(5), 60), (bach(8), 56), (bach(12), 60), (bach(16), 54), (bach(20), 60),
                     (bach(24), 68), (bach(28), 74), (bach(31), 72), (bach(32), 64), (bach(33), 60), (bach(36), 48)],
            mouvement=[(bach(1), 1.0), (bach(32), 1.0), (bach(33), 0.92), (bach(34), 0.82), (bach(35), 0.66), (bach(36), 0.5)],
            pedale=[(bach(1), bach(36), 2.0)],
            chant=3,
            dessous=-2,
            gauche=-3,
        ),
    ]


def preparer(morceau: Morceau, graine: int = 1810) -> dict:
    notes, longueur, tempos, pedale_source = lire_notes(SOURCES / morceau.fichier, morceau.piste_droite)
    morceau.tempos = tempos if morceau.tempo_source else []
    notes = articuler(nuancer(notes, morceau, graine), morceau)
    passages = construire_passages(morceau.segments, longueur)
    depliees = sans_chevauchement(deplier(notes, passages))
    fin = max(n.debut + n.duree for n in depliees)
    if morceau.pedale_source:
        pedale = reprendre_pedale(pedale_source, passages, fin)
    else:
        pedale = poser_pedale(depliees, morceau, passages, fin)
    chemin = ecrire(morceau, depliees, pedale, passages, fin)
    duree = mido.MidiFile(chemin).length
    return {"fichier": chemin.name, "notes_source": len(notes), "notes": len(depliees), "duree": duree,
            "pedales": sum(1 for _, enfoncee in pedale if enfoncee),
            "grave": min(n.hauteur for n in depliees), "aigu": max(n.hauteur for n in depliees)}


def main() -> None:
    fiches = []
    for morceau in catalogue():
        bilan = preparer(morceau)
        minutes, secondes = divmod(round(bilan["duree"]), 60)
        fiches.append({
            "id": morceau.identifiant, "titre": morceau.titre, "compositeur": morceau.compositeur,
            "oeuvre": morceau.oeuvre, "annee": morceau.annee, "fichier": f"midi/{bilan['fichier']}",
            "duree": round(bilan["duree"], 1), "notes": bilan["notes"], "grave": bilan["grave"],
            "aigu": bilan["aigu"], "credit": morceau.credit, "licence": morceau.licence,
        })
        print(f"{bilan['fichier']:32s} {bilan['notes']:5d} notes ({bilan['notes_source']} dans la source)  "
              f"{minutes} min {secondes:02d} s  {bilan['pedales']} coups de pédale")
    texte = json.dumps(fiches, ensure_ascii=False, indent=2) + "\n"
    (SORTIE / "catalogue.json").write_text(texte, encoding="utf-8")


if __name__ == "__main__":
    main()

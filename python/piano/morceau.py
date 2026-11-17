"""Lecture d'un fichier MIDI avec mido et préparation des notes à jouer."""

from __future__ import annotations

import io
from bisect import bisect_left
from dataclasses import dataclass, field

CONTROLEUR_PEDALE = 64
CANAL_PERCUSSIONS = 9  # canal 10 en General MIDI : la batterie, injouable au piano
NOTE_MIN, NOTE_MAX = 21, 108  # les 88 touches


@dataclass
class Note:
    midi: int
    temps: float  # instant de l'attaque, en secondes
    duree: float  # durée pendant laquelle la touche est enfoncée
    velocite: float  # de 0 à 1
    main: int  # 0 = main droite, 1 = main gauche
    fin: float = 0.0  # instant où la note cesse de sonner (pédale comprise)


@dataclass
class Morceau:
    notes: list[Note]
    pedale: list[tuple[float, float]]  # intervalles [début, fin[ où la pédale est enfoncée
    duree: float
    tenue_max: float
    titre: str = ""
    temps: list[float] = field(default_factory=list, repr=False)  # attaques, pour la recherche dichotomique

    def indice_a(self, instant: float) -> int:
        """Indice de la première note dont l'attaque est >= `instant`."""
        return bisect_left(self.temps, instant)

    def touches_enfoncees(self, position: float, duree_min: float = 0.09) -> list[Note]:
        """Notes dont la touche est enfoncée à l'instant `position`."""
        actives = []
        for i in range(self.indice_a(position - self.tenue_max - duree_min), len(self.notes)):
            note = self.notes[i]
            if note.temps > position:
                break
            if position < note.temps + max(note.duree, duree_min):
                actives.append(note)
        return actives

    def pedale_a(self, position: float) -> bool:
        return any(debut <= position < fin for debut, fin in self.pedale)


def intervalles_pedale(evenements: list[tuple[float, bool]]) -> list[tuple[float, float]]:
    """Transforme les changements de pédale en intervalles [début, fin[."""
    intervalles = []
    debut = None
    for instant, enfoncee in sorted(evenements):
        if enfoncee and debut is None:
            debut = instant
        elif not enfoncee and debut is not None:
            intervalles.append((debut, instant))
            debut = None
    if debut is not None:
        intervalles.append((debut, float("inf")))
    return intervalles


def preparer(notes: list[Note], evenements_pedale: list[tuple[float, bool]] = (), titre: str = "") -> Morceau:
    """Calcule pour chaque note l'instant où elle cesse réellement de sonner.

    Une touche relâchée pendant que la pédale est enfoncée continue de résonner jusqu'au
    relâchement de la pédale, sauf si la même touche est rejouée avant.
    """
    notes = sorted(notes, key=lambda n: (n.temps, n.midi))
    intervalles = intervalles_pedale(list(evenements_pedale))
    debuts = [debut for debut, _ in intervalles]
    for note in notes:
        lache = note.temps + note.duree
        note.fin = lache
        rang = bisect_left(debuts, lache + 1e-12) - 1  # dernier intervalle commençant avant le relâchement
        if rang >= 0 and intervalles[rang][0] <= lache < intervalles[rang][1]:
            note.fin = intervalles[rang][1]

    prochaine: dict[int, float] = {}
    for note in reversed(notes):
        suivante = prochaine.get(note.midi)
        if suivante is not None and note.temps < suivante < note.fin:
            note.fin = suivante
        prochaine[note.midi] = note.temps

    duree = 0.0
    tenue_max = 0.0
    for note in notes:
        if note.fin == float("inf"):
            note.fin = note.temps + note.duree
        note.fin = max(note.fin, note.temps + 0.03)
        duree = max(duree, note.fin)
        tenue_max = max(tenue_max, note.duree)
    finis = [(debut, fin if fin != float("inf") else duree) for debut, fin in intervalles]
    return Morceau(notes, finis, duree, tenue_max, titre, [n.temps for n in notes])


def charger(octets: bytes, titre: str = "") -> Morceau:
    """Lit un fichier MIDI (contenu binaire) avec mido.

    Les instants sont convertis de tics en secondes en suivant la carte de tempo du
    fichier : le morceau est donc joué au tempo voulu, changements de mouvement compris.
    Les fichiers trouvés en ligne peuvent contenir d'autres instruments : la batterie et les
    notes hors des 88 touches sont ignorées, tout le reste est joué au piano.
    """
    import mido

    # clip : les fichiers trouvés en ligne ne sont pas toujours parfaitement conformes
    fichier = mido.MidiFile(file=io.BytesIO(octets), clip=True)
    # Les pistes sont fusionnées en gardant, pour chaque message, sa piste d'origine.
    messages = []
    for indice, piste in enumerate(fichier.tracks):
        tic = 0
        for message in piste:
            tic += message.time
            messages.append((tic, indice, message))
    messages.sort(key=lambda m: m[0])

    tempo = 500000  # microsecondes par noire (120 à la noire) tant que le fichier ne dit rien
    secondes = 0.0
    dernier_tic = 0
    ouvertes: dict[tuple[int, int], list[tuple[float, float]]] = {}
    brutes: list[tuple[int, Note]] = []
    pedale: list[tuple[float, bool]] = []
    for tic, piste, message in messages:
        secondes += mido.tick2second(tic - dernier_tic, fichier.ticks_per_beat, tempo)
        dernier_tic = tic
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type in ("note_on", "note_off") and (
                message.channel == CANAL_PERCUSSIONS or not NOTE_MIN <= message.note <= NOTE_MAX):
            continue
        elif message.type == "note_on" and message.velocity > 0:
            ouvertes.setdefault((piste, message.note), []).append((secondes, message.velocity / 127))
        elif message.type in ("note_off", "note_on"):
            en_cours = ouvertes.get((piste, message.note))
            if en_cours:
                debut, velocite = en_cours.pop(0)
                brutes.append((piste, Note(message.note, debut, secondes - debut, velocite, 0)))
        elif message.type == "control_change" and message.control == CONTROLEUR_PEDALE:
            pedale.append((secondes, message.value >= 64))

    # La première piste qui contient des notes est la main droite.
    pistes = sorted({piste for piste, _ in brutes})
    for piste, note in brutes:
        note.main = min(1, pistes.index(piste))
    return preparer([note for _, note in brutes], pedale, titre)

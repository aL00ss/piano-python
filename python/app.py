"""L'application dans le navigateur, entièrement écrite en Python.

Pyodide exécute ce fichier dans la page. Le module `js` donne accès au DOM et à la Web Audio
API : c'est Python qui dessine le clavier, écoute le clavier et les pointeurs, synthétise le
son des 88 notes avec NumPy, lit les fichiers MIDI avec mido et programme chaque note.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import numpy as np

from js import Float32Array, Object, Response, Uint8Array, document, fetch, navigator, performance, window
from pyodide.ffi import create_once_callable, create_proxy, to_js

from piano import morceau as lecture_midi
from piano import synthese
from piano.analyse import resumer
from piano.theorie import (BASE_DEFAUT, HAUTEUR_NOIRE, NB_BLANCHES, NOTE_MAX, NOTE_MIN, PAR_MIDI, TOUCHES,
                           caractere, decaler_base, etiquettes, formater_duree, nom_francais, note_du_code,
                           touche_sous)

PERIODE = 25  # ms entre deux passes de programmation
HORIZON = 0.25  # secondes programmées à l'avance
HORIZON_ARRIERE_PLAN = 2.5  # onglet masqué : les minuteries sont ralenties
DELAI_DEPART = 0.12
RELACHEMENT = 0.09  # constante de temps de l'étouffoir, en secondes
LARGEUR_MIN_SOURIS = 22  # px par touche blanche
LARGEUR_MIN_TACTILE = 40

CACHE_PREFIXE = "piano-python-sons-"
MESSAGE_SANS_SON = "Aucune sortie audio utilisable : le son reviendra tout seul dès qu'une sortie sera disponible."

_proxys = []  # les fonctions Python confiées au navigateur doivent rester en vie


def proxy(fonction):
    p = create_proxy(fonction)
    _proxys.append(p)
    return p


def ecouter(cible, evenement: str, fonction) -> None:
    cible.addEventListener(evenement, proxy(fonction))


def horloge() -> float:
    return performance.now() / 1000


def lire_preference(cle: str, defaut):
    """Réglage mémorisé d'une visite à l'autre (le navigateur peut refuser le stockage)."""
    try:
        valeur = window.localStorage.getItem(f"piano.{cle}")
        return defaut if valeur is None else json.loads(valeur)
    except Exception:
        return defaut


def ecrire_preference(cle: str, valeur) -> None:
    try:
        window.localStorage.setItem(f"piano.{cle}", json.dumps(valeur))
    except Exception:
        pass  # stockage indisponible : le réglage vaut pour cette visite


def gain_de_velocite(velocite: float) -> float:
    """Vélocité (0 à 1) -> gain : un pianissimo reste audible, un fortissimo ne sature pas."""
    v = min(1.0, max(0.0, velocite))
    return 0.08 + 0.92 * v**1.6


# ============================================================================== son


class Moteur:
    """Joue les notes synthétisées par NumPy à travers la Web Audio API."""

    def __init__(self) -> None:
        self.tampons = {}  # midi -> AudioBuffer (utilisable par n'importe quel contexte audio)
        self.voix = {}  # midi -> voix du jeu en direct
        self.enfoncees: set[int] = set()
        self.tenues: set[int] = set()
        self.pedale_enfoncee = False
        self.notes_jouees = 0
        self.pret = False
        self.reconnexions = 0
        self.niveau_volume = 0.9
        self.salle = False  # la réverbération a-t-elle été installée ?
        self._onde = Float32Array.new(1024)
        self._construire()

    def _construire(self) -> None:
        """Crée le contexte audio et la chaîne d'effets."""
        constructeur = getattr(window, "AudioContext", None) or window.webkitAudioContext
        self.ctx = constructeur.new(to_js({"latencyHint": "interactive"}, dict_converter=Object.fromEntries))
        c = self.ctx
        self.sortie = c.createGain()
        self.sortie.gain.value = self.niveau_volume
        compresseur = c.createDynamicsCompressor()
        compresseur.threshold.value = -16
        compresseur.ratio.value = 3
        compresseur.attack.value = 0.004
        compresseur.release.value = 0.2
        self.analyseur = c.createAnalyser()
        self.analyseur.fftSize = 1024
        self.sortie.connect(compresseur)
        compresseur.connect(c.destination)
        compresseur.connect(self.analyseur)
        self.bus = c.createGain()
        self.bus.connect(self.sortie)
        self.reverb = c.createConvolver()
        self.reverb.connect(self.sortie)
        envoi = c.createGain()
        envoi.gain.value = 0.2
        self.bus.connect(envoi)
        envoi.connect(self.reverb)

    def reconnecter(self) -> None:
        """Repart d'un contexte audio neuf, quand la sortie audio change.

        Un contexte créé sans sortie utilisable (Firefox, par exemple) peut rester suspendu pour
        toujours. Les sons déjà synthétisés sont réutilisés ; seule la réverbération est recalculée,
        car elle doit avoir la fréquence d'échantillonnage du nouveau contexte.
        """
        ancien = self.ctx
        self.voix.clear()
        self.enfoncees.clear()
        self.tenues.clear()
        self._construire()
        if self.salle:
            fe = int(self.ctx.sampleRate)
            self.installer_salle(synthese.reponse_de_salle(fe), fe)
        self.reconnexions += 1
        try:
            ancien.close()
        except Exception:
            pass

    @property
    def actif(self) -> bool:
        return self.ctx.state == "running"

    def _tampon(self, canaux, fe: int):
        tampon = self.ctx.createBuffer(len(canaux), len(canaux[0]), fe)
        for indice, canal in enumerate(canaux):
            tampon.copyToChannel(to_js(canal), indice)  # tableau NumPy float32 -> Float32Array
        return tampon

    def installer_salle(self, reponse, fe: int) -> None:
        self.reverb.buffer = self._tampon(list(reponse), fe)
        self.salle = True

    def ajouter_note(self, midi: int, son, fe: int) -> None:
        self.tampons[midi] = self._tampon([son], fe)

    def volume(self, valeur: float) -> None:
        self.niveau_volume = min(1.2, max(0.0, valeur))
        self.sortie.gain.setTargetAtTime(self.niveau_volume, self.ctx.currentTime, 0.02)

    def demarrer(self) -> None:
        """À appeler depuis un geste de l'utilisateur : les navigateurs bloquent le son avant."""
        if self.ctx.state != "running":
            try:
                self.ctx.resume()
            except Exception:
                pass

    def maintenant(self) -> float:
        return self.ctx.currentTime

    def _voix(self, midi: int, quand: float, velocite: float, destination):
        """Lance une note : source -> filtre (le timbre s'éclaircit avec la force) -> gain."""
        c = self.ctx
        source = c.createBufferSource()
        source.buffer = self.tampons[midi]
        filtre = c.createBiquadFilter()
        filtre.type = "lowpass"
        filtre.frequency.value = 900 + 13000 * min(1.0, velocite) ** 1.5
        filtre.Q.value = 0.4
        gain = c.createGain()
        gain.gain.value = gain_de_velocite(velocite)
        source.connect(filtre)
        filtre.connect(gain)
        gain.connect(destination)
        source.start(quand)
        self.notes_jouees += 1
        return source, gain

    @staticmethod
    def _etouffer(voix, quand: float) -> None:
        source, gain = voix
        gain.gain.setTargetAtTime(0, quand, RELACHEMENT)
        source.stop(quand + 8 * RELACHEMENT)

    # ---------------------------------------------------------------- jeu en direct

    def attaquer(self, midi: int, velocite: float = 0.75) -> None:
        if midi not in self.tampons:
            return
        instant = self.ctx.currentTime
        precedente = self.voix.pop(midi, None)
        if precedente:
            self._etouffer(precedente, instant)
        self.tenues.discard(midi)
        self.enfoncees.add(midi)
        self.voix[midi] = self._voix(midi, instant, velocite, self.bus)

    def relacher(self, midi: int) -> None:
        if midi not in self.enfoncees:
            return
        self.enfoncees.discard(midi)
        if self.pedale_enfoncee:
            self.tenues.add(midi)
        else:
            voix = self.voix.pop(midi, None)
            if voix:
                self._etouffer(voix, self.ctx.currentTime)

    def pedale(self, enfoncee: bool) -> None:
        """Pédale forte : les notes relâchées résonnent tant qu'elle est enfoncée."""
        self.pedale_enfoncee = enfoncee
        if enfoncee:
            return
        for midi in list(self.tenues):
            if midi not in self.enfoncees:
                voix = self.voix.pop(midi, None)
                if voix:
                    self._etouffer(voix, self.ctx.currentTime)
        self.tenues.clear()

    # ------------------------------------------------------- lecture d'un morceau

    def creer_session(self) -> "Session":
        return Session(self)

    def niveau(self) -> float:
        """Niveau efficace du signal de sortie (0 = silence)."""
        self.analyseur.getFloatTimeDomainData(self._onde)
        onde = self._onde.to_py()
        return (sum(v * v for v in onde) / len(onde)) ** 0.5


class Session:
    """Les notes programmées d'une lecture. L'arrêter coupe aussi celles prévues dans le futur."""

    def __init__(self, moteur: Moteur) -> None:
        self.moteur = moteur
        self.gain = moteur.ctx.createGain()
        self.gain.connect(moteur.bus)
        self.ouverte = True

    def jouer(self, midi: int, quand: float, duree: float, velocite: float) -> None:
        if not self.ouverte or midi not in self.moteur.tampons:
            return
        voix = self.moteur._voix(midi, quand, velocite, self.gain)
        self.moteur._etouffer(voix, quand + max(0.03, duree))

    def arreter(self) -> None:
        if not self.ouverte:
            return
        self.ouverte = False
        self.gain.gain.setTargetAtTime(0, self.moteur.ctx.currentTime, 0.02)
        gain = self.gain
        window.setTimeout(create_once_callable(lambda: gain.disconnect()), 400)


class Lecteur:
    """Programme les notes un peu à l'avance sur l'horloge audio, précise à l'échantillon près."""

    def __init__(self, moteur: Moteur, sur_etat) -> None:
        self.moteur = moteur
        self.sur_etat = sur_etat
        self.morceau = None
        self.session = None
        self.minuterie = None
        self.etat = "arret"  # "arret" | "lecture" | "pause"
        self.vitesse = 1.0
        self.origine_audio = 0.0  # instant audio correspondant à origine_position
        self.origine_position = 0.0
        self.prochaine = 0  # indice de la prochaine note à programmer
        self._passe = proxy(lambda *_: self.programmer())
        document.addEventListener("visibilitychange", self._passe)

    @property
    def duree(self) -> float:
        return self.morceau.duree if self.morceau else 0.0

    @property
    def position(self) -> float:
        """Position de lecture, en secondes du morceau."""
        if self.etat != "lecture":
            return self.origine_position
        ecoule = (self.moteur.maintenant() - self.origine_audio) * self.vitesse
        return min(self.duree, self.origine_position + ecoule)

    def charger(self, morceau) -> None:
        self.arreter()
        self.morceau = morceau

    def jouer(self, depuis: float = 0.0) -> None:
        if not self.morceau:
            return
        self._couper()
        self.origine_position = max(0.0, min(depuis, self.duree))
        self.origine_audio = self.moteur.maintenant() + DELAI_DEPART
        self.prochaine = self.morceau.indice_a(self.origine_position)
        self.session = self.moteur.creer_session()
        self._changer("lecture")
        self.minuterie = window.setInterval(self._passe, PERIODE)
        self.programmer()

    def arreter(self) -> None:
        self._couper()
        self.origine_position = 0.0
        if self.etat != "arret":
            self._changer("arret")

    def pause(self) -> None:
        """Met la lecture en pause : la position est conservée."""
        if self.etat != "lecture":
            return
        position = self.position
        self._couper()
        self.origine_position = position
        self._changer("pause")

    def reprendre(self) -> None:
        """Reprend là où la pause a arrêté la lecture."""
        if self.etat == "pause":
            self.jouer(self.origine_position)

    def chercher(self, position: float) -> None:
        """Saute à une position (en secondes) ; en pause, la lecture reprendra de là."""
        if self.etat == "lecture":
            self.jouer(position)
        elif self.etat == "pause":
            self.origine_position = max(0.0, min(position, self.duree))

    def changer_vitesse(self, valeur: float) -> None:
        vitesse = min(4.0, max(0.25, valeur))
        if self.etat == "lecture":
            position = self.position
            self.vitesse = vitesse
            self.jouer(position)
        else:
            self.vitesse = vitesse

    def _couper(self) -> None:
        if self.minuterie is not None:
            window.clearInterval(self.minuterie)
            self.minuterie = None
        if self.session:
            self.session.arreter()
            self.session = None

    def _changer(self, etat: str) -> None:
        self.etat = etat
        self.sur_etat(etat)

    def programmer(self) -> None:
        """Programme toutes les notes qui démarrent dans l'horizon à venir."""
        if self.etat != "lecture" or not self.session:
            return
        maintenant = self.moteur.maintenant()
        horizon = HORIZON_ARRIERE_PLAN if document.hidden else HORIZON
        limite = self.origine_position + (maintenant + horizon - self.origine_audio) * self.vitesse
        notes = self.morceau.notes
        while self.prochaine < len(notes) and notes[self.prochaine].temps < limite:
            note = notes[self.prochaine]
            self.prochaine += 1
            quand = self.origine_audio + (note.temps - self.origine_position) / self.vitesse
            duree = (note.fin - note.temps) / self.vitesse
            self.session.jouer(note.midi, max(quand, maintenant), duree, note.velocite)
        if self.position >= self.duree:
            # les dernières notes finissent de résonner : on ne coupe pas la session tout de suite
            window.clearInterval(self.minuterie)
            self.minuterie = None
            session, self.session = self.session, None
            window.setTimeout(create_once_callable(lambda: session.arreter()), 1500)
            self.origine_position = 0.0
            self._changer("arret")


# ========================================================================== clavier


class Piano:
    """Le clavier à l'écran : 88 touches jouables à la souris, au doigt et en glissando."""

    def __init__(self, racine, sur_attaque, sur_relache, sur_geometrie) -> None:
        self.racine = racine
        self.sur_attaque = sur_attaque
        self.sur_relache = sur_relache
        self.sur_geometrie = sur_geometrie
        self.elements = {}  # midi -> élément
        self.pointeurs: dict[int, int] = {}  # identifiant du pointeur -> midi
        self.largeur = 24.0
        self.decalage = 0.0
        self.tactile = bool(window.matchMedia("(pointer: coarse)").matches)

        self.piste = document.createElement("div")
        self.piste.className = "clavier-piste"
        for t in TOUCHES:
            el = document.createElement("div")
            el.className = f"touche {'noire' if t.noire else 'blanche'} attente"
            el.dataset.midi = str(t.midi)
            el.style.setProperty("--x", str(t.x))
            el.style.setProperty("--l", str(t.largeur))
            lettre = document.createElement("span")
            lettre.className = "lettre"
            el.append(lettre)
            if t.midi % 12 == 0:
                nom = document.createElement("span")
                nom.className = "nom"
                nom.textContent = nom_francais(t.midi)
                el.append(nom)
            self.elements[t.midi] = el
            self.piste.append(el)
        racine.append(self.piste)

        ecouter(racine, "pointerdown", self._appui)
        ecouter(racine, "pointermove", self._glisse)
        for evenement in ("pointerup", "pointercancel", "lostpointercapture"):
            ecouter(racine, evenement, self._leve)
        ecouter(racine, "contextmenu", lambda e: e.preventDefault())
        ecouter(window, "resize", lambda _e: self.redimensionner())
        self.redimensionner()
        self.centrer_sur(60, anime=False)

    # -------------------------------------------------------------------- géométrie

    @property
    def largeur_visible(self) -> float:
        return self.racine.clientWidth

    @property
    def largeur_totale(self) -> float:
        return self.largeur * NB_BLANCHES

    @property
    def deborde(self) -> bool:
        """Vrai si le clavier dépasse de l'écran (on peut alors le faire défiler)."""
        return self.largeur_totale > self.largeur_visible + 1

    def redimensionner(self) -> None:
        visible = self.largeur_visible
        if not visible:
            return
        minimum = LARGEUR_MIN_TACTILE if self.tactile else LARGEUR_MIN_SOURIS
        centre = (self.decalage + visible / 2) / self.largeur
        self.largeur = max(visible / NB_BLANCHES, minimum)
        self.piste.style.setProperty("--w", f"{self.largeur}px")
        self.piste.style.width = f"{self.largeur_totale}px"
        self.definir_decalage(centre * self.largeur - visible / 2, force=True)

    def definir_decalage(self, px: float, force: bool = False) -> None:
        maximum = max(0.0, self.largeur_totale - self.largeur_visible)
        valeur = min(maximum, max(0.0, px))
        if not force and abs(valeur - self.decalage) < 0.05:
            return
        self.decalage = valeur
        self.piste.style.transform = f"translate3d({-valeur}px, 0, 0)"
        self.sur_geometrie()

    def centrer_sur(self, midi: int, anime: bool = True) -> None:
        t = PAR_MIDI[max(NOTE_MIN, min(NOTE_MAX, midi))]
        self.piste.classList.toggle("anime", anime)
        self.definir_decalage((t.x + 3.5) * self.largeur - self.largeur_visible / 2)

    # -------------------------------------------------------------------- affichage

    def allumer(self, midi: int, classe: str) -> None:
        el = self.elements.get(midi)
        if el:
            el.classList.add(classe)

    def eteindre(self, midi: int, classe: str) -> None:
        el = self.elements.get(midi)
        if el:
            el.classList.remove(classe)

    def etiqueter(self, lettres: dict[int, str]) -> None:
        """Affiche les lettres du clavier de l'ordinateur sur les touches concernées."""
        for midi, el in self.elements.items():
            texte = lettres.get(midi, "")
            el.firstChild.textContent = texte
            el.classList.toggle("reliee", texte != "")

    # ----------------------------------------------------------------------- saisie

    def _note_sous(self, evenement):
        cadre = self.racine.getBoundingClientRect()
        x = (evenement.clientX - cadre.left + self.decalage) / self.largeur
        y = min(1.0, max(0.0, (evenement.clientY - cadre.top) / cadre.height))
        midi = touche_sous(x, y)
        if midi is None:
            return None
        # Plus on frappe bas sur la touche, plus la note est forte.
        hauteur = HAUTEUR_NOIRE if PAR_MIDI[midi].noire else 1.0
        return midi, 0.45 + 0.5 * min(1.0, y / hauteur)

    def _appui(self, e) -> None:
        if e.pointerType == "mouse" and e.button != 0:
            return
        cible = self._note_sous(e)
        if not cible:
            return
        e.preventDefault()
        self.racine.setPointerCapture(e.pointerId)
        self.pointeurs[e.pointerId] = cible[0]
        self.sur_attaque(*cible)

    def _glisse(self, e) -> None:
        actuelle = self.pointeurs.get(e.pointerId)
        if actuelle is None:
            return
        cible = self._note_sous(e)
        if not cible or cible[0] == actuelle:
            return
        # glissando : on passe d'une touche à l'autre sans lever le doigt
        self.sur_relache(actuelle)
        self.pointeurs[e.pointerId] = cible[0]
        self.sur_attaque(*cible)

    def _leve(self, e) -> None:
        midi = self.pointeurs.pop(e.pointerId, None)
        if midi is not None:
            self.sur_relache(midi)


# ====================================================================== application


class Application:
    def __init__(self) -> None:
        self.moteur = Moteur()
        self.lecteur = Lecteur(self.moteur, self.maj_cartes)
        self.piano = Piano(document.querySelector("#clavier"), self.sur_pointeur, self.relacher, self.dessiner_apercu)
        self.base = BASE_DEFAUT
        self.clavier = "azerty"
        self.catalogue: list[dict] = []
        self.morceaux = {}  # id -> morceau préparé
        self.courant: dict | None = None
        self.demande = 0
        self.appuis: dict[int, int] = {}  # midi -> nombre de sources qui tiennent la touche
        self.codes: dict[str, int] = {}  # touche physique -> midi joué
        self.allumees: dict[int, str] = {}  # midi -> classe posée par la lecture automatique
        self.suivi_suspendu = 0.0
        self.pret = asyncio.Event()
        self.apercu = document.querySelector("#apercu")
        self.apercu_dessine = 0
        self.statut = document.querySelector("#statut")
        self.statut_texte = document.querySelector("#statut-texte")
        # sortie audio : voir verifier_sortie()
        self.son_voulu = False
        self.son_bloque = False
        self.en_attente: dict | None = None
        self.reprendre_apres = False
        self.verification = None
        self.relancer = False
        self.derniere_reconnexion = 0.0

        ecouter(document, "keydown", self.touche_enfoncee)
        ecouter(document, "keyup", self.touche_relachee)
        ecouter(window, "blur", lambda _e: self.tout_relacher())
        ecouter(document, "click", self.rendre_le_clavier)
        ecouter(document.querySelector("#disposition"), "click", lambda _e: self.changer_disposition())
        ecouter(self.apercu, "pointerdown", self.apercu_appui)
        ecouter(self.apercu, "pointermove", self.apercu_glisse)

        # barre de lecture
        self.barre = document.querySelector("#lecture-barre")
        self.tempo = document.querySelector("#tempo")
        self.volume = document.querySelector("#volume")
        self.glisse = False  # l'utilisateur déplace le curseur : l'affichage ne le contrarie pas
        ecouter(document.querySelector("#lecture-bouton"), "click", lambda _e: self.basculer_lecture())
        ecouter(self.barre, "input", self.barre_glisse)
        ecouter(self.barre, "change", self.barre_lachee)
        ecouter(self.tempo, "input", lambda _e: self.appliquer_tempo(int(self.tempo.value)))
        ecouter(self.tempo, "change", lambda _e: self.tempo_lache())
        ecouter(self.volume, "input", lambda _e: self.volume_change())
        for glissiere in document.querySelectorAll(".glissiere"):
            # après un réglage à la souris, les flèches reviennent au changement d'octave
            ecouter(glissiere, "pointerup", lambda e: e.currentTarget.blur())
        self._image = proxy(lambda *_: self.image())
        # Le clavier a été dessiné avant que l'aperçu n'existe : on l'affiche maintenant.
        self.dessiner_apercu()

    # ---------------------------------------------------------------- jeu en direct

    def attaquer(self, midi: int, velocite: float = 0.75) -> None:
        self.reveiller_le_son()
        self.moteur.attaquer(midi, velocite)
        self.appuis[midi] = self.appuis.get(midi, 0) + 1
        self.piano.allumer(midi, "active")

    def relacher(self, midi: int) -> None:
        reste = self.appuis.get(midi, 0) - 1
        if reste > 0:
            self.appuis[midi] = reste
            return
        self.appuis.pop(midi, None)
        self.moteur.relacher(midi)
        self.piano.eteindre(midi, "active")

    def sur_pointeur(self, midi: int, velocite: float) -> None:
        self.suivi_suspendu = horloge() + 4
        self.attaquer(midi, velocite)

    def tout_relacher(self) -> None:
        for midi in list(self.appuis):
            self.appuis[midi] = 1
            self.relacher(midi)
        self.codes.clear()
        self.moteur.pedale(False)

    # ----------------------------------------------------- clavier de l'ordinateur

    def touche_enfoncee(self, e) -> None:
        if e.ctrlKey or e.metaKey or e.altKey:
            return
        code = e.code
        sur_glissiere = e.target.tagName == "INPUT"
        if code == "Escape":
            self.lecteur.arreter()
            return
        if code in ("Enter", "NumpadEnter"):
            if sur_glissiere or e.target.matches("button, a"):
                return
            e.preventDefault()
            if not e.repeat:
                self.basculer_lecture()
            return
        if code == "Space" and not e.target.matches("button, a"):
            e.preventDefault()
            if not e.repeat:
                self.moteur.pedale(True)
            return
        if code in ("ArrowLeft", "ArrowRight"):
            if sur_glissiere:
                return  # les flèches règlent la glissière qui a le focus
            e.preventDefault()
            if not e.repeat:
                self.changer_octave(-1 if code == "ArrowLeft" else 1)
            return
        midi = note_du_code(code, self.base)
        if midi is None:
            return
        e.preventDefault()  # évite la recherche rapide de Firefox sur « / » et « ' »
        if e.repeat or code in self.codes:
            return
        self.codes[code] = midi
        self.attaquer(midi, 0.72)

    def touche_relachee(self, e) -> None:
        if e.code == "Space":
            self.moteur.pedale(False)
            return
        midi = self.codes.pop(e.code, None)
        if midi is not None:
            self.relacher(midi)

    @staticmethod
    def rendre_le_clavier(e) -> None:
        """Après un clic à la souris, Espace doit actionner la pédale, pas le bouton cliqué."""
        if e.detail > 0:
            bouton = e.target.closest("button")
            if bouton:
                bouton.blur()

    def changer_octave(self, sens: int) -> None:
        base = decaler_base(self.base, sens)
        if base == self.base:
            return
        self.base = base
        self.maj_etiquettes()
        self.piano.centrer_sur(base + 12)

    def changer_disposition(self) -> None:
        self.clavier = "qwerty" if self.clavier == "azerty" else "azerty"
        ecrire_preference("disposition", self.clavier)
        self.maj_etiquettes()

    def maj_etiquettes(self) -> None:
        self.piano.etiqueter(etiquettes(self.base, self.clavier))
        document.querySelector("#disposition").textContent = self.clavier.upper()
        for identifiant, code in (("#aide-bas", "KeyZ"), ("#aide-bas-fin", "Slash"),
                                  ("#aide-haut", "KeyQ"), ("#aide-haut-fin", "BracketRight")):
            document.querySelector(identifiant).textContent = caractere(code, self.clavier)

    async def detecter_disposition(self) -> str:
        """AZERTY ou QWERTY : d'après le clavier si le navigateur le permet, sinon d'après la langue."""
        try:
            clavier = getattr(navigator, "keyboard", None)
            if clavier:
                carte = await clavier.getLayoutMap()
                q = carte.get("KeyQ")
                if q:
                    return "azerty" if str(q).lower() == "a" else "qwerty"
        except Exception:
            pass
        return "azerty" if str(navigator.language or "fr").lower().startswith("fr") else "qwerty"

    # --------------------------------------------------------------- clavier MIDI

    async def brancher_midi(self) -> None:
        """Un clavier maître déjà autorisé par le navigateur est reconnu automatiquement."""
        try:
            permission = await navigator.permissions.query(to_js({"name": "midi"}, dict_converter=Object.fromEntries))
            if permission.state != "granted":
                return
            acces = await navigator.requestMIDIAccess()
            message = proxy(self.message_midi)
            for entree in acces.inputs.values():
                entree.onmidimessage = message
        except Exception:
            pass

    def message_midi(self, e) -> None:
        donnees = e.data.to_py()
        statut, numero = donnees[0], donnees[1]
        valeur = donnees[2] if len(donnees) > 2 else 0
        commande = statut & 0xF0
        if commande == 0x90 and valeur > 0:
            self.attaquer(numero, valeur / 127)
        elif commande in (0x80, 0x90):
            self.relacher(numero)
        elif commande == 0xB0 and numero == 64:
            self.moteur.pedale(valeur >= 64)

    # ---------------------------------------------------------------------- morceaux

    def creer_carte(self, fiche: dict, rang: int):
        li = document.createElement("li")
        bouton = document.createElement("button")
        bouton.type = "button"
        bouton.className = "carte"
        bouton.dataset.id = fiche["id"]

        def ajouter(classe: str, texte: str):
            span = document.createElement("span")
            span.className = classe
            span.textContent = texte
            bouton.append(span)
            return span

        ajouter("carte-num", f"{rang + 1:02d}")
        icone = ajouter("carte-icone", "")
        icone.setAttribute("aria-hidden", "true")
        for _ in range(3):
            icone.append(document.createElement("i"))
        ajouter("carte-titre", fiche["titre"])
        ajouter("carte-auteur", fiche["compositeur"])
        ajouter("carte-meta", formater_duree(fiche["duree"]))
        ajouter("carte-barre", "").append(document.createElement("i"))
        bouton.setAttribute("aria-label", f"{fiche['titre']}, {fiche['compositeur']}, {formater_duree(fiche['duree'])}")
        bouton.setAttribute("aria-pressed", "false")
        bouton.title = f"{fiche['oeuvre']} ({fiche['annee']})"

        def clic(_e, fiche=fiche):
            self.reveiller_le_son()  # dans le geste de l'utilisateur
            asyncio.ensure_future(self.choisir(fiche))

        ecouter(bouton, "click", clic)
        li.append(bouton)
        return li

    async def charger_morceau(self, fiche: dict):
        if fiche["id"] not in self.morceaux:
            reponse = await fetch(fiche["fichier"])
            if not reponse.ok:
                raise OSError(f"Fichier introuvable : {fiche['fichier']}")
            octets = (await reponse.arrayBuffer()).to_py().tobytes()
            # mido lit le fichier ; les tics sont convertis en secondes selon sa carte de tempo
            morceau = lecture_midi.charger(octets, fiche["titre"])
            fiche["analyse"] = resumer(morceau)
            self.morceaux[fiche["id"]] = morceau
        return self.morceaux[fiche["id"]]

    async def choisir(self, fiche: dict) -> None:
        """Un clic sur un morceau le joue du début à la fin ; un second clic l'arrête."""
        self.demande += 1
        demande = self.demande
        if self.courant and self.courant["id"] == fiche["id"] and self.lecteur.etat == "lecture":
            self.lecteur.arreter()
            return
        if self.courant and self.courant["id"] == fiche["id"] and self.lecteur.etat == "pause":
            self.lecteur.reprendre()
            return
        carte = document.querySelector(f'.carte[data-id="{fiche["id"]}"]')
        carte.classList.add("charge")
        try:
            morceau = await self.charger_morceau(fiche)
            await self.pret.wait()
            if demande != self.demande:
                return
            # Sans sortie audio utilisable, on ne reste pas figé : le morceau est mis en attente
            # et démarrera tout seul dès que le son sera revenu.
            if not await self.essayer_le_son(2.0):
                if demande == self.demande:
                    self.son_indisponible(fiche)
                return
            if demande != self.demande:
                return
            self.en_attente = None
            self.courant = fiche
            self.lecteur.charger(morceau)
            self.lecteur.jouer(0)
        except Exception as erreur:  # noqa: BLE001 - on l'affiche à l'utilisateur
            window.console.error(str(erreur))
            self.statut_texte.textContent = "Impossible de charger ce morceau."
        finally:
            carte.classList.remove("charge")

    def maj_cartes(self, _etat: str = "") -> None:
        etat = self.lecteur.etat
        for carte in document.querySelectorAll(".carte"):
            actuelle = etat != "arret" and self.courant is not None and carte.dataset.id == self.courant["id"]
            carte.classList.toggle("en-cours", actuelle)
            carte.classList.toggle("pause", actuelle and etat == "pause")
            carte.setAttribute("aria-pressed", "true" if actuelle else "false")
            if not actuelle:
                carte.style.removeProperty("--ratio")
        document.body.classList.toggle("sans-son", self.son_bloque)
        if self.son_bloque:
            self.statut_texte.textContent = MESSAGE_SANS_SON
        elif etat != "arret":
            analyse = self.courant.get("analyse", {})
            tonalite = f" · {analyse['tonalite']}" if analyse.get("tonalite") else ""
            titre = f"{self.courant['titre']} — {self.courant['compositeur']}{tonalite}"
            self.statut_texte.textContent = titre if etat == "lecture" else f"En pause · {titre}"
        elif self.moteur.pret:
            self.statut_texte.textContent = "Piano prêt"
        self.maj_barre()

    # ------------------------------------------------------------- barre de lecture

    # ---------------------------------------------------------------- sortie audio
    # Le piano s'adapte tout seul aux changements de sortie audio (casque branché, sortie changée
    # dans le système…). Si le son ne peut pas démarrer, il le dit, garde le morceau demandé et
    # réessaie ; un contexte audio resté bloqué est remplacé par un neuf.

    def reveiller_le_son(self) -> None:
        """À chaque geste : tente de démarrer le son, et relance la vérification s'il manque."""
        self.son_voulu = True
        self.moteur.demarrer()
        if self.son_bloque:
            self.verifier_sortie()

    async def essayer_le_son(self, attente: float = 1.5) -> bool:
        """Vrai si le son tourne, après au plus `attente` secondes."""
        if self.moteur.actif:
            return True
        self.moteur.demarrer()
        fin = horloge() + attente
        while horloge() < fin:
            await asyncio.sleep(0.1)
            if self.moteur.actif:
                return True
        return self.moteur.actif

    def son_indisponible(self, fiche: dict | None = None) -> None:
        if fiche:
            self.en_attente = fiche
        if not self.son_bloque:
            self.son_bloque = True
            self.maj_cartes()

    def son_retabli(self) -> None:
        if not self.son_bloque:
            return
        self.son_bloque = False
        self.maj_cartes()
        if self.reprendre_apres:
            self.reprendre_apres = False
            self.lecteur.reprendre()
        if self.en_attente:
            fiche, self.en_attente = self.en_attente, None
            asyncio.ensure_future(self.choisir(fiche))

    def verifier_sortie(self, automatique: bool = False) -> None:
        """Vérifie que le son tourne ; sinon réessaie, puis reconstruit le contexte audio.

        Une vérification automatique (sans geste ni changement de sortie) reconstruit moins souvent.
        """
        if self.verification is not None and not self.verification.done():
            # un branchement ou un geste pendant une vérification n'est pas perdu : une
            # vérification complète suivra
            if not automatique:
                self.relancer = True
            return
        self.verification = asyncio.ensure_future(self._verifier_sortie(automatique))

    async def _verifier_sortie(self, automatique: bool) -> None:
        try:
            await self._verifier_une_fois(automatique)
        finally:
            if self.relancer:
                self.relancer = False
                if not self.moteur.actif:
                    self.verification = asyncio.ensure_future(self._verifier_sortie(False))

    async def _verifier_une_fois(self, automatique: bool) -> None:
        if await self.essayer_le_son():
            self.son_retabli()
            return
        if not self.moteur.pret or horloge() - self.derniere_reconnexion < (15 if automatique else 3):
            self.son_indisponible()
            return
        self.derniere_reconnexion = horloge()
        # L'horloge va changer : la lecture est mise en pause pour reprendre au même endroit.
        if self.lecteur.etat == "lecture":
            self.lecteur.pause()
            self.reprendre_apres = True
        self.moteur.reconnecter()
        if await self.essayer_le_son():
            self.son_retabli()
        else:
            self.son_indisponible()

    def surveiller_sortie(self) -> None:
        peripheriques = getattr(navigator, "mediaDevices", None)
        if peripheriques is not None and hasattr(peripheriques, "addEventListener"):
            ecouter(peripheriques, "devicechange", lambda _e: self.son_voulu and self.verifier_sortie())

        def filet(*_):
            # la sortie peut disparaître ou revenir sans événement
            if self.son_voulu and self.moteur.pret and not self.moteur.actif:
                self.verifier_sortie(automatique=True)

        window.setInterval(proxy(filet), 2000)

    def basculer_lecture(self) -> None:
        self.reveiller_le_son()
        if self.lecteur.etat == "lecture":
            self.lecteur.pause()
        elif self.lecteur.etat == "pause":
            self.lecteur.reprendre()
        elif self.courant:
            asyncio.ensure_future(self.choisir(self.courant))

    def maj_barre(self) -> None:
        etat = self.lecteur.etat
        actif = etat != "arret"
        document.querySelector("#lecture").dataset.etat = etat
        bouton = document.querySelector("#lecture-bouton")
        bouton.disabled = self.courant is None
        bouton.setAttribute("aria-label", "Pause" if etat == "lecture" else "Lecture")
        titre = f"{self.courant['titre']} — {self.courant['compositeur']}" if self.courant else "Choisissez un morceau"
        document.querySelector("#lecture-titre").textContent = titre
        self.barre.disabled = not actif
        duree = (self.lecteur.duree or self.courant["duree"]) if self.courant else 0
        document.querySelector("#lecture-duree").textContent = formater_duree(duree)
        if not actif:
            self.barre.value = 0
            document.querySelector("#lecture-position").textContent = formater_duree(0)

    def barre_glisse(self, _e) -> None:
        self.glisse = True
        position = int(self.barre.value) / 1000 * self.lecteur.duree
        document.querySelector("#lecture-position").textContent = formater_duree(position)

    def barre_lachee(self, _e) -> None:
        self.glisse = False
        self.lecteur.chercher(int(self.barre.value) / 1000 * self.lecteur.duree)

    def appliquer_tempo(self, valeur: int) -> None:
        self.tempo.value = valeur
        document.querySelector("#tempo-valeur").textContent = f"{valeur}\u00a0%"

    def tempo_lache(self) -> None:
        # appliqué au relâchement : la lecture n'est relancée qu'une fois
        self.lecteur.changer_vitesse(int(self.tempo.value) / 100)
        ecrire_preference("tempo", int(self.tempo.value))

    def appliquer_volume(self, valeur: int) -> None:
        self.volume.value = valeur
        document.querySelector("#volume-valeur").textContent = f"{valeur}\u00a0%"
        self.moteur.volume(valeur / 100 * 1.2)

    def volume_change(self) -> None:
        self.appliquer_volume(int(self.volume.value))
        ecrire_preference("volume", int(self.volume.value))

    # ------------------------------------------------------------------------ aperçu
    # Sur un petit écran, le clavier dépasse : une réglette permet de s'y déplacer.

    def dessiner_apercu(self) -> None:
        apercu = getattr(self, "apercu", None)
        if apercu is None:
            return
        apercu.hidden = not self.piano.deborde
        if apercu.hidden:
            return
        canvas = document.querySelector("#apercu-canvas")
        densite = min(2.0, window.devicePixelRatio or 1)
        largeur, hauteur = apercu.clientWidth, apercu.clientHeight
        if self.apercu_dessine != round(largeur * densite):
            self.apercu_dessine = round(largeur * densite)
            canvas.width = self.apercu_dessine
            canvas.height = round(hauteur * densite)
            ctx = canvas.getContext("2d")
            ctx.setTransform(densite, 0, 0, densite, 0, 0)
            unite = largeur / NB_BLANCHES
            ctx.fillStyle = "#e9e6dc"
            ctx.fillRect(0, 0, largeur, hauteur)
            ctx.fillStyle = "rgba(0,0,0,.25)"
            for i in range(1, NB_BLANCHES):
                ctx.fillRect(round(i * unite), 0, 1, hauteur)
            ctx.fillStyle = "#15161b"
            for t in TOUCHES:
                if t.noire:
                    ctx.fillRect(t.x * unite, 0, t.largeur * unite, hauteur * 0.6)
        fenetre = document.querySelector("#apercu-fenetre")
        fenetre.style.left = f"{self.piano.decalage / self.piano.largeur_totale * 100}%"
        fenetre.style.width = f"{self.piano.largeur_visible / self.piano.largeur_totale * 100}%"

    def _deplacer_apercu(self, e) -> None:
        cadre = self.apercu.getBoundingClientRect()
        ratio = (e.clientX - cadre.left) / cadre.width
        self.piano.definir_decalage(ratio * self.piano.largeur_totale - self.piano.largeur_visible / 2)
        self.suivi_suspendu = horloge() + 4

    def apercu_appui(self, e) -> None:
        self.apercu.setPointerCapture(e.pointerId)
        self._deplacer_apercu(e)

    def apercu_glisse(self, e) -> None:
        if self.apercu.hasPointerCapture(e.pointerId):
            self._deplacer_apercu(e)

    # ------------------------------------------------------------ boucle d'affichage

    def suivre(self, morceau, position: float) -> None:
        """Sur petit écran, le clavier suit les mains pendant la lecture."""
        piano = self.piano
        if not piano.deborde or horloge() < self.suivi_suspendu:
            return
        somme = nombre = 0
        notes = morceau.notes
        for i in range(morceau.indice_a(position - 0.3), len(notes)):
            if notes[i].temps >= position + 1.5:
                break
            t = PAR_MIDI[notes[i].midi]
            somme += t.x + t.largeur / 2
            nombre += 1
        if not nombre:
            return
        cible = somme / nombre * piano.largeur - piano.largeur_visible / 2
        piano.definir_decalage(piano.decalage + (cible - piano.decalage) * 0.05)

    def image(self) -> None:
        lecteur = self.lecteur
        morceau = lecteur.morceau if lecteur.etat == "lecture" else None
        position = lecteur.position

        # touches enfoncées par la lecture automatique
        voulues = {}
        if morceau:
            for note in morceau.touches_enfoncees(position):
                voulues[note.midi] = "main-d" if note.main == 0 else "main-g"
        for midi, classe in list(self.allumees.items()):
            if voulues.get(midi) != classe:
                self.piano.eteindre(midi, classe)
                del self.allumees[midi]
        for midi, classe in voulues.items():
            if midi not in self.allumees:
                self.piano.allumer(midi, classe)
                self.allumees[midi] = classe

        if morceau:
            carte = document.querySelector(".carte.en-cours")
            if carte:
                carte.style.setProperty("--ratio", str(position / lecteur.duree))
            self.suivre(morceau, position)
        if lecteur.etat != "arret" and not self.glisse and lecteur.duree:
            self.barre.value = round(position / lecteur.duree * 1000)
            texte = formater_duree(position)
            affichage = document.querySelector("#lecture-position")
            if affichage.textContent != texte:
                affichage.textContent = texte
        window.requestAnimationFrame(self._image)

    # --------------------------------------------------------------------- démarrage

    async def ouvrir_cache(self):
        """Cache du navigateur où sont gardés les sons déjà synthétisés.

        Son nom contient l'empreinte de synthese.py : si la synthèse change, les anciens sons
        sont oubliés. Le cache n'existe que sur une page sécurisée (https ou localhost).
        """
        try:
            caches = getattr(window, "caches", None)
            if caches is None:
                return None
            with open(synthese.__file__, "rb") as source:
                empreinte = hashlib.sha256(source.read()).hexdigest()[:16]
            nom = f"{CACHE_PREFIXE}{empreinte}"
            for ancien in (await caches.keys()).to_py():
                if ancien.startswith(CACHE_PREFIXE) and ancien != nom:
                    await caches.delete(ancien)
            return await caches.open(nom)
        except Exception:
            return None

    @staticmethod
    async def lire_son(cache, midi: int):
        """Son d'une note gardé en cache (entiers 16 bits), ou None."""
        if cache is None:
            return None
        try:
            reponse = await cache.match(f"sons/{midi}.pcm")
            if not reponse:
                return None
            octets = (await reponse.arrayBuffer()).to_py()
            return np.frombuffer(octets, dtype="<i2").astype(np.float32) / 32767
        except Exception:
            return None

    @staticmethod
    async def garder_son(cache, midi: int, son) -> None:
        if cache is None:
            return
        try:
            entiers = (np.clip(son, -1, 1) * 32767).astype("<i2").tobytes()
            await cache.put(f"sons/{midi}.pcm", Response.new(Uint8Array.new(to_js(entiers))))
        except Exception:
            pass  # cache plein ou refusé : les sons seront recalculés à la prochaine visite

    async def synthetiser_le_piano(self) -> None:
        """NumPy calcule le son des 88 notes, en partant du milieu du clavier.

        Les sons sont ensuite gardés dans le cache du navigateur : aux visites suivantes, le
        piano est prêt presque aussitôt, au lieu d'être recalculé.
        """
        # Le ConvolverNode exige la fréquence d'échantillonnage du contexte audio (44,1 ou 48 kHz) :
        # la réponse de salle est donc calculée à cette fréquence, pas à celle des notes.
        fe_salle = int(self.moteur.ctx.sampleRate)
        self.moteur.installer_salle(synthese.reponse_de_salle(fe_salle), fe_salle)
        cache = await self.ouvrir_cache()
        ordre = sorted(range(NOTE_MIN, NOTE_MAX + 1), key=lambda midi: abs(midi - 62))
        for rang, midi in enumerate(ordre, start=1):
            son = await self.lire_son(cache, midi)
            if son is None:
                son = synthese.synthetiser(midi)
                await self.garder_son(cache, midi, son)
                self.statut_texte.textContent = f"Python synthétise le piano… {rang}/{len(ordre)}"
            else:
                self.statut_texte.textContent = f"Python prépare le piano… {rang}/{len(ordre)}"
            self.moteur.ajouter_note(midi, son, synthese.FE)
            self.piano.eteindre(midi, "attente")
            self.statut.style.setProperty("--ratio", str(rang / len(ordre)))
            await asyncio.sleep(0)  # rend la main au navigateur après chaque note : la page reste fluide
        self.moteur.pret = True
        document.body.classList.add("pret")
        self.statut_texte.textContent = "Piano prêt"
        self.pret.set()

    async def demarrer(self) -> None:
        self.clavier = lire_preference("disposition", None) or await self.detecter_disposition()
        self.maj_etiquettes()
        self.appliquer_volume(int(lire_preference("volume", 75)))
        self.appliquer_tempo(int(lire_preference("tempo", 100)))
        self.lecteur.changer_vitesse(int(self.tempo.value) / 100)
        window.requestAnimationFrame(self._image)
        reponse = await fetch("midi/catalogue.json")
        self.catalogue = json.loads(await reponse.text())
        cartes = document.querySelector("#cartes")
        for rang, fiche in enumerate(self.catalogue):
            cartes.append(self.creer_carte(fiche, rang))
        asyncio.ensure_future(self.brancher_midi())
        self.surveiller_sortie()
        await self.synthetiser_le_piano()

    def crochet_de_test(self):
        """Accès pour les tests automatisés et la console du navigateur (window.__piano)."""
        lecteur, moteur = self.lecteur, self.moteur
        fonctions = {
            "pret": lambda: moteur.pret,
            "niveau": moteur.niveau,
            "notesJouees": lambda: moteur.notes_jouees,
            "etatLecteur": lambda: lecteur.etat,
            "pause": lecteur.pause,
            "sonBloque": lambda: self.son_bloque,
            "reconnexions": lambda: moteur.reconnexions,
            "reprendre": lecteur.reprendre,
            "position": lambda: lecteur.position,
            "duree": lambda: lecteur.duree,
            "vitesse": lecteur.changer_vitesse,
            "chercher": lecteur.chercher,
            "notes": lambda: to_js([[n.temps, n.midi, n.fin, n.velocite, n.main]
                                    for n in (lecteur.morceau.notes if lecteur.morceau else [])]),
            "attaquer": self.attaquer,
            "relacher": self.relacher,
        }
        return Object.fromEntries(to_js([[nom, proxy(f)] for nom, f in fonctions.items()]))


async def demarrer() -> None:
    application = Application()
    window.__piano = application.crochet_de_test()
    await application.demarrer()

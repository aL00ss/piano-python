"""L'application dans le navigateur, entièrement écrite en Python.

Pyodide exécute ce fichier dans la page. Le module `js` donne accès au DOM et à la Web Audio
API : c'est Python qui dessine le clavier, écoute le clavier et les pointeurs, synthétise le
son des 88 notes avec NumPy, lit les fichiers MIDI avec mido et programme chaque note.
"""

from __future__ import annotations

import asyncio
import json

from js import Float32Array, Object, document, fetch, navigator, performance, window
from pyodide.ffi import create_proxy, to_js

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

_proxys = []  # les fonctions Python confiées au navigateur doivent rester en vie


def proxy(fonction):
    p = create_proxy(fonction)
    _proxys.append(p)
    return p


def ecouter(cible, evenement: str, fonction) -> None:
    cible.addEventListener(evenement, proxy(fonction))


def horloge() -> float:
    return performance.now() / 1000


def gain_de_velocite(velocite: float) -> float:
    """Vélocité (0 à 1) -> gain : un pianissimo reste audible, un fortissimo ne sature pas."""
    v = min(1.0, max(0.0, velocite))
    return 0.08 + 0.92 * v**1.6


# ============================================================================== son


class Moteur:
    """Joue les notes synthétisées par NumPy à travers la Web Audio API."""

    def __init__(self) -> None:
        constructeur = getattr(window, "AudioContext", None) or window.webkitAudioContext
        self.ctx = constructeur.new(to_js({"latencyHint": "interactive"}, dict_converter=Object.fromEntries))
        c = self.ctx
        self.sortie = c.createGain()
        self.sortie.gain.value = 0.9
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

        self.tampons = {}  # midi -> AudioBuffer
        self.voix = {}  # midi -> voix du jeu en direct
        self.enfoncees: set[int] = set()
        self.tenues: set[int] = set()
        self.pedale_enfoncee = False
        self.notes_jouees = 0
        self.pret = False
        self._onde = Float32Array.new(1024)

    def _tampon(self, canaux, fe: int):
        tampon = self.ctx.createBuffer(len(canaux), len(canaux[0]), fe)
        for indice, canal in enumerate(canaux):
            tampon.copyToChannel(to_js(canal), indice)  # tableau NumPy float32 -> Float32Array
        return tampon

    def installer_salle(self, reponse, fe: int) -> None:
        self.reverb.buffer = self._tampon(list(reponse), fe)

    def ajouter_note(self, midi: int, son, fe: int) -> None:
        self.tampons[midi] = self._tampon([son], fe)

    def demarrer(self) -> None:
        """À appeler depuis un geste de l'utilisateur : les navigateurs bloquent le son avant."""
        if self.ctx.state != "running":
            self.ctx.resume()

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
        window.setTimeout(create_proxy(lambda: gain.disconnect()), 400)


class Lecteur:
    """Programme les notes un peu à l'avance sur l'horloge audio, précise à l'échantillon près."""

    def __init__(self, moteur: Moteur, sur_etat) -> None:
        self.moteur = moteur
        self.sur_etat = sur_etat
        self.morceau = None
        self.session = None
        self.minuterie = None
        self.etat = "arret"  # "arret" | "lecture"
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

    def chercher(self, position: float) -> None:
        if self.etat == "lecture":
            self.jouer(position)

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
            window.setTimeout(create_proxy(lambda: session.arreter()), 1500)
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

        ecouter(document, "keydown", self.touche_enfoncee)
        ecouter(document, "keyup", self.touche_relachee)
        ecouter(window, "blur", lambda _e: self.tout_relacher())
        ecouter(document, "click", self.rendre_le_clavier)
        ecouter(document.querySelector("#disposition"), "click", lambda _e: self.changer_disposition())
        ecouter(self.apercu, "pointerdown", self.apercu_appui)
        ecouter(self.apercu, "pointermove", self.apercu_glisse)
        self._image = proxy(lambda *_: self.image())

    # ---------------------------------------------------------------- jeu en direct

    def attaquer(self, midi: int, velocite: float = 0.75) -> None:
        self.moteur.demarrer()
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
        if code == "Escape":
            self.lecteur.arreter()
            return
        if code == "Space" and not e.target.matches("button, a"):
            e.preventDefault()
            if not e.repeat:
                self.moteur.pedale(True)
            return
        if code in ("ArrowLeft", "ArrowRight"):
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
            self.moteur.demarrer()  # dans le geste de l'utilisateur
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
        carte = document.querySelector(f'.carte[data-id="{fiche["id"]}"]')
        carte.classList.add("charge")
        try:
            morceau = await self.charger_morceau(fiche)
            await self.pret.wait()
            if demande != self.demande:
                return
            self.courant = fiche
            self.lecteur.charger(morceau)
            self.lecteur.jouer(0)
        except Exception as erreur:  # noqa: BLE001 - on l'affiche à l'utilisateur
            window.console.error(str(erreur))
            self.statut_texte.textContent = "Impossible de charger ce morceau."
        finally:
            carte.classList.remove("charge")

    def maj_cartes(self, _etat: str = "") -> None:
        lecture = self.lecteur.etat == "lecture"
        for carte in document.querySelectorAll(".carte"):
            actuelle = lecture and self.courant is not None and carte.dataset.id == self.courant["id"]
            carte.classList.toggle("en-cours", actuelle)
            carte.setAttribute("aria-pressed", "true" if actuelle else "false")
            if not actuelle:
                carte.style.removeProperty("--ratio")
        if lecture:
            analyse = self.courant.get("analyse", {})
            tonalite = f" · {analyse['tonalite']}" if analyse.get("tonalite") else ""
            self.statut_texte.textContent = f"{self.courant['titre']} — {self.courant['compositeur']}{tonalite}"
        elif self.moteur.pret:
            self.statut_texte.textContent = "Piano prêt"

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
        window.requestAnimationFrame(self._image)

    # --------------------------------------------------------------------- démarrage

    async def synthetiser_le_piano(self) -> None:
        """NumPy calcule le son des 88 notes, en partant du milieu du clavier."""
        self.moteur.installer_salle(synthese.reponse_de_salle(), synthese.FE)
        ordre = sorted(range(NOTE_MIN, NOTE_MAX + 1), key=lambda midi: abs(midi - 62))
        for rang, midi in enumerate(ordre, start=1):
            self.moteur.ajouter_note(midi, synthese.synthetiser(midi), synthese.FE)
            self.piano.eteindre(midi, "attente")
            if rang % 4 == 0 or rang == len(ordre):
                self.statut_texte.textContent = f"Python synthétise le piano… {rang}/{len(ordre)}"
                self.statut.style.setProperty("--ratio", str(rang / len(ordre)))
                await asyncio.sleep(0)  # rend la main au navigateur
        self.moteur.pret = True
        document.body.classList.add("pret")
        self.statut_texte.textContent = "Piano prêt"
        self.pret.set()

    async def demarrer(self) -> None:
        self.clavier = await self.detecter_disposition()
        self.maj_etiquettes()
        window.requestAnimationFrame(self._image)
        reponse = await fetch("midi/catalogue.json")
        self.catalogue = json.loads(await reponse.text())
        cartes = document.querySelector("#cartes")
        for rang, fiche in enumerate(self.catalogue):
            cartes.append(self.creer_carte(fiche, rang))
        asyncio.ensure_future(self.brancher_midi())
        await self.synthetiser_le_piano()

    def crochet_de_test(self):
        """Accès pour les tests automatisés et la console du navigateur (window.__piano)."""
        lecteur, moteur = self.lecteur, self.moteur
        fonctions = {
            "pret": lambda: moteur.pret,
            "niveau": moteur.niveau,
            "notesJouees": lambda: moteur.notes_jouees,
            "etatLecteur": lambda: lecteur.etat,
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

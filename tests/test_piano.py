"""Tests unitaires du piano : théorie, lecture MIDI, synthèse et analyse.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "python"))

import numpy as np  # noqa: E402

from piano import analyse, morceau, recherche, synthese, theorie  # noqa: E402
from piano.morceau import Note  # noqa: E402

CATALOGUE = json.loads((RACINE / "midi" / "catalogue.json").read_text(encoding="utf-8"))


class TestTheorie(unittest.TestCase):
    def test_88_touches_dont_52_blanches(self):
        self.assertEqual(len(theorie.TOUCHES), 88)
        self.assertEqual(sum(not t.noire for t in theorie.TOUCHES), theorie.NB_BLANCHES)
        self.assertEqual(theorie.TOUCHES[0].midi, 21)
        self.assertEqual(theorie.TOUCHES[-1].midi, 108)

    def test_noms_et_frequences(self):
        self.assertEqual(theorie.nom_francais(69), "La3")
        self.assertEqual(theorie.nom_anglais(69), "A4")
        self.assertEqual(theorie.nom_francais(60), "Do3")
        self.assertAlmostEqual(theorie.frequence(69), 440.0)
        self.assertAlmostEqual(theorie.frequence(81), 880.0)

    def test_touche_sous_le_pointeur(self):
        do_central = theorie.PAR_MIDI[60]
        self.assertEqual(theorie.touche_sous(do_central.x + 0.5, 0.9), 60)
        do_diese = theorie.PAR_MIDI[61]
        self.assertEqual(theorie.touche_sous(do_diese.x + do_diese.largeur / 2, 0.3), 61)
        # sous la touche noire, le bas de la touche reste une blanche
        self.assertIn(theorie.touche_sous(do_diese.x + do_diese.largeur / 2, 0.9), (60, 62))
        self.assertIsNone(theorie.touche_sous(-1, 0.5))
        self.assertIsNone(theorie.touche_sous(theorie.NB_BLANCHES, 0.5))

    def test_clavier_de_l_ordinateur(self):
        self.assertEqual(theorie.note_du_code("KeyQ"), 60)  # rangée du haut : do central
        self.assertEqual(theorie.note_du_code("KeyZ"), 48)
        self.assertEqual(theorie.note_du_code("Digit2"), 61)
        self.assertIsNone(theorie.note_du_code("Space"))
        self.assertEqual(theorie.caractere("KeyQ", "azerty"), "A")
        self.assertEqual(theorie.caractere("KeyQ", "qwerty"), "Q")
        self.assertEqual(theorie.decaler_base(theorie.BASE_MAX, 1), theorie.BASE_MAX)
        self.assertEqual(theorie.decaler_base(48, -1), 36)
        lettres = theorie.etiquettes(48, "azerty")
        self.assertEqual(lettres[60], "A")
        self.assertEqual(len(lettres), len(theorie.DEMI_TONS) - 5)  # rangées qui se chevauchent

    def test_formater_duree(self):
        self.assertEqual(theorie.formater_duree(185), "3:05")
        self.assertEqual(theorie.formater_duree(0), "0:00")


class TestMorceau(unittest.TestCase):
    def test_pedale_prolonge_les_notes(self):
        notes = [Note(60, 0.0, 0.5, 0.8, 0), Note(64, 1.0, 0.5, 0.8, 0)]
        m = morceau.preparer(notes, [(0.2, True), (2.0, False)])
        self.assertAlmostEqual(m.notes[0].fin, 2.0)  # relâchée sous la pédale
        self.assertAlmostEqual(m.notes[1].fin, 2.0)
        self.assertTrue(m.pedale_a(1.0))
        self.assertFalse(m.pedale_a(2.5))

    def test_note_rejouee_coupe_la_precedente(self):
        notes = [Note(60, 0.0, 0.4, 0.8, 0), Note(60, 0.6, 0.4, 0.8, 0)]
        m = morceau.preparer(notes, [(0.0, True), (3.0, False)])
        self.assertAlmostEqual(m.notes[0].fin, 0.6)
        self.assertAlmostEqual(m.notes[1].fin, 3.0)

    def test_touches_enfoncees(self):
        notes = [Note(60, 0.0, 1.0, 0.8, 0), Note(48, 0.5, 0.2, 0.8, 1)]
        m = morceau.preparer(notes)
        self.assertEqual([n.midi for n in m.touches_enfoncees(0.6)], [60, 48])
        self.assertEqual([n.midi for n in m.touches_enfoncees(0.9)], [60])
        self.assertEqual(m.indice_a(0.4), 1)

    def test_les_sept_morceaux_correspondent_au_catalogue(self):
        self.assertEqual(len(CATALOGUE), 7)
        for fiche in CATALOGUE:
            with self.subTest(fiche["id"]):
                m = morceau.charger((RACINE / fiche["fichier"]).read_bytes(), fiche["titre"])
                self.assertEqual(len(m.notes), fiche["notes"])
                self.assertAlmostEqual(m.duree, fiche["duree"], delta=0.1)
                hauteurs = [n.midi for n in m.notes]
                self.assertEqual((min(hauteurs), max(hauteurs)), (fiche["grave"], fiche["aigu"]))
                self.assertTrue(all(n.fin > n.temps for n in m.notes))
                self.assertEqual({n.main for n in m.notes}, {0, 1})
                self.assertEqual(m.temps, sorted(m.temps))


class TestSynthese(unittest.TestCase):
    def test_son_borne_et_deterministe(self):
        for midi in (21, 60, 108):
            with self.subTest(midi):
                son = synthese.synthetiser(midi)
                self.assertEqual(son.dtype, np.float32)
                self.assertEqual(len(son), int(synthese.duree_note(midi) * synthese.FE))
                self.assertLessEqual(float(np.abs(son).max()), 1.0)
                self.assertTrue(np.array_equal(son, synthese.synthetiser(midi)))

    def test_hauteur_de_la_fondamentale(self):
        son = synthese.synthetiser(69).astype(float)[: synthese.FE]
        spectre = np.abs(np.fft.rfft(son * np.hanning(len(son))))
        pic = np.argmax(spectre) * synthese.FE / len(son)
        self.assertAlmostEqual(pic, 440.0, delta=3.0)

    def test_niveau_homogene_sur_le_clavier(self):
        niveaux = [np.sqrt(np.mean(synthese.synthetiser(m)[: int(0.3 * synthese.FE)] ** 2)) for m in (24, 60, 96)]
        for niveau in niveaux:
            self.assertAlmostEqual(niveau, synthese.NIVEAU, delta=0.03)

    def test_hors_du_piano(self):
        with self.assertRaises(ValueError):
            synthese.synthetiser(20)

    def test_reponse_de_salle_a_toute_frequence(self):
        for fe in (32000, 44100, 48000):
            reponse = synthese.reponse_de_salle(fe)
            self.assertEqual(reponse.shape, (2, int(2.6 * fe)))
            self.assertTrue(np.isfinite(reponse).all())

    def test_rendu_et_wav(self):
        notes = [SimpleNamespace(midi=60, temps=0.0, fin=0.5, velocite=0.8),
                 SimpleNamespace(midi=64, temps=0.25, fin=0.5, velocite=0.6)]
        son = synthese.rendre(notes)
        self.assertGreater(len(son), synthese.FE)
        self.assertLessEqual(float(np.abs(son).max()), 0.9)
        wav = synthese.ecrire_wav(son)
        self.assertEqual(wav[:4], b"RIFF")


class TestAnalyse(unittest.TestCase):
    def test_tonalites_des_morceaux(self):
        attendues = {"lettre-a-elise": "la mineur", "prelude-en-do": "do majeur"}
        for fiche in CATALOGUE:
            if fiche["id"] in attendues:
                m = morceau.charger((RACINE / fiche["fichier"]).read_bytes())
                self.assertEqual(analyse.tonalite(m.notes), attendues[fiche["id"]])

    def test_resume(self):
        m = morceau.preparer([Note(60, 0.0, 1.0, 0.8, 0), Note(64, 0.0, 1.0, 0.8, 0), Note(48, 0.5, 1.0, 0.8, 1)])
        r = analyse.resumer(m)
        self.assertEqual(r["notes"], 3)
        self.assertEqual(r["polyphonie"], 3)
        self.assertEqual((r["main_droite"], r["main_gauche"]), (2, 1))
        self.assertEqual(analyse.resumer(morceau.preparer([]))["notes"], 0)



class TestRecherche(unittest.TestCase):
    def test_noms_lisibles(self):
        self.assertEqual(recherche.nom_lisible("chopin-ballade_1.mid"), "chopin ballade 1")
        self.assertEqual(recherche.nom_lisible("Clair-De-Lune.MIDI"), "Clair De Lune")
        self.assertEqual(recherche.nom_lisible(".mid"), "Sans titre")

    def test_les_noms_sont_echappes(self):
        self.assertEqual(recherche.echapper('<img src=x onerror="alert(1)">'),
                         "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;")

    def test_adresse_et_fiches(self):
        self.assertEqual(recherche.adresse("satie gymnopédie"),
                         "https://bitmidi.com/api/midi/search?q=satie%20gymnop%C3%A9die&page=0")
        reponse = {"result": {"results": [
            {"id": 23776, "name": "chopin-ballade1.mid", "downloadUrl": "/uploads/23776.mid", "plays": 10781},
            {"id": 2, "name": "sans-fichier.mid"},
        ]}}
        self.assertEqual(recherche.fiches(reponse), [{
            "id": "bitmidi-23776", "titre": "chopin ballade1", "compositeur": "BitMidi",
            "oeuvre": "chopin-ballade1.mid", "fichier": "https://bitmidi.com/uploads/23776.mid",
            "duree": 0, "ecoutes": 10781, "enLigne": True}])
        self.assertEqual(recherche.fiches({}), [])
        self.assertEqual(recherche.formater_nombre(10781), "10\u202f781")

    def test_batterie_et_notes_hors_clavier_ignorees(self):
        import io

        import mido
        fichier = mido.MidiFile()
        for canal, hauteurs in ((0, (60, 12, 120)), (9, (36, 38)), (1, (48,))):
            piste = mido.MidiTrack()
            for hauteur in hauteurs:
                piste.append(mido.Message("note_on", channel=canal, note=hauteur, velocity=80, time=0))
                piste.append(mido.Message("note_off", channel=canal, note=hauteur, velocity=0, time=240))
            fichier.tracks.append(piste)
        tampon = io.BytesIO()
        fichier.save(file=tampon)
        m = morceau.charger(tampon.getvalue())
        self.assertEqual(sorted(n.midi for n in m.notes), [48, 60])
        self.assertEqual(sorted(n.main for n in m.notes), [0, 1])


if __name__ == "__main__":
    unittest.main()

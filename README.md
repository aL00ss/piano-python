# Piano Python — un piano en Python

[![Tests](https://github.com/aL00ss/piano-python/actions/workflows/tests.yml/badge.svg)](https://github.com/aL00ss/piano-python/actions/workflows/tests.yml)

Un piano de **88 touches** dans le navigateur, écrit en **Python**. On peut y jouer avec le
clavier de l'ordinateur, la souris ou les doigts, et il interprète tout seul sept grands
classiques, du début à la fin.

👉 **Démo en ligne : <https://al00ss.github.io/piano-python/>**

La page est volontairement sobre : les boutons des morceaux, une barre de lecture et le clavier.

Le son n'utilise **aucun enregistrement** : chaque note est calculée par NumPy au chargement
de la page. Python s'exécute dans le navigateur grâce à [Pyodide](https://pyodide.org)
(CPython compilé en WebAssembly) ; le seul JavaScript de la page sert à le lancer.

## Les morceaux

| # | Morceau | Compositeur | Durée |
| --- | --- | --- | --- |
| 1 | La Lettre à Élise | Beethoven | 2:59 |
| 2 | Clair de lune | Debussy | 4:53 |
| 3 | Sonate au clair de lune — Adagio | Beethoven | 5:11 |
| 4 | Sonate au clair de lune — Presto | Beethoven | 6:53 |
| 5 | Marche turque | Mozart | 3:33 |
| 6 | Fantaisie-Impromptu, op. 66 | Chopin | 4:36 |
| 7 | Prélude en do majeur | Bach | 2:08 |

Un clic sur un bouton lance le morceau ; un second clic (ou `Échap`) l'arrête. La barre de lecture
met en pause et reprend (`Entrée`), permet de se déplacer dans le morceau et règle le tempo
(50 à 150 %) et le volume ; ces réglages et la disposition du clavier sont mémorisés.

Le piano s'adapte tout seul aux changements de **sortie audio** (casque branché, sortie changée
dans le système) : si aucune sortie n'est utilisable, il l'indique, garde le morceau demandé et
réessaie ; dès que le son revient, le morceau démarre ou reprend là où il s'était arrêté. Chaque morceau
est joué **au tempo de sa partition**, changements de mouvement, points d'orgue et ralentis
compris, avec ses nuances et sa pédale. On peut jouer par-dessus pendant la lecture.

## Rechercher une partition en ligne

La barre de recherche, au-dessus des morceaux, interroge [BitMidi](https://bitmidi.com) (plus de
100 000 fichiers MIDI). Un clic sur un résultat (ou `↓` puis `Entrée`) le joue aussitôt au piano,
sans ajouter de bouton : la barre de lecture le contrôle comme les autres morceaux. La batterie
et les notes hors des 88 touches sont ignorées ; les noms venus d'Internet sont affichés comme du
texte. La liste de suggestions est celle d'[autoComplete.js](https://tarekraafat.github.io/autoComplete.js/)
(sans dépendance, fournie dans `vendor/`).

## Jouer

| Moyen | Comment |
| --- | --- |
| Clavier | deux rangées de touches : `W`…`!` (octave grave) et `A`…`$` (octave aiguë), les touches noires sur la rangée du dessus |
| Souris, doigts | clic ou toucher, plusieurs doigts à la fois, glissando ; plus on frappe bas sur la touche, plus la note est forte |
| `Espace` | pédale forte |
| `Entrée` | lecture / pause du morceau |
| `←` `→` | changer d'octave |
| Clavier MIDI | reconnu automatiquement s'il est déjà autorisé dans le navigateur |

## Bibliothèques

| Bibliothèque | Rôle |
| --- | --- |
| [NumPy](https://numpy.org) | synthèse du son des 88 notes, réverbération, analyse de tonalité |
| [mido](https://mido.readthedocs.io) | lecture des fichiers MIDI et de leur carte de tempo |
| [Pyodide](https://pyodide.org) | exécution de Python dans le navigateur, accès au DOM et à la Web Audio API |
| `wave`, `dataclasses`, `bisect`, `asyncio` | bibliothèque standard |

D'autres options ont été étudiées : `music21` et `pretty_midi` (trop lourds pour le
navigateur), `pygame.midi` et `python-rtmidi` (pas de version WebAssembly).

## Comment le son est fabriqué

`python/piano/synthese.py` imite la physique d'une corde frappée par synthèse additive :

- les partiels sont décalés vers l'aigu par la raideur de la corde
  (inharmonicité : `f_k = k·f0·√(1 + B·k²)`) ;
- le point de frappe du marteau creuse certains partiels, et les aigus s'éteignent plus vite ;
- chaque partiel décroît en deux temps, et deux cordes très légèrement désaccordées
  produisent un lent battement ;
- un bref bruit de marteau accompagne l'attaque.

Chaque partiel est une exponentielle complexe amortie, prolongée bloc par bloc par une
contraction `np.einsum` (dans le navigateur, NumPy n'a pas de BLAS : `einsum` y est plusieurs fois
plus rapide que le produit matriciel). Les 88 notes sont calculées en environ 2 secondes dans
Chrome ou Edge, une vingtaine dans Firefox. Elles sont ensuite gardées dans le cache du
navigateur : aux visites suivantes, seuls Python et NumPy sont à charger. Le cache porte
l'empreinte de `synthese.py` : toute modification de la synthèse le renouvelle.

## Les fichiers MIDI

`outils/preparer_morceaux.py` part des gravures du Mutopia Project (`outils/sources/`), qui
contiennent les bonnes notes sous une forme mécanique, et en fait une interprétation :
reprises dépliées (première puis seconde fin), nuances, tempo et pédale.

```bash
pip install mido numpy
python outils/preparer_morceaux.py     # régénère midi/
```

## Structure

```text
index.html                    la page (et le lancement de Pyodide)
assets/style.css              styles
python/app.py                 l'application : clavier, son, lecture automatique
python/piano/theorie.py       notes, fréquences, géométrie, touches du clavier
python/piano/morceau.py       lecture MIDI (mido), pédale
python/piano/synthese.py      synthèse du piano (NumPy)
python/piano/analyse.py       tonalité, tessiture, polyphonie
python/piano/recherche.py     recherche de partitions en ligne (BitMidi)
midi/                         les sept morceaux
outils/                       préparation des morceaux
vendor/                       paquet mido et autoComplete.js
tests/                        tests unitaires
```

## En local

```bash
python -m http.server 8000                  # puis ouvrir http://localhost:8000
pip install numpy mido                      # pour les tests
python -m unittest discover -s tests -v     # 21 tests
```

> Le site doit être servi en HTTP (pas ouvert en `file://`).

## Crédits et licences

Conçu et développé par **Alexandre Besnier** — [github.com/aL00ss](https://github.com/aL00ss).
Code sous licence MIT.

- **Partitions** : gravures du [Mutopia Project](https://www.mutopiaproject.org), dans le
  domaine public (Stelios Samelis, Keith OHara, Rune Zedeler et Chris Sawer, Guy D. Lederfein,
  Tobias Erbsland). Les deux mouvements de la Sonate au clair de lune (Chris Sawer et Stewart
  Holmes) sont sous licence [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/),
  comme leurs adaptations `midi/sonate-au-clair-de-lune*.mid`.
- **mido** : licence MIT, © Ole Martin Bjørndalen. **NumPy** : licence BSD. **Pyodide** : MPL 2.0.

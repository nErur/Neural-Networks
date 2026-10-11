"""Tâches du banc : d'où viennent les données, comment on les découpe, comment on les montre.

Une tâche fournit :
    forme_entree, classes                      formes et noms des sorties
    donnees(partie) -> (X float32, y int64, groupes ou None)   partie : train / validation / test
    cycles_pretraitement()                     coût estimé du prétraitement sur l'ESP (cycles)
    pretraitement_esp()                        description pour l'export (None : non déployable)
    afficher(ax, x, titre)                     dessin d'un exemple (galerie d'erreurs)

Règle commune : on choisit sur « validation » ; « test » ne sert qu'une fois, pour
le réseau retenu (fonction explicite dans banc.essai).

« groupes » sert au découpage : deux exemples du même groupe (pour SpectraSense,
deux fenêtres du même cycle) ne doivent jamais se retrouver dans deux parties
différentes, sinon le réseau reconnaît en partie des données déjà vues.
"""
import os
import sys

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Tache:
    nom = "?"
    classes = ()
    forme_entree = ()

    def __init__(self, **options):
        self.options = options
        self._cache = {}

    def donnees(self, partie):
        if partie not in self._cache:
            self._cache[partie] = self._charger(partie)
        return self._cache[partie]

    def _charger(self, partie):
        raise NotImplementedError

    def cycles_pretraitement(self):
        return 0.0

    def pretraitement_esp(self):
        return None

    def afficher(self, ax, x, titre=""):
        ax.plot(np.ravel(x))
        ax.set_title(titre, fontsize=8)


# ---------------------------------------------------------------- MNIST

class Mnist(Tache):
    """MNIST de Nielsen, réduit par blocs f×f (esp32/python/donnees.py : même
    spécification que le prétraitement de l'ESP). Options : facteur (1, 2, 4)."""
    nom = "mnist"
    classes = tuple(str(i) for i in range(10))

    def __init__(self, facteur=2, **options):
        super().__init__(facteur=facteur, **options)
        self.facteur = facteur
        cote = 28 // facteur
        self.forme_entree = (cote * cote,)

    def _charger(self, partie):
        sys.path.insert(0, os.path.join(RACINE, "esp32", "python"))
        from donnees import charger_u8, pretraiter
        if "u8" not in self._cache:
            self._cache["u8"] = charger_u8()
        img, y = self._cache["u8"][partie]
        return pretraiter(img, self.facteur), y.astype(np.int64), None

    def cycles_pretraitement(self):
        # mesuré sur l'ESP (f = 2) : 23 600 cycles pour 196 sorties ≈ 196 × (63 + 15·f²)
        return self.forme_entree[0] * (63 + 15 * self.facteur ** 2)

    def pretraitement_esp(self):
        return {"cote_brut": 28, "facteur": self.facteur, "diviseur": 255 * self.facteur ** 2}

    def afficher(self, ax, x, titre=""):
        cote = int(round(np.sqrt(np.size(x))))
        ax.imshow(np.reshape(x, (cote, cote)), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(titre, fontsize=8)
        ax.axis("off")


# ---------------------------------------------------------------- signal 1D synthétique

class Signal(Tache):
    """Signaux 1D synthétiques, pour valider les convolutions avant SpectraSense.

    Quatre formes (créneau, rampe, double impulsion, salve sinusoïdale), placées à
    une position aléatoire, amplitude variable, bruit gaussien. Un perceptron doit
    apprendre chaque position séparément ; une convolution reconnaît la forme
    où qu'elle soit. Options : longueur (128), bruit (0.15), n_train (8000)."""
    nom = "signal"
    classes = ("créneau", "rampe", "double", "salve")

    def __init__(self, longueur=128, bruit=0.15, n_train=8000, **options):
        super().__init__(longueur=longueur, bruit=bruit, n_train=n_train, **options)
        self.L, self.bruit, self.n_train = longueur, bruit, n_train
        self.forme_entree = (1, longueur)

    def _charger(self, partie):
        n = {"train": self.n_train, "validation": 2000, "test": 2000}[partie]
        rng = np.random.default_rng({"train": 11, "validation": 12, "test": 13}[partie])
        L, w = self.L, self.L // 4
        X = np.zeros((n, 1, L), dtype=np.float64)
        y = rng.integers(0, 4, n)
        t = np.arange(w)
        motifs = [np.ones(w), t / (w - 1),
                  np.r_[np.ones(w // 4), np.zeros(w // 2), np.ones(w - w // 4 - w // 2)],
                  np.sin(2 * np.pi * 3 * t / w)]
        for i in range(n):
            debut = rng.integers(0, L - w + 1)
            X[i, 0, debut:debut + w] = rng.uniform(0.5, 1.0) * motifs[y[i]]
        X += self.bruit * rng.standard_normal(X.shape)
        return X.astype(np.float32), y.astype(np.int64), None


# ---------------------------------------------------------------- SpectraSense (gabarit)

class SpectraSense(Tache):
    """Gabarit de la tâche SpectraSense : à compléter quand les données seront prêtes.

    À définir :
      - source : cycles isolés (CSV de SpectraSense), puissance rééchantillonnée
        à pas fixe (tenue d'ordre 0), canal « âge de la donnée » (lectures PZEM perdues) ;
      - fenêtres glissantes : forme_entree = (canaux, longueur) ; étiquette = phase en cours ;
      - groupes = identifiant du cycle : découpage train / validation / test PAR CYCLE
        (fonction decouper_par_groupe) ;
      - mesure propre : délai entre le début d'une phase et le moment où le réseau la
        publie avec une confiance suffisante (critère principal du projet) ;
      - plus tard : une tête de sortie par appareil.
    """
    nom = "spectrasense"

    def _charger(self, partie):
        raise NotImplementedError("tâche SpectraSense pas encore branchée sur les données")


def decouper_par_groupe(groupes, fractions=(0.7, 0.15, 0.15), graine=0):
    """Indices (train, validation, test) tels qu'aucun groupe ne soit coupé en deux."""
    uniques = np.unique(groupes)
    rng = np.random.default_rng(graine)
    rng.shuffle(uniques)
    bornes = np.cumsum(np.r_[0, fractions]) * len(uniques)
    parties = [uniques[int(round(a)):int(round(b))] for a, b in zip(bornes[:-1], bornes[1:])]
    return [np.flatnonzero(np.isin(groupes, p)) for p in parties]


TACHES = {"mnist": Mnist, "signal": Signal, "spectrasense": SpectraSense}
_INSTANCES = {}


def tache(nom, options=None):
    """Instance partagée (les données ne sont chargées qu'une fois par session)."""
    cle = (nom, tuple(sorted((options or {}).items())))
    if cle not in _INSTANCES:
        _INSTANCES[cle] = TACHES[nom](**(options or {}))
    return _INSTANCES[cle]

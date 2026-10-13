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
# dépôt SpectraSense (voisin de celui-ci par défaut ; sinon variable d'environnement SPECTRASENSE)
DOSSIER_SPECTRASENSE = os.environ.get("SPECTRASENSE", os.path.join(os.path.dirname(RACINE), "spectrasense"))


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
    """Ligne arrière-cuisine (ESP + PZEM) : fenêtres de puissance -> état de la ligne.

    Les fenêtres sont fabriquées côté SpectraSense (tools/dataset/fenetres.py), une seule fois
    par jeu d'options ; le banc ne fait que les lire (numpy seulement, pas de pandas ici).
    Un exemple = les `longueur` dernières secondes (canaux × longueur), étiquette = état de la
    ligne à la dernière seconde, parmi 12 classes (une seule chose tourne à la fois).
    Seules les secondes « porte ouverte » (P > 20 W récemment) sont des exemples : sur l'ESP,
    les sorties du réseau sont ignorées porte fermée.

    Options :
      longueur     fenêtre en secondes (128)
      canaux       "log+lin" : ln(1+P)/ln(3001) et P/3000 ; "valide" possible en plus
      copies       passes augmentées de l'entraînement (pertes de lecture, amplitude) (2)
      synth        part de grille-pains synthétiques dans l'entraînement (0.10)
      equilibrage  "racine" (chaque classe pèse comme la racine de sa fréquence) ou "aucun"

    Validation : réelle, sans augmentation, avec 2 grille-pains synthétiques par matin
    (la validation n'en contient aucun de réel). Test : réel, tel quel.
    groupes = ouverture de porte d'origine (les copies augmentées d'un même événement partagent
    leur groupe).
    Le vrai critère (latence, fausses alertes / semaine) se mesure côté SpectraSense :
        python tools/evaluation/evaluer.py --source reseau --modele <banc/resultats/modeles/ID.npz>
    """
    nom = "spectrasense"
    classes = ("repos", "allumage_chauffe", "allumage_regulation", "allumage_fin",
               "cafe_broyage", "cafe_extraction", "cafe_fin", "pic_maintien", "extinction",
               "grille_pain", "bouilloire", "consommateur_inconnu")

    def __init__(self, longueur=128, canaux="log+lin", copies=2, synth=0.10, equilibrage="racine",
                 **options):
        super().__init__(longueur=longueur, canaux=canaux, copies=copies, synth=synth,
                         equilibrage=equilibrage, **options)
        self.canaux = tuple(canaux.split("+"))
        self.forme_entree = (len(self.canaux), int(longueur))
        self.dossier = os.path.join(
            DOSSIER_SPECTRASENSE, "data", "fenetres",
            "L%d_%s_c%d_s%d_%s" % (longueur, "+".join(self.canaux), copies, round(100 * synth), equilibrage))

    def _charger(self, partie):
        chemin = os.path.join(self.dossier, partie + ".npz")
        if not os.path.exists(chemin):
            raise FileNotFoundError(
                "fenêtres absentes : %s\nles fabriquer depuis le dépôt spectrasense :\n"
                "    python tools/dataset/fenetres.py --longueur %d --canaux %s --copies %d --synth %g "
                "--equilibrage %s" % (chemin, self.forme_entree[1], ",".join(self.canaux),
                                      self.options["copies"], self.options["synth"],
                                      self.options["equilibrage"]))
        z = np.load(chemin)
        return z["X"], z["y"], z["groupes"]

    def cycles_pretraitement(self):
        # par seconde : une moyenne de case, un logarithme, un décalage de la fenêtre circulaire
        return 600.0 * len(self.canaux)

    def pretraitement_esp(self):
        return {"pas_s": 1, "longueur": self.forme_entree[1], "canaux": list(self.canaux),
                "p_ref_w": 3000.0, "porte_w": 20.0, "porte_tenue_s": 5}

    def afficher(self, ax, x, titre=""):
        x = np.reshape(x, self.forme_entree)
        for c, nom in enumerate(self.canaux):
            ax.plot(x[c], lw=0.8, label=nom)
        ax.axvline(x.shape[1] - 1, color="r", lw=0.5, ls=":")
        ax.set_title(titre, fontsize=8)
        ax.tick_params(labelsize=6)


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

"""Mode lot : générer des configurations et les entraîner par paquets sur la carte graphique.

    from banc import lot
    cfgs = lot.grille(base, eta=[0.1, 0.3, 1.0], lmbda=[0, 5], graine=[1, 2, 3])
    lot.lancer_lot(cfgs)                     # carte graphique si présente, sinon processeur

Les configurations qui partagent l'architecture, la tâche, le nombre d'époques,
la taille de mini-batch et l'ordre des images (config.CLES_PAQUET) sont regroupées
en paquets de R réseaux entraînés ensemble. η, λ, η final et graine des poids
peuvent différer d'un réseau à l'autre. Chaque réseau est ensuite enregistré dans
le journal exactement comme un essai interactif : un essai en lot se rejoue à
l'identique avec essai.lancer(config).

Mesures (esp32/ESSAIS.md, section 5) : la carte sature vers 512 réseaux [196, 30, 10] ;
au-delà, le temps par réseau ne baisse plus.
"""
import itertools
import time

import numpy as np

from .config import cle_paquet, completer, resume
from .essai import _resultat, enregistrer, existe, preparer
from .moteur import Paquet
from .taches import tache


# ---------------------------------------------------------------- générer des configurations

def grille(base=None, **axes):
    """Produit cartésien : grille(base, eta=[0.1, 0.5], couches=[[...], [...]])."""
    noms = list(axes)
    return [completer(base, **dict(zip(noms, valeurs)))
            for valeurs in itertools.product(*(axes[n] for n in noms))]


def aleatoire(base=None, n=50, graine=0, **espaces):
    """Tirage aléatoire : espaces = {nom: liste de valeurs} ou {nom: ("log", min, max)}
    ou {nom: ("lin", min, max)}. Plus efficace qu'une grille quand beaucoup de
    paramètres comptent peu (Bergstra & Bengio, 2012)."""
    rng = np.random.default_rng(graine)
    cfgs = []
    for _ in range(n):
        tirage = {}
        for nom, esp in espaces.items():
            if isinstance(esp, tuple) and esp[0] == "log":
                tirage[nom] = float(np.exp(rng.uniform(np.log(esp[1]), np.log(esp[2]))))
            elif isinstance(esp, tuple) and esp[0] == "lin":
                tirage[nom] = float(rng.uniform(esp[1], esp[2]))
            else:
                tirage[nom] = esp[rng.integers(len(esp))]
        cfgs.append(completer(base, **tirage))
    return cfgs


def avec_graines(cfgs, graines=(1, 2, 3)):
    """Chaque configuration répétée avec plusieurs graines de poids (pour mesurer le bruit)."""
    return [completer(c, graine=g) for c in cfgs for g in graines]


# ---------------------------------------------------------------- lancer

def lancer_lot(cfgs, dispositif=None, taille_paquet=256, sauter_existants=True, rappel=print):
    """Entraîne toutes les configurations ; renvoie la liste des résultats du journal."""
    import torch
    dispositif = dispositif or ("cuda" if torch.cuda.is_available() else "cpu")
    cfgs = [completer(c) for c in cfgs]
    if sauter_existants:
        deja = [c for c in cfgs if existe(c)]
        cfgs = [c for c in cfgs if not existe(c)]
        if deja and rappel:
            rappel("%d configuration(s) déjà dans le journal : sautée(s)" % len(deja))
    groupes = {}
    for c in cfgs:
        groupes.setdefault(cle_paquet(c), []).append(c)
    paquets = [g[i:i + taille_paquet] for g in groupes.values() for i in range(0, len(g), taille_paquet)]
    if rappel:
        rappel("%d réseau(x) en %d paquet(s) sur %s" % (len(cfgs), len(paquets), dispositif))
    resultats, debut = [], time.perf_counter()
    for i, paquet in enumerate(paquets):
        resultats += _lancer_paquet(paquet, dispositif)
        if rappel:
            meilleur = max(resultats, key=lambda r: r["validation"])
            rappel("paquet %d/%d (%d réseaux) fini, %.0f s écoulées ; meilleur : %.2f %% %s"
                   % (i + 1, len(paquets), len(paquet), time.perf_counter() - debut,
                      meilleur["validation"], meilleur["resume"]))
    return resultats


def _lancer_paquet(cfgs, dispositif):
    c0 = cfgs[0]
    t = tache(c0["tache"], c0["tache_options"])
    p = Paquet(cfgs, t.forme_entree, len(t.classes), "torch", dispositif)
    _, d = preparer(p, c0)
    Xtr, Ytr, _ = d["train"]
    Xva, _, yva = d["validation"]
    historiques = [[] for _ in cfgs]
    calcul = 0.0
    for e in range(c0["epoques"]):
        t0 = time.perf_counter()
        p.epoque(Xtr, Ytr)
        if dispositif.startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        calcul += time.perf_counter() - t0
        prec, cout = p.evaluer(Xva, yva)
        for r in range(p.R):
            historiques[r].append({"epoque": e + 1, "validation": round(float(prec[r]), 3),
                                   "cout": round(float(cout[r]), 5), "eta": float(p.dernier_eta[r])})
    resultats = []
    for r, c in enumerate(cfgs):
        res = _resultat(c, historiques[r], calcul / p.R, "torch-%s-lot%d" % (dispositif, p.R))
        enregistrer(res, p.poids(r))
        resultats.append(res)
    return resultats


if __name__ == "__main__":
    # démonstration : 3 η × 2 λ × 3 graines sur MNIST 14×14, 10 époques
    cfgs = avec_graines(grille({"epoques": 10}, eta=[0.1, 0.5, 2.0], lmbda=[0.0, 5.0]))
    for r in sorted(lancer_lot(cfgs), key=lambda r: -r["validation"])[:5]:
        print("%.2f %%  %s" % (r["validation"], r["resume"]))

"""Étape 1 du pipeline ESP32 : entraîne un petit réseau sur MNIST réduit.

    python esp32\\python\\train.py                    # [196, 30, 10], 14×14
    python esp32\\python\\train.py --facteur 1        # référence 28×28 [784, 30, 10]
    python esp32\\python\\train.py --facteur 4        # 7×7 [49, 30, 10]

Réseau : ReseauEtendu (outils/reseau_etendu.py), couche cachée sigmoïde,
sortie softmax + coût log-vraisemblance, régularisation L2 de network2,
planning cosinus (même formule que outils/visu.py).

Sortie : esp32/modeles/<nom>.npz, lu par export.py.
"""
import argparse
import datetime
import json
import os
import random
import sys
import time
import warnings
from math import cos, pi

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
sys.path[:0] = [ICI, os.path.join(RACINE, "nielsen", "src"), os.path.join(RACINE, "outils")]
warnings.filterwarnings("ignore", category=SyntaxWarning)   # docstrings de Nielsen

from donnees import charger_u8, pretraiter            # noqa: E402
from reseau_etendu import ReseauEtendu                 # noqa: E402


def eta_cosinus(t, eta, epoques, echauffement, eta_fin, m_par_epoque):
    """η après t époques (t réel) : montée linéaire puis cosinus, comme visu.py."""
    w = echauffement
    if w > 0 and t < w:
        return eta * (t + 1.0 / m_par_epoque) / w
    if epoques <= w:
        return eta
    return eta_fin + 0.5 * (eta - eta_fin) * (1 + cos(pi * (t - w) / (epoques - w)))


def precision(net, x, y):
    """Pourcentage de bonnes réponses, en float64 (référence Python)."""
    a = x.T.astype(np.float64)                         # (n_in, N) : toutes les images d'un coup
    for b, w in zip(net.biases[:-1], net.weights[:-1]):
        a = net.f(w @ a + b)
    z = net.weights[-1] @ a + net.biases[-1]           # argmax(softmax) = argmax(z)
    return 100.0 * np.mean(np.argmax(z, axis=0) == y)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--facteur", type=int, default=2, help="réduction f×f (1, 2, 4)")
    p.add_argument("--caches", type=int, nargs="+", default=[30], help="tailles des couches cachées")
    p.add_argument("--epoques", type=int, default=30)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--eta", type=float, default=0.5, help="η maximal du planning cosinus")
    p.add_argument("--echauffement", type=float, default=1.0)
    p.add_argument("--eta-fin", type=float, default=0.0)
    p.add_argument("--lmbda", type=float, default=5.0)
    p.add_argument("--graine", type=int, default=1)
    p.add_argument("--n-images", type=int, default=None, help="sous-ensemble d'entraînement (essais rapides)")
    p.add_argument("--nom", default=None, help="nom du fichier de sortie (défaut : d'après l'architecture)")
    args = p.parse_args()

    random.seed(args.graine)
    np.random.seed(args.graine)

    d = charger_u8()
    x_tr, y_tr = pretraiter(d["train"][0], args.facteur), d["train"][1]
    x_va, y_va = pretraiter(d["validation"][0], args.facteur), d["validation"][1]
    x_te, y_te = pretraiter(d["test"][0], args.facteur), d["test"][1]
    if args.n_images:
        x_tr, y_tr = x_tr[:args.n_images], y_tr[:args.n_images]

    n_in = x_tr.shape[1]
    tailles = [n_in] + args.caches + [10]
    net = ReseauEtendu(tailles, activation="sigmoide", sortie="softmax")

    # format de Nielsen : liste de (x (n_in, 1), y one-hot (10, 1)), en float64
    un_hot = np.eye(10)[y_tr][:, :, None]
    donnees = list(zip(x_tr.astype(np.float64)[:, :, None], un_hot))
    n = len(donnees)
    m_par_epoque = -(-n // args.batch)

    print("Réseau %s · entrées %d×%d · %d images · %d époques · m=%d · η=%g (cosinus) · λ=%g"
          % (tailles, 28 // args.facteur, 28 // args.facteur, n, args.epoques,
             args.batch, args.eta, args.lmbda))
    historique = []
    faits, t0 = 0, time.perf_counter()
    for e in range(args.epoques):
        debut = time.perf_counter()
        random.shuffle(donnees)
        for k in range(0, n, args.batch):
            eta_k = eta_cosinus(faits / m_par_epoque, args.eta, args.epoques,
                                args.echauffement, args.eta_fin, m_par_epoque)
            net.update_mini_batch(donnees[k:k + args.batch], eta_k, args.lmbda, n)
            faits += 1
        calcul = time.perf_counter() - debut
        p_va = precision(net, x_va, y_va)
        historique.append((e + 1, p_va, calcul))
        print("  époque %2d  validation %.2f %%  η fin d'époque %.3f  (%.1f s)"
              % (e + 1, p_va, eta_k, calcul))

    p_te = precision(net, x_te, y_te)
    duree = time.perf_counter() - t0
    print("Test : %.2f %% · durée totale %.0f s" % (p_te, duree))

    nom = args.nom or "mnist_%dx%d_%s" % (28 // args.facteur, 28 // args.facteur,
                                         "_".join(str(t) for t in tailles))
    chemin = os.path.join(RACINE, "esp32", "modeles", nom + ".npz")
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    meta = {
        "nom": nom,
        "tailles": tailles,
        "activations": ["sigmoide"] * len(args.caches) + ["softmax"],
        "pretraitement": {"cote_brut": 28, "facteur": args.facteur,
                          "diviseur": 255 * args.facteur ** 2},
        "precision_test": round(p_te, 2),
        "precision_validation": round(historique[-1][1], 2),
        "hyperparametres": {k: v for k, v in vars(args).items() if k != "nom"},
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "historique": historique,
    }
    tableaux = {}
    for i, (w, b) in enumerate(zip(net.weights, net.biases)):
        tableaux["W%d" % i] = w
        tableaux["b%d" % i] = b.ravel()
    np.savez(chemin, meta=json.dumps(meta, ensure_ascii=False), **tableaux)
    print("Modèle écrit :", os.path.relpath(chemin, RACINE))


if __name__ == "__main__":
    main()

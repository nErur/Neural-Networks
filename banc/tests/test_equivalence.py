"""Chaîne de tests d'équivalence du moteur du banc.

    python -m banc.tests.test_equivalence          (depuis la racine du dépôt)

1. numpy float64 = network2 / ReseauEtendu de Nielsen (mêmes poids, même ordre)
2. gradients (dense, conv1d, maxpool ; sigmoïde, ReLU, tanh) = différences finies
3. un paquet de 3 réseaux = 3 réseaux entraînés séparément
4. PyTorch (processeur) = numpy, en float64 puis en float32
5. PyTorch sur la carte graphique = numpy float32 (si une carte est présente)
"""
import os
import sys
import time
import warnings

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [RACINE, os.path.join(RACINE, "nielsen", "src"), os.path.join(RACINE, "outils")]
warnings.filterwarnings("ignore", category=SyntaxWarning)

from banc.config import completer                     # noqa: E402
from banc.moteur import Paquet                        # noqa: E402
from banc.taches import tache                         # noqa: E402

RESULTATS = []


def verdict(nom, ecart, seuil):
    ok = ecart <= seuil
    RESULTATS.append(ok)
    print("  [%s] %-62s écart %.1e (seuil %.0e)" % ("OK " if ok else "ÉCHEC", nom, ecart, seuil))


def ecart_params(pa, pb, r_a=0, r_b=0):
    e = 0.0
    for a, b in zip(pa.poids(r_a), pb.poids(r_b)):
        if a is not None:
            e = max(e, np.abs(a["W"] - b["W"]).max(), np.abs(a["b"] - b["b"]).max())
    return e


def donnees_mnist(n):
    X, y, _ = tache("mnist", {"facteur": 2}).donnees("train")
    return X[:n], y[:n]


# ---------------------------------------------------------------- 1. Nielsen
def test_nielsen():
    print("1. numpy float64 = network2 de Nielsen (ReseauEtendu, sigmoïde + softmax)")
    from reseau_etendu import ReseauEtendu
    X, y = donnees_mnist(50000)
    cfg = completer(precision="float64", eta=0.5, lmbda=5.0)
    p = Paquet([cfg], (196,), 10)
    net = ReseauEtendu([196, 30, 10], sortie="softmax")
    for i, w in enumerate(p.poids(0)):                # mêmes poids de départ
        net.weights[i] = w["W"].copy()
        net.biases[i] = w["b"].reshape(-1, 1).copy()
    Y = np.eye(10)[y]
    rng = np.random.default_rng(5)
    n, m = 50000, 10
    for k in range(300):
        idx = rng.choice(n, m, replace=False)
        eta = 0.5 * (1 + np.cos(k / 300))             # η qui varie, comme un planning
        net.update_mini_batch([(X[i].astype(np.float64)[:, None], Y[i][:, None]) for i in idx], eta, 5.0, n)
        p.pas(p.xp.vers(X[idx]), p.xp.vers(Y[idx]), np.array([eta]), np.array([5.0]), n)
    e = 0.0
    for i, w in enumerate(p.poids(0)):
        e = max(e, np.abs(net.weights[i] - w["W"]).max(), np.abs(net.biases[i].ravel() - w["b"]).max())
    verdict("300 mises à jour, poids identiques à Nielsen", e, 1e-10)


# ---------------------------------------------------------------- 2. gradients
def gradients_par_pas(p, X, Y):
    """Gradient (somme sur le batch) déduit d'un pas η = 1, λ = 0 : W' = W − (1/m)·g."""
    avant = [None if q is None else {k: v.copy() for k, v in q.items()} for q in p.params]
    m = X.shape[0]
    p.pas(X, Y, np.array([1.0]), np.array([0.0]), 1)
    g = [None if q is None else {k: m * (avant[i][k] - q[k]) for k, q2 in [(k, q[k]) for k in q]}
         for i, q in enumerate(p.params)]
    p.params = avant
    return g


def perte(p, X, y):
    sortie, _ = p._avant(X, False)
    return -np.log(sortie[0, np.arange(len(y)), y]).sum()


def test_gradients():
    print("2. gradients du moteur = différences finies (float64)")
    rng = np.random.default_rng(0)
    cas = {
        "dense sigmoïde + dense ReLU": ((12,), [{"type": "dense", "n": 7, "act": "sigmoide"},
                                               {"type": "dense", "n": 5, "act": "relu"}]),
        "conv1d tanh (pas 2) + maxpool + dense": ((2, 21), [{"type": "conv1d", "canaux": 3, "noyau": 5, "pas": 2, "act": "tanh"},
                                                          {"type": "maxpool1d", "taille": 2},
                                                          {"type": "dense", "n": 6, "act": "sigmoide"}]),
        "conv1d ReLU + conv1d sigmoïde + dense": ((1, 30), [{"type": "conv1d", "canaux": 4, "noyau": 3, "act": "relu"},
                                                          {"type": "conv1d", "canaux": 2, "noyau": 4, "pas": 3, "act": "sigmoide"},
                                                          {"type": "dense", "n": 5, "act": "tanh"}]),
    }
    for nom, (forme, couches) in cas.items():
        p = Paquet([completer(couches=couches, precision="float64", graine=3)], forme, 4)
        X = rng.standard_normal((6,) + forme)
        y = rng.integers(0, 4, 6)
        Y = np.eye(4)[y]
        g = gradients_par_pas(p, X, Y)
        pire, h = 0.0, 1e-6
        for i, q in enumerate(p.params):
            if q is None:
                continue
            for k in ("W", "b"):
                v = q[k].reshape(-1)
                for j in rng.choice(v.size, min(8, v.size), replace=False):
                    sauve = v[j]
                    v[j] = sauve + h
                    lp = perte(p, X, y)
                    v[j] = sauve - h
                    lm = perte(p, X, y)
                    v[j] = sauve
                    num = (lp - lm) / (2 * h)
                    ana = g[i][k].reshape(-1)[j]
                    pire = max(pire, abs(num - ana) / max(1e-6, abs(num) + abs(ana)))
        verdict(nom + " (écart relatif)", pire, 1e-6)


# ---------------------------------------------------------------- 3. paquet = réseaux séparés
def test_paquet():
    print("3. un paquet de 3 réseaux = 3 réseaux séparés (η, λ, graine différents)")
    X, y = donnees_mnist(2000)
    cfgs = [completer(precision="float64", eta=e, lmbda=l, graine=g, epoques=2)
            for e, l, g in ((0.5, 5.0, 1), (0.1, 0.0, 2), (1.0, 20.0, 3))]
    paquet = Paquet(cfgs, (196,), 10)
    Xm, Ym = paquet.xp.vers(X), paquet.xp.vers(np.eye(10)[y])
    for _ in range(2):
        paquet.epoque(Xm, Ym)
    e = 0.0
    for r, c in enumerate(cfgs):
        seul = Paquet([c], (196,), 10)
        for _ in range(2):
            seul.epoque(Xm, Ym)
        e = max(e, ecart_params(paquet, seul, r, 0))
    verdict("2 époques sur 2 000 images", e, 1e-12)


# ---------------------------------------------------------------- 4 et 5. PyTorch = numpy
def comparer_torch(dispositif, precision, seuil, couches=None, forme=(196,), X=None, y=None, nom=""):
    cfg = completer(precision=precision, couches=couches or completer()["couches"], epoques=1)
    a = Paquet([cfg], forme, 10 if X is None else int(y.max()) + 1)
    b = Paquet([cfg], forme, a.n_classes, moteur="torch", dispositif=dispositif)
    K = a.n_classes
    t0 = time.perf_counter()
    a.epoque(a.xp.vers(X), a.xp.vers(np.eye(K)[y]))
    t1 = time.perf_counter()
    b.epoque(b.xp.vers(X), b.xp.vers(np.eye(K)[y]))
    t2 = time.perf_counter()
    verdict("%s %s, %s (numpy %.2f s, torch %.2f s)" % (dispositif, precision, nom, t1 - t0, t2 - t1),
            ecart_params(a, b), seuil)


def test_torch():
    try:
        import torch
    except ImportError:
        print("4-5. PyTorch absent : sauté")
        return
    X, y = donnees_mnist(5000)
    sig = tache("signal").donnees("train")
    conv = [{"type": "conv1d", "canaux": 8, "noyau": 7, "pas": 2, "act": "relu"},
            {"type": "maxpool1d", "taille": 2}, {"type": "dense", "n": 16, "act": "relu"}]
    print("4. PyTorch (processeur) = numpy, une époque")
    comparer_torch("cpu", "float64", 1e-10, X=X, y=y, nom="MNIST [196,30,10]")
    comparer_torch("cpu", "float32", 1e-4, X=X, y=y, nom="MNIST [196,30,10]")
    comparer_torch("cpu", "float64", 1e-10, conv, (1, 128), sig[0][:2000], sig[1][:2000], "signal, conv1d")
    if torch.cuda.is_available():
        print("5. PyTorch sur la carte graphique = numpy float32")
        comparer_torch("cuda", "float32", 1e-4, X=X, y=y, nom="MNIST [196,30,10]")
        comparer_torch("cuda", "float32", 1e-4, conv, (1, 128), sig[0][:2000], sig[1][:2000], "signal, conv1d")
    else:
        print("5. pas de carte graphique : sauté")


if __name__ == "__main__":
    test_nielsen()
    test_gradients()
    test_paquet()
    test_torch()
    print("\nBilan : %d/%d contrôles OK" % (sum(RESULTATS), len(RESULTATS)))
    sys.exit(0 if all(RESULTATS) else 1)

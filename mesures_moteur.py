"""Mesures pour choisir l'architecture du moteur d'entraînement du banc.

    python mesures_moteur.py            (venv du dépôt activé, quelques minutes)

Toutes les mesures portent sur une époque complète d'un réseau [196, 30, 10]
(sigmoïde puis softmax, régularisation L2 comme network2) sur MNIST 14×14.
La rétropropagation est écrite à la main : pas de calcul automatique des
gradients, donc beaucoup moins d'opérations par mise à jour.

A. Processeur, un réseau : numpy et PyTorch, m = 10 et m = 100.
B. Carte graphique, R réseaux en même temps (1 à 2048), m = 10.
C. Carte graphique, même chose rejouée par un CUDA graph (un seul lancement
   par mise à jour au lieu d'une vingtaine).

Avant de mesurer, deux contrôles : la rétropropagation à la main donne les
mêmes gradients que le calcul automatique de PyTorch, et le CUDA graph donne
les mêmes poids que l'exécution normale.
"""
import os
import sys
import time

import numpy as np
import torch

RACINE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RACINE, "esp32", "python"))

ETA, LMBDA = 0.5, 5.0
TAILLES = (196, 30, 10)


def donnees():
    try:
        from donnees import charger_u8, pretraiter
        d = charger_u8()
        return pretraiter(d["train"][0], 2), d["train"][1].astype(np.int64)
    except Exception:
        rng = np.random.default_rng(0)
        print("  (MNIST introuvable : images aléatoires, seuls les temps comptent)")
        return rng.random((50000, 196), dtype=np.float32), rng.integers(0, 10, 50000)


# ---------------------------------------------------------------- une mise à jour, écrite à la main

def pas_torch(W1, b1, W2, b2, xb, Yb, eta, lmbda, n):
    """Une mise à jour SGD sur R réseaux empilés (poids (R, sortie, entrée)).
    xb : (m, 196) partagé par tous les réseaux, Yb : (m, 10) one-hot.
    Renvoie les nouveaux paramètres (mêmes formules que network2)."""
    m = xb.shape[0]
    a1 = torch.sigmoid(W1 @ xb.T + b1)                     # (R, 30, m)
    z2 = W2 @ a1 + b2                                      # (R, 10, m)
    p = torch.softmax(z2, dim=1)
    d2 = p - Yb.T                                          # δ de sortie : a - y
    d1 = (W2.transpose(1, 2) @ d2) * a1 * (1 - a1)         # rétropropagation
    k = eta / m
    decroit = 1 - eta * lmbda / n                          # L2 : poids seulement
    return (decroit * W1 - k * (d1 @ xb), b1 - k * d1.sum(2, keepdim=True),
            decroit * W2 - k * (d2 @ a1.transpose(1, 2)), b2 - k * d2.sum(2, keepdim=True))


def init_torch(R, dev, dtype=torch.float32, graine=0):
    g = torch.Generator().manual_seed(graine)
    W1 = torch.randn(R, 30, 196, generator=g, dtype=dtype) / 14
    b1 = torch.randn(R, 30, 1, generator=g, dtype=dtype)
    W2 = torch.randn(R, 10, 30, generator=g, dtype=dtype) / 30 ** 0.5
    b2 = torch.randn(R, 10, 1, generator=g, dtype=dtype)
    return [t.to(dev) for t in (W1, b1, W2, b2)]


def controle_gradients():
    """La rétropropagation à la main = autograd, en float64."""
    torch.manual_seed(1)
    W1, b1, W2, b2 = init_torch(3, "cpu", torch.float64)
    xb = torch.rand(10, 196, dtype=torch.float64)
    yb = torch.randint(0, 10, (10,))
    Yb = torch.nn.functional.one_hot(yb, 10).double()
    main = pas_torch(W1, b1, W2, b2, xb, Yb, ETA, 0.0, 1)
    params = [t.clone().requires_grad_() for t in (W1, b1, W2, b2)]
    z = params[2] @ torch.sigmoid(params[0] @ xb.T + params[1]) + params[3]
    perte = sum(torch.nn.functional.cross_entropy(z[r].T, yb, reduction="mean") for r in range(3))
    grads = torch.autograd.grad(perte, params)
    auto = [p.detach() - ETA * g for p, g in zip(params, grads)]
    ecart = max(float((a - b).abs().max()) for a, b in zip(main, auto))
    print("  [%s] rétropropagation à la main = calcul automatique (écart %.1e)"
          % ("OK " if ecart < 1e-12 else "ÉCHEC", ecart))
    return ecart < 1e-12


# ---------------------------------------------------------------- A. processeur

def epoque_numpy(x, Y, m, dtype):
    rng = np.random.default_rng(0)
    W1 = (rng.standard_normal((30, 196)) / 14).astype(dtype)
    b1 = rng.standard_normal((30, 1)).astype(dtype)
    W2 = (rng.standard_normal((10, 30)) / 30 ** 0.5).astype(dtype)
    b2 = rng.standard_normal((10, 1)).astype(dtype)
    x, Y = x.astype(dtype), Y.astype(dtype)
    n = len(x)
    ordre = rng.permutation(n)
    decroit, k = dtype(1 - ETA * LMBDA / n), dtype(ETA / m)
    t0 = time.perf_counter()
    with np.errstate(over="ignore"):
        _boucle_numpy(x, Y, m, n, ordre, decroit, k, W1, b1, W2, b2)
    return time.perf_counter() - t0


def _boucle_numpy(x, Y, m, n, ordre, decroit, k, W1, b1, W2, b2):
    for d in range(0, n, m):
        i = ordre[d:d + m]
        xb, Yb = x[i], Y[i]
        a1 = 1 / (1 + np.exp(-(W1 @ xb.T + b1)))
        z2 = W2 @ a1 + b2
        e = np.exp(z2 - z2.max(0))
        d2 = e / e.sum(0) - Yb.T
        d1 = (W2.T @ d2) * a1 * (1 - a1)
        W2 = decroit * W2 - k * (d2 @ a1.T); b2 = b2 - k * d2.sum(1, keepdims=True)
        W1 = decroit * W1 - k * (d1 @ xb);   b1 = b1 - k * d1.sum(1, keepdims=True)


def epoque_torch(x, Y, m, R, dev):
    """Exécution normale : chaque opération est lancée séparément."""
    p = init_torch(R, dev)
    n = len(x)
    ordre = torch.randperm(n, device=dev)
    synchro(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        for d in range(0, n, m):
            i = ordre[d:d + m]
            p = pas_torch(*p, x[i], Y[i], ETA, LMBDA, n)
    synchro(dev)
    return time.perf_counter() - t0, p


def synchro(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- C. CUDA graph

def epoque_graph(x, Y, m, R, ordre=None):
    """Une mise à jour complète enregistrée une fois, puis rejouée n/m fois.
    Les mini-batches de l'époque sont rangés d'avance ; un compteur sur la carte
    désigne le suivant, si bien que rien ne repasse par Python entre deux pas."""
    dev = torch.device("cuda")
    n = len(x) - len(x) % m
    nb = n // m
    if ordre is None:
        ordre = torch.randperm(len(x), device=dev)
    lots_x = x[ordre[:n]].view(nb, m, 196).contiguous()
    lots_Y = Y[ordre[:n]].view(nb, m, 10).contiguous()
    params = init_torch(R, dev)
    depart = [t.clone() for t in params]
    compteur = torch.zeros(1, dtype=torch.long, device=dev)

    def pas():
        xb = lots_x.index_select(0, compteur).squeeze(0)
        Yb = lots_Y.index_select(0, compteur).squeeze(0)
        nouveaux = pas_torch(*params, xb, Yb, ETA, LMBDA, len(x))
        for p, q in zip(params, nouveaux):
            p.copy_(q)
        compteur.add_(1)

    with torch.no_grad():
        flux = torch.cuda.Stream()
        flux.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(flux):                     # échauffement exigé par PyTorch
            for _ in range(3):
                pas()
        torch.cuda.current_stream().wait_stream(flux)
        graphe = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphe):
            pas()
        for p, q in zip(params, depart):                  # on repart des poids initiaux
            p.copy_(q)
        compteur.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(nb):
            graphe.replay()
        torch.cuda.synchronize()
    return time.perf_counter() - t0, params


def controle_graph(x, Y):
    """CUDA graph = exécution normale, sur la même époque."""
    dev = torch.device("cuda")
    torch.manual_seed(3)
    ordre = torch.randperm(len(x), device=dev)
    _, p_graph = epoque_graph(x, Y, 10, 4, ordre)
    p = init_torch(4, dev)
    n = len(x) - len(x) % 10
    with torch.no_grad():
        for d in range(0, n, 10):
            i = ordre[d:d + 10]
            p = pas_torch(*p, x[i], Y[i], ETA, LMBDA, len(x))
    ecart = max(float((a - b).abs().max()) for a, b in zip(p, p_graph))
    ok = ecart < 1e-5
    print("  [%s] CUDA graph = exécution normale (écart %.1e)" % ("OK " if ok else "ÉCHEC", ecart))
    return ok


# ---------------------------------------------------------------- programme

def main():
    x_np, y_np = donnees()
    Y_np = np.eye(10, dtype=np.float32)[y_np]
    print("Contrôles")
    ok = controle_gradients()
    gpu = torch.cuda.is_available()
    if gpu:
        xg, Yg = torch.from_numpy(x_np).cuda(), torch.from_numpy(Y_np).cuda()
        ok &= controle_graph(xg, Yg)

    print("\nA. Processeur, un réseau (s par époque)")
    xc, Yc = torch.from_numpy(x_np), torch.from_numpy(Y_np)
    epoque_numpy(x_np[:5000], Y_np[:5000], 10, np.float32)          # échauffement
    for m in (10, 100):
        t32 = epoque_numpy(x_np, Y_np, m, np.float32)
        t64 = epoque_numpy(x_np, Y_np, m, np.float64)
        tt, _ = epoque_torch(xc, Yc, m, 1, "cpu")
        print("  m=%-3d  numpy float32 %.2f · numpy float64 %.2f · PyTorch %.2f" % (m, t32, t64, tt))

    if not gpu:
        print("\nPas de carte graphique : mesures B et C sautées.")
        return 0 if ok else 1

    print("\nB/C. Carte graphique, m=10 (s par époque, puis par réseau)")
    epoque_torch(xg, Yg, 100, 1, "cuda")                            # échauffement
    for R in (1, 32, 128, 512, 2048):
        tn, _ = epoque_torch(xg, Yg, 10, R, "cuda")
        tg, _ = epoque_graph(xg, Yg, 10, R)
        print("  R=%-5d normal %6.2f s (%.4f/réseau) · CUDA graph %6.2f s (%.4f/réseau)"
              % (R, tn, tn / R, tg, tg / R))
    print("  Mémoire maximale utilisée sur la carte : %.0f Mo"
          % (torch.cuda.max_memory_allocated() / 2**20))
    print("\nBilan : %s" % ("contrôles OK" if ok else "un contrôle a échoué, mesures à ignorer"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

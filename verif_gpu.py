"""Vérifie que la TITAN Xp est utilisable par PyTorch et mesure le gain réel.

    python verif_gpu.py            (venv du dépôt activé)

1. Environnement : versions, pilote, architecture de la carte (Pascal = sm_61).
2. Contrôles : calculs exacts sur la carte, float64, reproductibilité.
3. Mesures : une époque d'un réseau [196, 30, 10] sur MNIST 14×14, processeur
   contre carte graphique, mini-batch 10 et 100, puis 32 réseaux entraînés
   en même temps (le cas des lots du banc d'essai).
"""
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

RACINE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RACINE, "esp32", "python"))

RESULTATS = []


def verdict(nom, ok, detail=""):
    RESULTATS.append((nom, ok))
    print("  [%s] %s%s" % ("OK " if ok else "ÉCHEC", nom, (" : " + detail) if detail else ""))


# ---------------------------------------------------------------- 1. environnement

def environnement():
    print("1. Environnement")
    print("  Python %s, PyTorch %s, CUDA de PyTorch %s, cuDNN %s"
          % (platform.python_version(), torch.__version__, torch.version.cuda,
             torch.backends.cudnn.version()))
    try:
        sortie = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        print("  nvidia-smi : %s" % (sortie or "aucune carte"))
    except (OSError, subprocess.TimeoutExpired):
        print("  nvidia-smi introuvable")
    dispo = torch.cuda.is_available()
    verdict("CUDA disponible pour PyTorch", dispo)
    if not dispo:
        return False
    cap = torch.cuda.get_device_capability(0)
    sm = "sm_%d%d" % cap
    archs = torch.cuda.get_arch_list()
    print("  Carte : %s, architecture %s, %.1f Go"
          % (torch.cuda.get_device_name(0), sm,
             torch.cuda.get_device_properties(0).total_memory / 2**30))
    print("  Architectures compilées dans PyTorch : %s" % " ".join(archs))
    verdict("code compilé pour %s présent" % sm, sm in archs,
            "" if sm in archs else "PyTorch ne contient pas de code pour cette carte")
    return True


# ---------------------------------------------------------------- 2. contrôles

def controles(gpu):
    print("\n2. Contrôles de calcul")
    g = torch.Generator().manual_seed(0)
    A = torch.randn(512, 512, generator=g, dtype=torch.float64)
    B = torch.randn(512, 512, generator=g, dtype=torch.float64)
    ref = A.numpy() @ B.numpy()

    cpu64 = (A @ B).numpy()
    verdict("float64 processeur = numpy", np.abs(cpu64 - ref).max() < 1e-10,
            "écart %.1e" % np.abs(cpu64 - ref).max())
    if not gpu:
        return
    dev = torch.device("cuda")
    gpu64 = (A.to(dev) @ B.to(dev)).cpu().numpy()
    verdict("float64 carte = numpy", np.abs(gpu64 - ref).max() < 1e-10,
            "écart %.1e" % np.abs(gpu64 - ref).max())
    gpu32 = (A.float().to(dev) @ B.float().to(dev)).cpu().numpy()
    rel = np.abs(gpu32 - ref).max() / np.abs(ref).max()
    verdict("float32 carte proche de float64", rel < 1e-5, "écart relatif %.1e" % rel)

    # reproductibilité : deux entraînements identiques sur la carte
    poids = [entrainer_synthetique(dev, graine=1) for _ in range(2)]
    egal = all(torch.equal(a, b) for a, b in zip(poids[0], poids[1]))
    verdict("deux entraînements identiques sur la carte donnent les mêmes poids", egal)


def entrainer_synthetique(dev, graine, pas=200):
    torch.manual_seed(graine)
    x = torch.rand(2000, 196, device=dev)
    y = torch.randint(0, 10, (2000,), device=dev)
    W1 = torch.randn(30, 196, device=dev) / 14
    W2 = torch.randn(10, 30, device=dev) / 5.5
    for k in range(pas):
        i = slice((k * 10) % 2000, (k * 10) % 2000 + 10)
        W1.requires_grad_(True); W2.requires_grad_(True)
        perte = torch.nn.functional.cross_entropy(torch.sigmoid(x[i] @ W1.T) @ W2.T, y[i])
        g1, g2 = torch.autograd.grad(perte, (W1, W2))
        with torch.no_grad():
            W1, W2 = W1 - 0.5 * g1, W2 - 0.5 * g2
    return [W1.detach().cpu(), W2.detach().cpu()]


# ---------------------------------------------------------------- 3. mesures

def donnees_mnist():
    try:
        from donnees import charger_u8, pretraiter
        d = charger_u8()
        return pretraiter(d["train"][0], 2), d["train"][1].astype(np.int64), "MNIST 14×14"
    except Exception:                               # pas de données : images aléatoires
        rng = np.random.default_rng(0)
        return (rng.random((50000, 196), dtype=np.float32),
                rng.integers(0, 10, 50000), "données aléatoires (MNIST introuvable)")


def epoque(x, y, dev, m, n_reseaux=1):
    """Une époque de SGD sur n_reseaux réseaux [196, 30, 10] indépendants,
    entraînés en même temps (poids empilés, produits matriciels groupés)."""
    torch.manual_seed(0)
    R, n = n_reseaux, len(x)
    W1 = (torch.randn(R, 30, 196, device=dev) / 14).requires_grad_()
    b1 = torch.zeros(R, 30, 1, device=dev, requires_grad=True)
    W2 = (torch.randn(R, 10, 30, device=dev) / 5.5).requires_grad_()
    b2 = torch.zeros(R, 10, 1, device=dev, requires_grad=True)
    ordre = torch.randperm(n, device=dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(0, n, m):
        i = ordre[k:k + m]
        a = x[i].T.expand(R, -1, -1)                          # (R, 196, m)
        z = W2 @ torch.sigmoid(W1 @ a + b1) + b2              # (R, 10, m)
        perte = torch.nn.functional.cross_entropy(
            z.permute(0, 2, 1).reshape(-1, 10), y[i].repeat(R), reduction="sum") / len(i)
        gs = torch.autograd.grad(perte, (W1, b1, W2, b2))
        with torch.no_grad():
            for p, g in zip((W1, b1, W2, b2), gs):
                p -= 0.5 * g
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def mesures(gpu):
    x_np, y_np, nom = donnees_mnist()
    print("\n3. Mesures : une époque de [196, 30, 10] sur %s (%d images)" % (nom, len(x_np)))
    print("  Référence : le code de Nielsen prend environ 1,7 s par époque sur ce réseau.")
    appareils = [torch.device("cpu")] + ([torch.device("cuda")] if gpu else [])
    for dev in appareils:
        x = torch.from_numpy(x_np).to(dev)
        y = torch.from_numpy(y_np).to(dev)
        epoque(x, y, dev, 100)                                  # échauffement
        for m in (10, 100):
            t = epoque(x, y, dev, m)
            print("  %-4s m=%-3d : %.2f s par époque" % (dev.type, m, t))
        if dev.type == "cuda":
            for R in (8, 32):
                t = epoque(x, y, dev, 10, n_reseaux=R)
                print("  cuda m=10, %2d réseaux en même temps : %.2f s, soit %.3f s par réseau"
                      % (R, t, t / R))


if __name__ == "__main__":
    gpu = environnement()
    controles(gpu)
    mesures(gpu)
    echecs = [n for n, ok in RESULTATS if not ok]
    print("\nBilan : %d contrôle(s), %d échec(s)%s"
          % (len(RESULTATS), len(echecs), "" if not echecs else " : " + " ; ".join(echecs)))
    sys.exit(1 if echecs else 0)

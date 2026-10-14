"""Moteur d'entraînement du banc : R réseaux de même architecture, entraînés ensemble.

Un seul code pour numpy (interactif, R = 1, processeur) et PyTorch (lot, R grand,
carte graphique) : les deux « dialectes » ne diffèrent que par la petite classe
_Numpy / _Torch ci-dessous. La rétropropagation est écrite à la main (pas de
calcul automatique des gradients), mêmes formules que network2 de Nielsen :

    W ← (1 − η·λ/n)·W − (η/m)·Σ ∂C/∂W      (L2 sur les poids, pas sur les biais)
    b ← b − (η/m)·Σ ∂C/∂b

Couches : dense, conv1d (sans bourrage, pas réglable), maxpool1d. Activations
cachées : sigmoïde, ReLU, tanh. Sortie : softmax + coût log-vraisemblance.

Formes (R réseaux, N exemples) :
    dense    activations (R, N, n)        poids W (R, n_out, n_in), b (R, 1, n_out)
    conv1d   activations (R, N, C, L)     noyaux K (R, F, C, k),    b (R, 1, F, 1)
L'entrée (N, ...) est partagée par les R réseaux (dimension R de taille 1).

Règle des mini-batches : on ne traite que des mini-batches complets ; les
n mod m dernières images d'une permutation sont sautées pour cette époque.
"""
import math

import numpy as np

from .config import completer


# ---------------------------------------------------------------- dialectes numpy / torch

class _Numpy:
    nom = "numpy"

    def __init__(self, precision):
        self.dt = np.float32 if precision == "float32" else np.float64

    def vers(self, a):                      # tableau numpy -> tableau du moteur
        return np.asarray(a, dtype=self.dt)

    def entiers(self, a):
        return np.asarray(a, dtype=np.int64)

    def numpy(self, x):
        return np.asarray(x)

    def zeros(self, forme):
        return np.zeros(forme, dtype=self.dt)

    exp, tanh, sqrt = staticmethod(np.exp), staticmethod(np.tanh), staticmethod(np.sqrt)

    @staticmethod
    def somme(x, axe, garder=False):
        return x.sum(axis=axe, keepdims=garder)

    @staticmethod
    def maxi(x, axe, garder=False):
        return x.max(axis=axe, keepdims=garder)

    @staticmethod
    def argmax(x, axe):
        return x.argmax(axis=axe)

    @staticmethod
    def permuter(x, axes):
        return x.transpose(axes)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    def positif(self, x):
        return (x > 0).astype(self.dt)

    @staticmethod
    def fenetres(x, k, s):
        """(..., L) -> (..., Lo, k) : les fenêtres de longueur k, au pas s."""
        return np.lib.stride_tricks.sliding_window_view(x, k, axis=-1)[..., ::s, :]

    @staticmethod
    def indexer(x, i):
        return x[i]

    @staticmethod
    def log(x):
        return np.log(x)


class _Torch:
    nom = "torch"

    def __init__(self, precision, dispositif):
        import torch
        self.t = torch
        self.dt = torch.float32 if precision == "float32" else torch.float64
        self.dev = torch.device(dispositif)

    def vers(self, a):
        return self.t.as_tensor(np.asarray(a), dtype=self.dt, device=self.dev)

    def entiers(self, a):
        return self.t.as_tensor(np.asarray(a), dtype=self.t.int64, device=self.dev)

    def numpy(self, x):
        return x.detach().cpu().numpy()

    def zeros(self, forme):
        return self.t.zeros(forme, dtype=self.dt, device=self.dev)

    def exp(self, x):
        return self.t.exp(x)

    def tanh(self, x):
        return self.t.tanh(x)

    def sqrt(self, x):
        return self.t.sqrt(x)

    @staticmethod
    def somme(x, axe, garder=False):
        return x.sum(dim=axe, keepdim=garder)

    @staticmethod
    def maxi(x, axe, garder=False):
        return x.amax(dim=axe, keepdim=garder)

    @staticmethod
    def argmax(x, axe):
        return x.argmax(dim=axe)

    @staticmethod
    def permuter(x, axes):
        return x.permute(axes)

    def relu(self, x):
        return self.t.clamp_min(x, 0)

    def positif(self, x):
        return (x > 0).to(self.dt)

    @staticmethod
    def fenetres(x, k, s):
        return x.unfold(-1, k, s)

    @staticmethod
    def indexer(x, i):
        return x.index_select(0, i)

    def log(self, x):
        return self.t.log(x)


def dialecte(moteur="numpy", precision="float32", dispositif="cpu"):
    return _Numpy(precision) if moteur == "numpy" else _Torch(precision, dispositif)


# ---------------------------------------------------------------- architecture

def plan_couches(cfg, forme_entree, n_classes):
    """Liste des couches avec leurs formes d'entrée et de sortie (par exemple)."""
    plan, forme = [], tuple(forme_entree)
    for c in cfg["couches"] + [{"type": "dense", "n": n_classes, "act": "softmax"}]:
        c = dict(c)
        if c["type"] == "conv1d":
            if len(forme) != 2:
                raise ValueError("conv1d attend une entrée (canaux, longueur), pas %s" % (forme,))
            C, L = forme
            k, s = c["noyau"], c.get("pas", 1)
            Lo = (L - k) // s + 1
            if Lo < 1:
                raise ValueError("noyau %d trop long pour une longueur %d" % (k, L))
            c.update(entree=forme, sortie=(c["canaux"], Lo), pas=s, act=c.get("act", "sigmoide"))
        elif c["type"] == "maxpool1d":
            C, L = forme
            c.update(entree=forme, sortie=(C, L // c["taille"]))
        else:
            n_in = int(np.prod(forme))
            c.update(entree=forme, n_in=n_in, sortie=(c["n"],), act=c.get("act", "sigmoide"))
        plan.append(c)
        forme = c["sortie"]
    return plan


def initialiser(plan, cfg, graine):
    """Poids de départ d'UN réseau (numpy float64), comme network2 : biais N(0, 1) (ou 0 avec
    init_biais = "zero"), poids N(0, 1/n_entrée) (ou N(0, 2/n_entrée) pour ReLU, « He »)."""
    rng = np.random.default_rng(graine)
    params = []
    for c in plan:
        if c["type"] == "maxpool1d":
            params.append(None)
            continue
        if c["type"] == "dense":
            fan_in, forme_W, forme_b = c["n_in"], (c["n"], c["n_in"]), (c["n"],)
        else:
            C = c["entree"][0]
            fan_in, forme_W, forme_b = C * c["noyau"], (c["canaux"], C, c["noyau"]), (c["canaux"],)
        gain = 2.0 if (cfg["init"] == "auto" and c["act"] == "relu") or cfg["init"] == "he" else 1.0
        b = rng.standard_normal(forme_b)               # tiré même si inutilisé : même suite aléatoire pour W
        if cfg.get("init_biais", "normal") == "zero":
            b = np.zeros(forme_b)
        W = rng.standard_normal(forme_W) * math.sqrt(gain / fan_in)
        params.append({"W": W, "b": b})
    return params


# ---------------------------------------------------------------- planning de η

def eta_planning(xp, eta, eta_fin, echauffement, epoques, t, m_par_epoque, planning):
    """η de chaque réseau après t époques (t réel) ; même formule que outils/visu.py."""
    if planning == "constant":
        return eta
    w = echauffement
    if w > 0 and t < w:
        return eta * ((t + 1.0 / m_par_epoque) / w)
    if epoques <= w:
        return eta
    return eta_fin + 0.5 * (eta - eta_fin) * (1 + math.cos(math.pi * (t - w) / (epoques - w)))


# ---------------------------------------------------------------- le paquet de réseaux

class Paquet:
    """R réseaux de même architecture (mêmes CLES_PAQUET), η / λ / graine propres à chacun."""

    def __init__(self, configs, forme_entree, n_classes, moteur="numpy", dispositif="cpu"):
        self.cfgs = [completer(c) for c in configs]
        c0 = self.cfgs[0]
        self.R = len(self.cfgs)
        self.xp = dialecte(moteur, c0["precision"], dispositif)
        self.plan = plan_couches(c0, forme_entree, n_classes)
        self.n_classes = n_classes
        xp = self.xp
        # paramètres empilés : (R, ...) ; initialisés en float64 sur le processeur
        inits = [initialiser(self.plan, c, c["graine"]) for c in self.cfgs]
        self.params = []
        for i, c in enumerate(self.plan):
            if c["type"] == "maxpool1d":
                self.params.append(None)
                continue
            W = np.stack([p[i]["W"] for p in inits])
            b = np.stack([p[i]["b"] for p in inits])
            b = b[:, None, :] if c["type"] == "dense" else b[:, None, :, None]
            self.params.append({"W": xp.vers(W), "b": xp.vers(b)})
        self.eta = np.array([c["eta"] for c in self.cfgs], dtype=np.float64)
        self.eta_fin = np.array([c["eta_fin"] for c in self.cfgs], dtype=np.float64)
        self.lmbda = np.array([c["lmbda"] for c in self.cfgs], dtype=np.float64)
        self.pas_faits = 0
        self.epoques_faites = 0

    # ------------------------------------------------------------ propagation avant
    def _avant(self, X, garder):
        """X : (N, ...) partagé. Renvoie la sortie (R, N, K) et, si garder, le cache."""
        xp = self.xp
        a = X[None]                                       # (1, N, ...)
        cache = []
        for c, p in zip(self.plan, self.params):
            entree = a
            if c["type"] == "dense":
                if a.ndim > 3:                            # aplatissement avant la 1re couche dense
                    a = a.reshape(a.shape[0], a.shape[1], -1)
                z = a @ xp.permuter(p["W"], (0, 2, 1)) + p["b"]
                a_aplatie = a
                a = self._activer(z, c["act"])
                cache.append((entree, a_aplatie, z, a))
            elif c["type"] == "conv1d":
                k, s = c["noyau"], c["pas"]
                F, Lo = c["sortie"]
                C = c["entree"][0]
                cols = xp.fenetres(a, k, s)               # (R|1, N, C, Lo, k)
                cols = xp.permuter(cols, (0, 1, 3, 2, 4)).reshape(cols.shape[0], -1, C * k)
                Kmat = p["W"].reshape(self.R, F, C * k)
                z = (cols @ xp.permuter(Kmat, (0, 2, 1)))  # (R, N*Lo, F)
                z = xp.permuter(z.reshape(self.R, -1, Lo, F), (0, 1, 3, 2)) + p["b"]
                a = self._activer(z, c["act"])
                cache.append((entree, cols, z, a))
            else:                                         # maxpool1d
                t = c["taille"]
                L2 = c["sortie"][1]
                blocs = a[..., :L2 * t].reshape(a.shape[:-1] + (L2, t))
                a = xp.maxi(blocs, -1)
                cache.append((entree, blocs, None, a))
            if not garder:
                cache = []
        return a, cache

    def _activer(self, z, act):
        xp = self.xp
        if act == "sigmoide":
            return 1 / (1 + xp.exp(-z))
        if act == "relu":
            return xp.relu(z)
        if act == "tanh":
            return xp.tanh(z)
        e = xp.exp(z - xp.maxi(z, -1, True))             # softmax stable
        return e / xp.somme(e, -1, True)

    def _derivee(self, z, a, act):
        if act == "sigmoide":
            return a * (1 - a)
        if act == "relu":
            return self.xp.positif(z)
        return 1 - a * a                                  # tanh

    # ------------------------------------------------------------ une mise à jour
    def pas(self, Xb, Yb, eta, lmbda, n):
        """Une mise à jour sur le mini-batch (Xb, Yb one-hot). eta, lmbda : (R,) du moteur."""
        xp = self.xp
        m = Xb.shape[0]
        sortie, cache = self._avant(Xb, True)
        delta = sortie - Yb[None]                         # δ de sortie : a − y   (R, N, K)
        grads = [None] * len(self.plan)
        for i in range(len(self.plan) - 1, -1, -1):
            c, p = self.plan[i], self.params[i]
            entree, x2, z, a = cache[i]
            if i < len(self.plan) - 1 and c["type"] != "maxpool1d":
                delta = delta * self._derivee(z, a, c["act"])
            if c["type"] == "dense":
                gW = xp.permuter(delta, (0, 2, 1)) @ x2   # (R, n_out, n_in) = Σ sur le batch
                gb = xp.somme(delta, 1, True)
                grads[i] = (gW, gb)
                if i > 0:
                    delta = (delta @ p["W"]).reshape((self.R,) + entree.shape[1:])
            elif c["type"] == "conv1d":
                k, s = c["noyau"], c["pas"]
                F, Lo = c["sortie"]
                C, L = c["entree"]
                dzp = xp.permuter(delta, (0, 1, 3, 2)).reshape(self.R, -1, F)   # (R, N*Lo, F)
                gW = (xp.permuter(dzp, (0, 2, 1)) @ x2).reshape(self.R, F, C, k)
                gb = xp.somme(xp.somme(delta, 3, True), 1, True)
                grads[i] = (gW, gb)
                if i > 0:
                    dcols = (dzp @ p["W"].reshape(self.R, F, C * k)).reshape(self.R, m, Lo, C, k)
                    dA = xp.zeros((self.R, m, C, L))
                    for j in range(k):                    # replie les fenêtres sur l'entrée
                        dA[..., j:j + s * (Lo - 1) + 1:s] += xp.permuter(dcols[..., j], (0, 1, 3, 2))
                    delta = dA
            else:                                         # maxpool1d : le gradient va au maximum
                blocs, sortie_p = x2, a
                t = c["taille"]
                gagnant = (blocs == sortie_p[..., None])
                gagnant = gagnant * 1.0 if xp.nom == "numpy" else gagnant.to(blocs.dtype)
                # en cas d'égalité, on ne garde que le premier maximum
                premier = xp.argmax(gagnant, -1)
                masque = self._un_parmi(premier, t, blocs)
                dA = xp.zeros((self.R,) + entree.shape[1:])
                L2 = c["sortie"][1]
                dA[..., :L2 * t] = (masque * delta[..., None]).reshape(dA[..., :L2 * t].shape)
                delta = dA
        # mise à jour, comme network2
        for i, c in enumerate(self.plan):
            if grads[i] is None:
                continue
            gW, gb = grads[i]
            p = self.params[i]
            fW = (1,) * (p["W"].ndim - 1)
            fb = (1,) * (p["b"].ndim - 1)
            p["W"] = (1 - eta * lmbda / n).reshape((-1,) + fW) * p["W"] - (eta / m).reshape((-1,) + fW) * gW
            p["b"] = p["b"] - (eta / m).reshape((-1,) + fb) * gb

    def _un_parmi(self, indices, t, modele):
        """indices (…) -> masque one-hot (…, t) du type de modele."""
        xp = self.xp
        if xp.nom == "numpy":
            return (indices[..., None] == np.arange(t)).astype(modele.dtype)
        return (indices[..., None] == xp.t.arange(t, device=xp.dev)).to(modele.dtype)

    # ------------------------------------------------------------ une époque
    def epoque(self, X, Y, rappel=None, tous_les=None):
        """Une époque de SGD. X (n, ...), Y one-hot (n, K) : tableaux du moteur.
        Ordre des images : graine_ordre et numéro d'époque (identique pour tout le paquet)."""
        c0 = self.cfgs[0]
        n, m = X.shape[0], c0["batch"]
        M = n // m
        rng = np.random.default_rng([c0["graine_ordre"], self.epoques_faites])
        ordre = self.xp.entiers(rng.permutation(n)[:M * m])
        for k in range(M):
            idx = ordre[k * m:(k + 1) * m]
            t = self.pas_faits / M
            eta = np.array([eta_planning(None, e, f, c0["echauffement"], c0["epoques"], t, M, c0["planning"])
                            for e, f in zip(self.eta, self.eta_fin)])
            self.pas(self.xp.indexer(X, idx), self.xp.indexer(Y, idx),
                     self.xp.vers(eta), self.xp.vers(self.lmbda), n)
            self.pas_faits += 1
            if rappel and tous_les and (k + 1) % tous_les == 0:
                rappel(self, (k + 1) / M)
        self.epoques_faites += 1
        self.dernier_eta = eta

    # ------------------------------------------------------------ évaluation
    def proba(self, X, morceau=None):
        """Probabilités de sortie (R, N, K), en numpy, par morceaux pour limiter la mémoire."""
        N = X.shape[0]
        taille_ex = max(int(np.prod(c["sortie"])) for c in self.plan)
        morceau = morceau or max(1, min(N, int(2e7 // (self.R * taille_ex))))
        sorties = []
        for d in range(0, N, morceau):
            a, _ = self._avant(X[d:d + morceau], False)
            sorties.append(self.xp.numpy(a))
        return np.concatenate(sorties, axis=1)

    def evaluer(self, X, y):
        """Précision (%) et coût moyen (log-vraisemblance) de chaque réseau : tableaux (R,)."""
        p = self.proba(X)
        y = np.asarray(y)
        precision = 100.0 * (p.argmax(-1) == y[None]).mean(1)
        cout = -np.log(np.take_along_axis(p, y[None, :, None], -1)[..., 0] + 1e-12).mean(1)
        return precision, cout

    # ------------------------------------------------------------ poids
    def poids(self, r):
        """Paramètres du réseau r, en numpy float64, au format de export.py :
        W (n_out, n_in) ou (F, C, k), b (n_out,)."""
        sortie = []
        for c, p in zip(self.plan, self.params):
            if p is None:
                sortie.append(None)
                continue
            W = self.xp.numpy(p["W"][r]).astype(np.float64)
            b = self.xp.numpy(p["b"][r]).astype(np.float64).reshape(-1)
            sortie.append({"W": W, "b": b})
        return sortie

"""Analyses du banc : courbes, frontière précision/coût, confiance, erreurs, graines.

Toutes travaillent sur le jeu de VALIDATION (le test est réservé à essai.precision_test).
"""
import numpy as np
import matplotlib.pyplot as plt

from .config import completer
from .essai import charger_reseau, journal

COULEURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
GRILLE = "#e6e5e0"


def _style(ax):
    ax.grid(True, color=GRILLE, linewidth=0.8)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)


# ---------------------------------------------------------------- courbes d'apprentissage

def courbes(resultats, ax=None, ylim=None):
    """Précision de validation par époque, un trait par essai."""
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    for i, r in enumerate(resultats):
        h = r["historique"]
        ax.plot([e["epoque"] for e in h], [e["validation"] for e in h], "-o", ms=3,
                color=COULEURS[i % len(COULEURS)], label="%.2f %%  %s" % (r["validation"], r["resume"]))
    ax.set_xlabel("époque")
    ax.set_ylabel("précision de validation (%)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7, loc="lower right")
    _style(ax)
    return ax


# ---------------------------------------------------------------- frontière précision / coût

def frontiere_pareto(points):
    """Indices des points non dominés (coût plus bas OU précision plus haute que tous les autres)."""
    ordre = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    garde, meilleur = [], -np.inf
    for i in ordre:
        if points[i][1] > meilleur:
            garde.append(i)
            meilleur = points[i][1]
    return garde


def frontiere(resultats=None, cout="us_esp", ax=None, etiquettes=True, log=None):
    """Chaque essai est un point (coût embarqué, précision de validation). Les points de la
    frontière sont les seuls compromis intéressants : à coût égal, rien ne fait mieux.
    cout : us_esp (temps estimé sur l'ESP), params, flash_ko, macs."""
    resultats = resultats if resultats is not None else journal()
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    pts = [(r["couts"][cout], r["validation"]) for r in resultats]
    dep = [r["couts"]["deployable"] for r in resultats]
    x, y = np.array(pts).T if pts else ([], [])
    ax.scatter([a for a, d in zip(x, dep) if d], [b for b, d in zip(y, dep) if d], s=18,
               color="#9a9890", label="essais")
    ax.scatter([a for a, d in zip(x, dep) if not d], [b for b, d in zip(y, dep) if not d], s=18,
               marker="x", color="#9a9890", label="pas encore déployable")
    f = frontiere_pareto(pts)
    ax.plot([pts[i][0] for i in f], [pts[i][1] for i in f], "-o", color=COULEURS[0], ms=5,
            label="frontière")
    if etiquettes:
        for i in f:
            ax.annotate(resultats[i]["resume"].split(" η")[0], pts[i], fontsize=7,
                        xytext=(4, -10), textcoords="offset points")
    ax.set_xlabel({"us_esp": "temps estimé sur l'ESP (µs)", "params": "nombre de poids",
                   "flash_ko": "poids (Ko, float32)", "macs": "multiplications-additions"}[cout])
    ax.set_ylabel("précision de validation (%)")
    if log is None:                                   # log seulement si les coûts s'étalent
        log = len(pts) > 1 and max(x) > 20 * max(min(x), 1e-9)
    if log:
        ax.set_xscale("log")
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)
    return [resultats[i] for i in f]


# ---------------------------------------------------------------- confiance

def confiance(res, axes=None):
    """À gauche : confiance des bonnes et des mauvaises réponses.
    À droite : si l'on refuse de répondre sous un seuil, quelle part des exemples reste
    traitée (couverture), et avec quelle précision — le critère de SpectraSense."""
    p, cfg, t = charger_reseau(res)
    X, y, _ = t.donnees("validation")
    proba = p.proba(p.xp.vers(X))[0]
    pred, conf = proba.argmax(1), proba.max(1)
    juste = pred == y
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    bins = np.linspace(0, 1, 41)
    axes[0].hist(conf[juste], bins=bins, color=COULEURS[2], alpha=0.8, label="justes (%d)" % juste.sum())
    axes[0].hist(conf[~juste], bins=bins, color=COULEURS[1], alpha=0.8, label="fausses (%d)" % (~juste).sum())
    axes[0].set_yscale("log")
    axes[0].set_xlabel("confiance (probabilité maximale)")
    axes[0].legend(fontsize=8)
    seuils = np.linspace(0, 0.999, 200)
    couverture = np.array([(conf >= s).mean() for s in seuils]) * 100
    prec = np.array([juste[conf >= s].mean() * 100 if (conf >= s).any() else np.nan for s in seuils])
    axes[1].plot(couverture, prec, color=COULEURS[0])
    for s in (0.5, 0.9, 0.99):
        i = np.searchsorted(seuils, s)
        axes[1].plot(couverture[i], prec[i], "o", color=COULEURS[0], ms=4)
        axes[1].annotate("seuil %.2f : %.1f %% traités, %.2f %% justes" % (s, couverture[i], prec[i]),
                         (couverture[i], prec[i]), fontsize=7, xytext=(8, -4), textcoords="offset points")
    axes[1].set_xlabel("couverture : exemples traités (%)")
    axes[1].set_ylabel("précision sur les exemples traités (%)")
    axes[1].invert_xaxis()
    for ax in axes:
        _style(ax)
    return {"precision": float(juste.mean() * 100),
            "erreurs_sures_90": int(((~juste) & (conf > 0.9)).sum())}


# ---------------------------------------------------------------- confusion et erreurs

def confusion(res, ax=None):
    p, cfg, t = charger_reseau(res)
    X, y, _ = t.donnees("validation")
    pred = p.proba(p.xp.vers(X))[0].argmax(1)
    K = len(t.classes)
    M = np.zeros((K, K), dtype=int)
    np.add.at(M, (y, pred), 1)
    ax = ax or plt.subplots(figsize=(5.5, 5))[1]
    hors = M.copy()
    np.fill_diagonal(hors, 0)
    ax.imshow(hors, cmap="Oranges")
    for i in range(K):
        for j in range(K):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7,
                        color="#9a9890" if i == j else "black")
    ax.set_xticks(range(K), t.classes, fontsize=8)
    ax.set_yticks(range(K), t.classes, fontsize=8)
    ax.set_xlabel("prédit")
    ax.set_ylabel("vrai")
    return M


def erreurs(res, n=16, colonnes=8):
    """Les erreurs les plus assurées : celles qu'aucun seuil de confiance ne rattrape."""
    p, cfg, t = charger_reseau(res)
    X, y, _ = t.donnees("validation")
    proba = p.proba(p.xp.vers(X))[0]
    pred, conf = proba.argmax(1), proba.max(1)
    fausses = np.flatnonzero(pred != y)
    fausses = fausses[np.argsort(-conf[fausses])][:n]
    lignes = int(np.ceil(len(fausses) / colonnes)) or 1
    fig, axes = plt.subplots(lignes, colonnes, figsize=(1.6 * colonnes, 1.8 * lignes), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, fausses):
        ax.axis("on")
        t.afficher(ax, X[i], "%s→%s %.0f%%" % (t.classes[y[i]], t.classes[pred[i]], 100 * conf[i]))
    fig.tight_layout()
    return fausses


# ---------------------------------------------------------------- bruit entre graines

def par_graines(resultats=None):
    """Regroupe les essais qui ne diffèrent que par la graine des poids :
    moyenne, écart-type, min, max de la précision de validation."""
    resultats = resultats if resultats is not None else journal()
    groupes = {}
    for r in resultats:
        cle = r["resume"].rsplit(" g=", 1)[0]
        groupes.setdefault(cle, []).append(r["validation"])
    lignes = []
    for cle, v in groupes.items():
        v = np.array(v)
        lignes.append({"config": cle, "n": len(v), "moyenne": v.mean(), "ecart_type": v.std(ddof=1) if len(v) > 1 else np.nan,
                       "min": v.min(), "max": v.max()})
    lignes.sort(key=lambda l: -l["moyenne"])
    print("%-60s %3s %8s %7s %7s %7s" % ("configuration", "n", "moyenne", "σ", "min", "max"))
    for l in lignes:
        print("%-60s %3d %8.2f %7.2f %7.2f %7.2f" % (l["config"][:60], l["n"], l["moyenne"], l["ecart_type"],
                                                    l["min"], l["max"]))
    return lignes

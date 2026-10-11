"""Lancer un essai, l'enregistrer dans le journal, le recharger.

Journal : banc/resultats/journal.jsonl, une ligne par essai (le dernier essai d'un
identifiant fait foi). Poids : banc/resultats/modeles/<id>.npz. Les deux sont
ignorés par git : ce sont des résultats, régénérables à partir des configurations.
"""
import datetime
import json
import os
import time

import numpy as np

from .config import completer, identifiant, resume
from .couts import couts
from .moteur import Paquet
from .taches import tache

DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats")
JOURNAL = os.path.join(DOSSIER, "journal.jsonl")


# ---------------------------------------------------------------- données au format du moteur

def preparer(paquet, cfg, parties=("train", "validation")):
    t = tache(cfg["tache"], cfg["tache_options"])
    sortie = {}
    for partie in parties:
        X, y, _ = t.donnees(partie)
        if partie == "train" and cfg["n_images"]:
            X, y = X[:cfg["n_images"]], y[:cfg["n_images"]]
        sortie[partie] = (paquet.xp.vers(X), paquet.xp.vers(np.eye(len(t.classes))[y]), y)
    return t, sortie


# ---------------------------------------------------------------- un essai (interactif)

def lancer(cfg=None, moteur="numpy", dispositif="cpu", rappel=None, sauver=True, **modifs):
    """Entraîne UN réseau. rappel(info) est appelé à chaque époque avec :
    {"epoque", "historique", "paquet", "duree_calcul"} ; il peut afficher des courbes."""
    cfg = completer(cfg, **modifs)
    t = tache(cfg["tache"], cfg["tache_options"])
    p = Paquet([cfg], t.forme_entree, len(t.classes), moteur, dispositif)
    _, d = preparer(p, cfg)
    Xtr, Ytr, _ = d["train"]
    Xva, _, yva = d["validation"]
    historique, calcul = [], 0.0
    for e in range(cfg["epoques"]):
        t0 = time.perf_counter()
        p.epoque(Xtr, Ytr)
        calcul += time.perf_counter() - t0
        prec, cout = p.evaluer(Xva, yva)
        historique.append({"epoque": e + 1, "validation": round(float(prec[0]), 3),
                           "cout": round(float(cout[0]), 5), "eta": float(p.dernier_eta[0])})
        if rappel:
            rappel({"epoque": e + 1, "historique": historique, "paquet": p, "duree_calcul": calcul})
    res = _resultat(cfg, historique, calcul, moteur)
    if sauver:
        enregistrer(res, p.poids(0))
    res["_paquet"] = p
    return res


def _resultat(cfg, historique, duree, moteur):
    return {
        "id": identifiant(cfg),
        "resume": resume(cfg),
        "config": cfg,
        "validation": historique[-1]["validation"],
        "meilleure_validation": max(h["validation"] for h in historique),
        "historique": historique,
        "couts": couts(cfg),
        "duree_s": round(duree, 2),
        "moteur": moteur,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- journal

def enregistrer(res, poids):
    os.makedirs(os.path.join(DOSSIER, "modeles"), exist_ok=True)
    tableaux = {}
    for i, w in enumerate(poids):
        if w is not None:
            tableaux["W%d" % i], tableaux["b%d" % i] = w["W"], w["b"]
    np.savez_compressed(os.path.join(DOSSIER, "modeles", res["id"] + ".npz"),
                        config=json.dumps(res["config"]), **tableaux)
    ligne = {k: v for k, v in res.items() if not k.startswith("_")}
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def journal(filtre=None):
    """Tous les essais (le dernier de chaque identifiant), éventuellement filtrés :
    filtre(res) -> bool, ou dict de valeurs de configuration attendues."""
    if not os.path.exists(JOURNAL):
        return []
    derniers = {}
    with open(JOURNAL, encoding="utf-8") as f:
        for ligne in f:
            if ligne.strip():
                r = json.loads(ligne)
                derniers[r["id"]] = r
    res = list(derniers.values())
    if isinstance(filtre, dict):
        res = [r for r in res if all(r["config"].get(k) == v for k, v in filtre.items())]
    elif callable(filtre):
        res = [r for r in res if filtre(r)]
    return res


def existe(cfg):
    return os.path.exists(os.path.join(DOSSIER, "modeles", identifiant(cfg) + ".npz"))


def charger_reseau(res_ou_id, moteur="numpy"):
    """Paquet d'un réseau (R = 1) avec ses poids entraînés, pour les analyses."""
    id_ = res_ou_id if isinstance(res_ou_id, str) else res_ou_id["id"]
    z = np.load(os.path.join(DOSSIER, "modeles", id_ + ".npz"))
    cfg = json.loads(str(z["config"]))
    t = tache(cfg["tache"], cfg["tache_options"])
    p = Paquet([cfg], t.forme_entree, len(t.classes), moteur)
    for i, q in enumerate(p.params):
        if q is not None:
            q["W"] = p.xp.vers(z["W%d" % i][None])
            b = z["b%d" % i]
            q["b"] = p.xp.vers(b[None, None, :] if q["b"].ndim == 3 else b[None, None, :, None])
    return p, cfg, t


def precision_test(res_ou_id):
    """LA mesure sur le jeu de test, pour le réseau retenu seulement (elle est notée
    dans le journal : un réseau choisi d'après le test aurait une précision trop optimiste)."""
    p, cfg, t = charger_reseau(res_ou_id)
    X, y, _ = t.donnees("test")
    prec, _ = p.evaluer(p.xp.vers(X), y)
    res = [r for r in journal() if r["id"] == identifiant(cfg)][0]
    res["test"] = round(float(prec[0]), 3)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    return res["test"]

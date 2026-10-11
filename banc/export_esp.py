"""Pont vers l'ESP : un essai du journal -> esp32/modeles/<nom>.npz, au format de train.py.

    from banc.export_esp import vers_esp
    chemin = vers_esp("3f2a9c1e5b")          # identifiant du journal
    # puis : python esp32\\python\\export.py <chemin>   (headers C++ + autocontrôles)

La précision sur le jeu de test est mesurée à ce moment-là (une seule fois, pour le
réseau retenu) et inscrite dans le modèle, comme le faisait train.py.
"""
import datetime
import json
import os

import numpy as np

from .config import resume
from .couts import couts
from .essai import charger_reseau, journal, precision_test
from .taches import RACINE


def vers_esp(res_ou_id, nom=None):
    id_ = res_ou_id if isinstance(res_ou_id, str) else res_ou_id["id"]
    p, cfg, t = charger_reseau(id_)
    c = couts(cfg)
    if not c["deployable"]:
        raise ValueError("architecture pas encore exécutable par le moteur C++ de l'ESP : " + resume(cfg))
    poids = p.poids(0)
    tailles = [int(np.prod(t.forme_entree))] + [w["W"].shape[0] for w in poids]
    activations = [couche.get("act", "sigmoide") for couche in cfg["couches"]] + ["softmax"]
    res = [r for r in journal() if r["id"] == id_][0]
    test = res.get("test", None)
    if test is None:
        test = precision_test(id_)
    nom = nom or "%s_%s_%s" % (cfg["tache"], "_".join(str(n) for n in tailles), id_)
    meta = {
        "nom": nom,
        "tailles": tailles,
        "activations": activations,
        "pretraitement": t.pretraitement_esp(),
        "precision_test": test,
        "precision_validation": res["validation"],
        "hyperparametres": cfg,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "historique": [(h["epoque"], h["validation"], 0.0) for h in res["historique"]],
        "banc_id": id_,
    }
    tableaux = {}
    for i, w in enumerate(poids):
        tableaux["W%d" % i], tableaux["b%d" % i] = w["W"], w["b"]
    chemin = os.path.join(RACINE, "esp32", "modeles", nom + ".npz")
    np.savez(chemin, meta=json.dumps(meta, ensure_ascii=False), **tableaux)
    print("Modèle écrit : %s (validation %.2f %%, test %.2f %%, ESP estimé %.0f µs)"
          % (os.path.relpath(chemin, RACINE), res["validation"], test, c["us_esp"]))
    print("Ensuite : python esp32\\python\\export.py %s" % os.path.relpath(chemin, RACINE))
    return chemin

"""Configuration d'un essai : un dictionnaire qui décrit tout, et rien d'autre.

Deux essais de même configuration donnent le même réseau, quel que soit le
moteur (numpy en interactif, PyTorch en lot) : mêmes poids de départ (graine),
même ordre des images (graine_ordre), mêmes formules.
"""
import copy
import hashlib
import json

DEFAUT = {
    # --- données
    "tache": "mnist",                 # mnist | signal | spectrasense
    "tache_options": {},              # ex. {"facteur": 2} pour mnist
    "n_images": None,                 # sous-ensemble d'entraînement (None : tout)
    # --- réseau : couches cachées ; la couche de sortie softmax est ajoutée d'office
    "couches": [{"type": "dense", "n": 30, "act": "sigmoide"}],
    "init": "auto",                   # auto : 1/sqrt(n) (sigmoïde, tanh), sqrt(2/n) (ReLU)
    # --- apprentissage
    "epoques": 30,
    "batch": 10,
    "eta": 0.5,                       # η maximal du planning
    "planning": "cosinus",            # cosinus | constant
    "echauffement": 1.0,              # cosinus : époques de montée linéaire
    "eta_fin": 0.0,                   # cosinus : η à la fin
    "lmbda": 5.0,                     # régularisation L2 (poids seulement, comme network2)
    "graine": 1,                      # poids de départ
    "graine_ordre": 0,                # ordre des images (partagé dans un paquet du mode lot)
    "precision": "float32",           # float32 | float64
    "note": "",                       # texte libre, ne change pas l'identifiant
}

# options par défaut de chaque tâche : une option omise et sa valeur par défaut
# donnent la même configuration, donc le même identifiant
OPTIONS_TACHES = {
    "mnist": {"facteur": 2},
    "signal": {"longueur": 128, "bruit": 0.15, "n_train": 8000},
    "spectrasense": {},
}
TYPES = {"dense", "conv1d", "maxpool1d"}
ACTIVATIONS = {"sigmoide", "relu", "tanh"}
# ce qui doit être identique entre les réseaux d'un même paquet (mode lot)
CLES_PAQUET = ("tache", "tache_options", "n_images", "couches", "init", "epoques", "batch",
               "planning", "echauffement", "graine_ordre", "precision")


def completer(cfg=None, **modifs):
    """Configuration complète et vérifiée, à partir des valeurs par défaut."""
    c = copy.deepcopy(DEFAUT)
    for source in (cfg or {}), modifs:
        for k, v in source.items():
            if k not in DEFAUT:
                raise KeyError("paramètre inconnu : %r (connus : %s)" % (k, ", ".join(DEFAUT)))
            c[k] = copy.deepcopy(v)
    if c["tache"] not in OPTIONS_TACHES:
        raise ValueError("tâche inconnue : %r" % c["tache"])
    c["tache_options"] = {**OPTIONS_TACHES[c["tache"]], **c["tache_options"]}
    if not isinstance(c["couches"], list) or not all(isinstance(x, dict) for x in c["couches"]):
        raise ValueError("couches : une liste de couches, ex. [{\"type\": \"dense\", \"n\": 30, "
                         "\"act\": \"relu\"}] ; reçu %r (dans une grille du lot : une liste de ces "
                         "listes, une par architecture)" % (c["couches"],))
    for couche in c["couches"]:
        if couche.get("type") not in TYPES:
            raise ValueError("type de couche inconnu : %r" % couche)
        if couche["type"] in ("dense", "conv1d") and couche.get("act", "sigmoide") not in ACTIVATIONS:
            raise ValueError("activation inconnue : %r" % couche)
    if c["planning"] not in ("cosinus", "constant"):
        raise ValueError("planning : cosinus ou constant")
    if c["precision"] not in ("float32", "float64"):
        raise ValueError("precision : float32 ou float64")
    return c


def identifiant(cfg):
    """Empreinte courte de la configuration (la note n'en fait pas partie)."""
    c = {k: v for k, v in completer(cfg).items() if k != "note"}
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:10]


def cle_paquet(cfg):
    c = completer(cfg)
    return json.dumps({k: c[k] for k in CLES_PAQUET}, sort_keys=True)


def resume(cfg):
    """Description lisible d'une ligne."""
    c = completer(cfg)
    morceaux = []
    for couche in c["couches"]:
        if couche["type"] == "dense":
            morceaux.append("D%d%s" % (couche["n"], _abr(couche)))
        elif couche["type"] == "conv1d":
            morceaux.append("C%dk%d%s%s" % (couche["canaux"], couche["noyau"],
                                            "s%d" % couche["pas"] if couche.get("pas", 1) > 1 else "",
                                            _abr(couche)))
        else:
            morceaux.append("P%d" % couche["taille"])
    opts = ",".join("%s=%s" % kv for kv in sorted(c["tache_options"].items()))
    return "%s%s [%s] η=%g λ=%g m=%d %dép g=%d" % (
        c["tache"], "(%s)" % opts if opts else "", "-".join(morceaux), c["eta"], c["lmbda"],
        c["batch"], c["epoques"], c["graine"])


def _abr(couche):
    return {"sigmoide": "s", "relu": "r", "tanh": "t"}[couche.get("act", "sigmoide")]

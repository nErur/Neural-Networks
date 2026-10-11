"""Coût embarqué d'une architecture sur l'ESP32 : mémoire et temps d'inférence estimé.

Coûts unitaires mesurés sur la carte cnn-test (esp32/ESSAIS.md, section 4), à
240 MHz, poids en RAM, moteur compilé en -O2. Les valeurs marquées « estimé »
n'ont pas encore été mesurées : à remplacer après un profilage.
"""
from .moteur import plan_couches
from .taches import tache

FREQUENCE = 240e6
CYCLES = {
    "mac": 5.3,          # multiplication-addition (mesuré : 5,2 à 5,5)
    "sigmoide": 436,     # expf + division (mesuré)
    "softmax": 453,      # par neurone, sortie de 10 (mesuré : 453 à 532)
    "tanh": 450,         # estimé (tanhf ≈ expf + division)
    "relu": 46,          # mesuré sur 50 neurones, appel compris (estimé à 4 avant mesure)
    "max": 3,            # estimé, par comparaison du max-pooling
    "fixe": 8000,        # mesuré : appel, recherche du maximum, chronométrage (7 150 à 9 000, ~33 µs)
}
# ce que le moteur C++ de l'ESP sait exécuter aujourd'hui
DEPLOYABLE = {"types": {"dense"}, "activations": {"sigmoide", "relu", "tanh", "softmax"}}


def couts(cfg, n_classes=None):
    """Dictionnaire des coûts d'une configuration (une ligne du journal)."""
    t = tache(cfg["tache"], cfg["tache_options"])
    plan = plan_couches(cfg, t.forme_entree, n_classes or len(t.classes))
    params = macs = 0
    cycles = CYCLES["fixe"] + t.cycles_pretraitement()
    largeur = int(_taille(t.forme_entree))
    deployable = t.pretraitement_esp() is not None
    detail = []
    for c in plan:
        n_sortie = int(_taille(c["sortie"]))
        largeur = max(largeur, n_sortie)
        if c["type"] == "dense":
            p, mac = c["n"] * c["n_in"] + c["n"], c["n"] * c["n_in"]
        elif c["type"] == "conv1d":
            C, k = c["entree"][0], c["noyau"]
            p = c["canaux"] * C * k + c["canaux"]
            mac = n_sortie * C * k
        else:
            p, mac = 0, 0
        cyc = mac * CYCLES["mac"]
        if c["type"] == "maxpool1d":
            cyc += n_sortie * (c["taille"] - 1) * CYCLES["max"]
        else:
            cyc += n_sortie * CYCLES[c["act"]]
        params += p
        macs += mac
        cycles += cyc
        deployable &= c["type"] in DEPLOYABLE["types"] and c.get("act", "softmax") in DEPLOYABLE["activations"]
        detail.append({"couche": c["type"], "params": p, "macs": mac, "us": cyc / FREQUENCE * 1e6})
    return {
        "params": params,
        "flash_ko": params * 4 / 1024,                       # float32
        "ram_ko": (params + 2 * largeur + int(_taille(t.forme_entree))) * 4 / 1024,   # poids en RAM + tampons
        "macs": macs,
        "us_esp": cycles / FREQUENCE * 1e6,
        "deployable": bool(deployable),
        "detail": detail,
    }


def _taille(forme):
    n = 1
    for d in forme:
        n *= d
    return n

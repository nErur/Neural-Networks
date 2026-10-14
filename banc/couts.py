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
    "softmax": 453,      # par neurone, sortie de 10 (mesuré : 453 à 532 ; 650 pour 12 sorties le 14/10)
    "tanh": 450,         # estimé (tanhf ≈ expf + division)
    "relu": 46,          # mesuré sur 50 neurones, appel compris (estimé à 4 avant mesure) ;
                         # sur les 1024 sorties d'une convolution : 5,4 par neurone (14/10)
    "max": 20,           # mesuré (14/10, SpectraSense) : 20 à 22 par comparaison, boucle comprise
    # convolution 1D (mesuré le 14/10 sur 162dcb9fbf, poids en RAM) : la boucle intérieure ne fait que
    # k tours ; son entrée/sortie coûte ~15 cycles par (filtre, position, canal) et le biais / la
    # sortie ~19 par (filtre, position), en plus des 5,3 par multiplication-addition.
    # Conv k5 sur 8 et 16 canaux : 8,5 à 8,7 cycles par mult.-add. (modèle : 8,6 et 8,4).
    "conv_canal": 15,
    "conv_position": 19,
    "fixe": 8000,        # mesuré : appel, recherche du maximum, chronométrage (7 150 à 9 000, ~33 µs)
}
# ce que le moteur C++ de l'ESP sait exécuter aujourd'hui
# (conv1d et maxpool1d : testés sur PC et sur la carte le 14/10/2026, 16/16 fenêtres, 3,4 ms)
DEPLOYABLE = {"types": {"dense", "conv1d", "maxpool1d"}, "activations": {"sigmoide", "relu", "tanh", "softmax"}}


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
        if c["type"] == "conv1d":
            F, Lo = c["sortie"]
            cyc += F * Lo * (c["entree"][0] * CYCLES["conv_canal"] + CYCLES["conv_position"])
        if c["type"] == "maxpool1d":
            cyc += n_sortie * (c["taille"] - 1) * CYCLES["max"]
        elif c["type"] == "conv1d" and c["act"] == "relu":
            cyc += n_sortie * 5.4
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

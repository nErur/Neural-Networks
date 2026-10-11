"""Données MNIST en uint8 et prétraitement de référence (la « spécification »).

Tout ce qui touche aux pixels passe par ce fichier : train.py, export.py et,
plus tard, le C++ de l'ESP doivent produire EXACTEMENT les mêmes entrées.

Spécification du prétraitement (facteur f, côté de sortie 28 // f) :
    1. image brute 28×28 en uint8 (0..255), ligne par ligne ;
    2. somme ENTIÈRE de chaque bloc f×f ;
    3. une seule division en float32 : somme / (255 · f²) ;
    4. aplatissement ligne par ligne.
Les étapes 2 et 3 sont exactes et déterministes en float32 : le C
(`(float)somme / 1020.0f` pour f = 2) donne le même résultat au bit près.
"""
import gzip
import os
import pickle

import numpy as np

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FICHIER_MNIST = os.path.join(RACINE, "nielsen", "data", "mnist.pkl.gz")
COTE = 28


def charger_u8():
    """Renvoie {"train", "validation", "test"} -> (images uint8 (N, 28, 28), étiquettes uint8).

    Le fichier de Nielsen stocke chaque pixel sous la forme octet / 256 (maximum
    0,99609375) : la reconversion octet = valeur × 256 est exacte, on le vérifie.
    """
    with gzip.open(FICHIER_MNIST, "rb") as f:
        parties = pickle.load(f, encoding="latin1")
    sortie = {}
    for nom, (x, y) in zip(("train", "validation", "test"), parties):
        u = x.astype(np.float64) * 256.0
        if np.abs(u - np.round(u)).max() != 0.0 or u.max() > 255:
            raise ValueError("MNIST : pixels inattendus (diviseur 256 attendu)")
        sortie[nom] = (np.round(u).astype(np.uint8).reshape(-1, COTE, COTE),
                       y.astype(np.uint8))
    return sortie


def pretraiter(images_u8, facteur):
    """(N, 28, 28) uint8 -> (N, (28/f)²) float32, selon la spécification ci-dessus."""
    if COTE % facteur:
        raise ValueError("le facteur doit diviser 28")
    c = COTE // facteur
    sommes = (images_u8.astype(np.int32)
              .reshape(-1, c, facteur, c, facteur).sum(axis=(2, 4)))
    diviseur = np.float32(255 * facteur * facteur)
    return (sommes.astype(np.float32) / diviseur).reshape(len(images_u8), c * c)

# Neural-Networks

Le labo d'Antoine pour apprendre les réseaux de neurones, et la partie embarquée de SpectraSense
(un réseau convolutif qui reconnaît les appareils de l'arrière-cuisine sur un ESP32).
Avant de modifier quoi que ce soit, les agents lisent `CLAUDE.md`.

| Dossier | Contenu |
|---|---|
| `nielsen/` | code du livre *Neural Networks and Deep Learning* (Nielsen), porté en Python 3 — historique |
| `experiences/` | notebooks d'apprentissage : essais MNIST (`banc_essai*.ipynb`), poids (`poids.ipynb`), convolution expliquée pas à pas (`convolution_pas_a_pas.ipynb`) |
| `banc/` | banc d'essai : entraîner, comparer et choisir des réseaux pour l'ESP32 — `banc/README.md` |
| `esp32/` | moteur d'inférence C++, composants ESPHome `nn_mnist` et `nn_spectrasense`, export, tests — `esp32/README.md` |
| `outils/`, `mesures_moteur.py`, `verif_gpu.py` | mesures qui ont servi à choisir le moteur du banc (historique) |

Dépôt voisin : `spectrasense` (préparation des données, évaluation, chaîne Python de référence).

## Activer l'environnement (PowerShell)

    cd C:\AItools\Projets\Neural-Networks
    .\.venv\Scripts\Activate.ps1

(Recréer : `py -3.12 -m venv .venv` puis `pip install -r requirements.txt`.)

## Test de réception

    python -W ignore test_reception.py 1     # une époque (~3 s)
    python -W ignore test_reception.py       # réseau complet

Au départ : réseau du chapitre 1 ([784, 30, 10], 30 époques, mini-batch 10, η = 3.0 ; ~80 s), résultat
9479–9503 / 10000 (≈ 95 %). Le script tourne maintenant avec [784, 15, 10].
`-W ignore` masque un SyntaxWarning sur une docstring de `network.py` (`\partial`), laissée telle quelle.

## Versions

- Python 3.12.10 (64 bits), venv + pip (uv non installé)
- numpy 2.5.3, matplotlib 3.11.2, ipykernel 7.4.0
- torch 2.14.1+cu126 (CUDA 12.6), torchvision 0.29.1+cu126 — GPU NVIDIA TITAN Xp (sm_61, supporté par cu126)
- Theano volontairement absent (network3.py non portable)

## Code de Nielsen

`nielsen/` = clone de mnielsen/neural-networks-and-deep-learning (72db53d, Python 2.7).
Portage Python 3 de `network.py`, `network2.py`, `mnist_loader.py` : commit séparé.
Comparer avec l'original : `git diff d3a694a da0a789 -- nielsen/src`.

## ESP32

`esp32/` : d'abord un petit réseau MNIST sur ESP32 via ESPHome (plan historique : `esp32/PLAN.md`),
puis le réseau SpectraSense, qui tourne en direct sur la carte de test `cnn-test`. Mesures et décisions :
`esp32/ESSAIS.md`.

## Banc d'essai

`banc/` : entraîner, comparer et choisir des réseaux pour l'ESP32 (MNIST, puis SpectraSense).
Notebook : `banc/banc_esp.ipynb`. Voir `banc/README.md`.

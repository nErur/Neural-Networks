# Neural Networks and Deep Learning (Nielsen) — environnement

## Activer l'environnement (PowerShell)

    cd C:\AItools\Projets\Neural-Networks
    .\.venv\Scripts\Activate.ps1

(Recréer : `py -3.12 -m venv .venv` puis `pip install -r requirements.txt`.)

## Test de réception

Essai d'une époque (~3 s) puis réseau du chapitre 1 ([784, 30, 10], 30 époques,
mini-batch 10, eta = 3.0 ; ~80 s, ~95 % attendu) :

    python -W ignore test_reception.py 1
    python -W ignore test_reception.py

Résultat obtenu : 9479–9503 / 10000 sur les dernières époques (≈ 95 %).
`-W ignore` masque un SyntaxWarning sur une docstring de `network.py` (`\partial`), laissée telle quelle.

## Versions

- Python 3.12.10 (64 bits), venv + pip (uv non installé)
- numpy 2.5.3, matplotlib 3.11.2, ipykernel 7.4.0
- torch 2.14.1+cu126 (CUDA 12.6), torchvision 0.29.1+cu126 — GPU NVIDIA TITAN Xp détecté (sm_61, supporté par cu126)
- Theano volontairement absent (network3.py non portable)

## Code de Nielsen

`nielsen/` = clone de mnielsen/neural-networks-and-deep-learning (72db53d, Python 2.7).
Portage Python 3 de `network.py`, `network2.py`, `mnist_loader.py` : commit séparé.
Comparer avec l'original : `git diff d3a694a da0a789 -- nielsen/src`.

## ESP32

`esp32/` = déploiement d'un petit réseau MNIST sur ESP32 via ESPHome, banc d'essai du pipeline SpectraSense. Plan : `esp32/PLAN.md`.

## Banc d'essai

`banc/` = entraîner, comparer et choisir des réseaux pour l'ESP32 (MNIST, puis SpectraSense). Notebook : `banc/banc_esp.ipynb`. Voir `banc/README.md`.

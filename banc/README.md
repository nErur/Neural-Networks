# Banc d'essai — réseaux pour l'ESP32

Entraîner, comparer et choisir des réseaux **sous les contraintes de l'ESP32**, d'abord sur MNIST,
ensuite pour SpectraSense. Choix et mesures qui justifient sa conception : `esp32/ESSAIS.md`, section 5.

## Démarrer

Ouvrir `banc/banc_esp.ipynb` dans VS Code (venv du dépôt) et suivre les cellules :
paramètres → préparation → essai interactif → analyses → comparaison → lot → export vers l'ESP.

## Principes

- **Un essai = une configuration** (`config.py`) : tâche, couches, η, λ, planning, graines.
  Elle a un identifiant ; même configuration = même réseau, quel que soit le moteur.
- **Un seul moteur** (`moteur.py`), écrit une fois pour numpy et PyTorch. Il entraîne R réseaux de même
  architecture ensemble. Interactif : numpy, R = 1, processeur. Lot : PyTorch, R jusqu'à ~512, carte graphique.
  Rétropropagation écrite à la main, formules de `network2` (L2 sur les poids seulement).
- **Journal** (`banc/resultats/`, ignoré par git) : une ligne par essai et ses poids. Les essais en lot
  et en interactif y sont rangés de la même façon.
- **On choisit sur la validation.** Le test n'est mesuré qu'une fois, pour le réseau retenu
  (`essai.precision_test`, appelé par l'export).
- **Coût embarqué** (`couts.py`) : poids, mémoire, multiplications-additions et temps estimé sur l'ESP,
  calibrés sur le profilage de la carte (348 µs estimés contre 344 µs mesurés pour [196, 30, 10]).

## Fichiers

| Fichier | Rôle |
|---|---|
| `config.py` | configuration par défaut, vérification, identifiant, résumé |
| `taches.py` | MNIST (réduction f×f, même spécification que l'ESP), signal 1D synthétique, gabarit SpectraSense, découpage par groupe |
| `moteur.py` | couches dense, conv1d, maxpool1d ; sigmoïde, ReLU, tanh ; softmax ; planning cosinus |
| `essai.py` | essai interactif, journal, rechargement d'un réseau, mesure du test |
| `lot.py` | grille, tirage aléatoire, graines multiples ; entraînement par paquets |
| `couts.py` | coût estimé sur l'ESP32 (coûts unitaires mesurés) |
| `analyses.py` | courbes, frontière précision/coût, confiance et couverture, confusion, erreurs, bruit entre graines |
| `export_esp.py` | essai du journal → `esp32/modeles/*.npz` → `esp32/python/export.py` |
| `tests/test_equivalence.py` | la chaîne de tests ci-dessous |

## Tests d'équivalence

    python -m banc.tests.test_equivalence

1. numpy float64 = `network2` de Nielsen, 300 mises à jour (écart mesuré 2e-15) ;
2. gradients dense / conv1d / maxpool = différences finies (écart relatif ≤ 6e-8) ;
3. un paquet de 3 réseaux = 3 réseaux séparés (écart nul) ;
4. PyTorch processeur = numpy (1e-15 en float64, 1e-6 en float32) ;
5. PyTorch carte graphique (TITAN Xp) = numpy float32 (mesuré : 1,2e-6 en dense, 8e-7 en conv1d).

À relancer après toute modification du moteur.

## Pour SpectraSense

- Tâche `spectrasense` (`taches.SpectraSense`) : fenêtres de la ligne arrière-cuisine (ESP + PZEM),
  forme (canaux, longueur), **12 classes** (repos, 8 phases de la machine à café, grille-pain,
  bouilloire, consommateur inconnu ; une seule chose tourne à la fois), seulement les secondes
  « porte ouverte ». Options : `longueur`, `canaux`, `copies`, `synth`, `equilibrage`.
- **Les valeurs par défaut de `config.py` (128, synth 0.10, racine) ne sont pas celles du modèle déployé.**
  Modèle déployé `162dcb9fbf` : `longueur` 130, `canaux` "log+lin", `copies` 2, `synth` 0.22,
  `equilibrage` "aucun", ReLU, `init_biais` "zero" (jeu `L130_log+lin_c2_s22_aucun`).
  Copie versionnée des poids : `esp32/modeles/spectrasense_162dcb9fbf.npz`.
- Les fenêtres sont fabriquées côté SpectraSense (dépôt voisin, ou variable `SPECTRASENSE`) :
  `python tools/dataset/fenetres.py --longueur 130 --canaux log,lin --synth 0.22 --equilibrage aucun` ; le banc ne fait que les lire.
- La précision par fenêtre sert au tri ; le vrai critère (latence, fausses alertes / semaine) :
  `python tools/evaluation/evaluer.py --source reseau --modele <banc/resultats/modeles/ID.npz>`
  (dans le dépôt spectrasense, **sur la validation**).
- Les convolutions 1D sont déployables : `esp32/python/export_conv.py <id>` (voir `esp32/README.md`).
  `export_esp.py` ne gère que les réseaux denses MNIST.
- `init_biais` : "normal" (N(0, 1), comme network2) ou "zero". Avec ReLU, préférer "zero" ; même ainsi,
  le modèle déployé a 7 filtres morts (sortie toujours nulle, `esp32/ESSAIS.md` section 8.3).
- Préférer **ReLU** dans les couches cachées : environ 46 cycles contre ~436 pour une sigmoïde sur l'ESP.
- Coûts recalés sur le profilage de `162dcb9fbf` (3,4 ms mesurés sur la carte).

## Limites connues

- Lot : pas encore de CUDA graph (mesuré ×2 à ×2,6 pour des paquets ≤ 256 réseaux).
- Une seule tête de sortie (softmax) ; plusieurs têtes (un appareil chacune) viendront avec SpectraSense.
- Les n mod m dernières images d'une époque sont sautées (mini-batches complets seulement).

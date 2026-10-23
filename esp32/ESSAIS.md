# Journal des essais et des choix — MNIST sur ESP32

Ce fichier garde la trace des mesures qui ont fixé nos choix, pour ne pas se reposer les mêmes
questions. Chaque essai : la question, la prédiction faite **avant**, la mesure, la décision.
Plan général : `PLAN.md`. Mode d'emploi : `README.md`.

Dates : 10 et 11 octobre 2026. Matériel de référence en fin de fichier.

---

## 1. Chaîne de déploiement

| Choix | Pourquoi | Alternative écartée |
|---|---|---|
| Carte gérée par l'**ESPHome de Home Assistant** | Mises à jour d'ESPHome avec HA, une seule interface, un seul `secrets.yaml` | CLI ESPHome sur le PC : gardée en secours (`cnn-test-local.yaml`) |
| Composant `nn_mnist` lu par HA **depuis GitHub** (`external_components`, `path: esp32/esphome/components`, `refresh: 0s`) | Chaque firmware correspond à un commit ; plus tard un tag figé pour la carte PZEM | Copie à la main par le File editor : pas de traçabilité, poids à recopier à chaque entraînement |
| Accès GitHub de HA : **jeton fine-grained en lecture seule**, ce dépôt seulement | ESPHome stocke l'identifiant en clair ; ESPHome ne gère pas les clés SSH | Deploy key SSH |
| Push depuis le PC : **Git Credential Manager** (connexion par le navigateur) | Aucune clé à gérer | Mot de passe GitHub : refusé par GitHub depuis 2021 |
| Framework **ESP-IDF**, choisi dès le premier flash USB | Recommandé par ESPHome ; changer de framework par OTA peut poser des problèmes de partitions | Arduino |
| Secrets (`!secret`) aussi dans `cnn-test.yaml` | La config de HA est identique, ligne pour ligne, à celle du dépôt, sans secret commité | Clés recopiées en clair dans le YAML |

**Pièges rencontrés**
- HA compile le **dernier commit poussé** : un changement non poussé n'arrive pas sur la carte (« v1 » au lieu de « v2 »).
- Nouvelles entités invisibles dans HA : *Intégration ESPHome → ⋮ → Recharger*.
- **Conflit de noms** : ESPHome 2026.8 inclut tous les `.h` du composant dans `main.cpp`, avec `using namespace esphome`.
  Le modèle généré ne doit donc pas s'appeler `nn_mnist` → espace de noms **`nn_model`**.
  Mon contrôle hors ESP ne compilait pas `main.cpp` ; il reproduit désormais son schéma.
- Une boucle qui garde la main trop longtemps déclenche un avertissement d'ESPHome (« took a long time ») :
  autotest et profilage traitent **une image par passage dans `loop()`**.

---

## 2. Données et prétraitement

**Diviseur des images de Nielsen.** Prédiction : octet / 256 et non / 255. Mesure : maximum 0,99609375,
reconversion ×256 exacte sur les 70 000 images. → On repart d'entiers `uint8` (`donnees.py`), comme sur l'ESP.

**Spécification du prétraitement** (`donnees.py`, `nn_engine.cpp`) : somme **entière** de chaque bloc f×f,
puis **une seule division** float32 par 255·f². → Entrée identique **au bit près** entre numpy, le C sur PC
et l'ESP (24/24 à chaque fois). C'est le seul étage où l'on exige l'égalité exacte.

**Résolution d'entrée** — réseau [n, 30, 10], 30 époques, η = 0,5 cosinus, λ = 5, m = 10, graine 1.

| Entrée | Poids | Test | Prédiction |
|---|---|---|---|
| 28×28 | 23 860 | 97,09 % | — |
| **14×14** | **6 220** | **96,86 %** | perte < 1 point ✓ (−0,23) |
| 7×7 | 1 810 | 94,48 % | perte de quelques points ✓ (−2,6) |

→ **14×14 retenu** : 4 fois moins de poids pour une perte inférieure au bruit d'un essai (≈ 0,5 point).

**Chiffre manuscrit (photo d'un 7, 220×207)** : le cadrage compte plus que tout.

| Préparation | Probabilité du 7 |
|---|---|
| Réduction directe en 28×28 | 31,8 % (3 : 31,2 %, 2 : 30,0 %) |
| Même chose, contraste étiré à 255 | 31,8 % (le contraste n'y est pour rien) |
| Cadrage MNIST : boîte 20×20, centre de masse au centre | **98,7 %** |
| Cadrage MNIST décalé de 1 / 2 / 3 px vers la gauche | 94,5 / 54,4 / 16,0 % (prédit 2) |

→ `send.py --png` cadre comme MNIST par défaut. Un perceptron ne sait pas qu'un chiffre décalé reste
le même chiffre : 3 pixels suffisent à le tromper. C'est ce que corrigent les convolutions (CNN).

---

## 3. Exactitude, du PC jusqu'à l'ESP

| Contrôle | Prédiction | Mesure |
|---|---|---|
| Prédictions float32 vs float64 (10 000 images) | 0 différence | 0 ✓ |
| Disposition des poids `W[j*n_in + k]` (boucle Python vs numpy) | ≈ 1e-7 | 3e-7 ✓ |
| Moteur C++ sur PC, 24 images : entrée / couche 0 / sortie | 0 / ≈1e-7 / ≈1e-7 | 0 / 3,0e-7 / 2,4e-7 ✓ |
| Prédiction C = prédiction Python (y compris ses 2 erreurs) | 24/24 | 24/24 ✓ |
| Test du test : poids transposés, ÷256, biais oublié | détectés au bon étage | 3/3 ✓ |
| **Sur l'ESP** (autotest) | 24/24, ≈ 3e-7 | **24/24, 3,0e-7** ✓ |
| **Images envoyées depuis le PC** (`send.py`, action API `classify`) | même prédiction que Python | **10/10**, mêmes confiances au 0,1 % près ✓ |

→ Tolérance de 1e-5 sur les couches : elle laisse de la marge aux différences d'ordre des additions
et à l'instruction `madd.s` (multiplication et addition fusionnées) de l'ESP.

Rejouer : `python esp32\python\export.py` (contrôles 1 et 2), `python esp32\tests\lancer_tests.py`.

---

## 4. Performance de l'inférence sur l'ESP — [196, 30, 10]

### 4.1 Temps total

| Configuration | Prédiction | Mesure |
|---|---|---|
| Poids en flash (`const`, par défaut) | 200–400 µs | **3,4–3,9 ms** (image seule), 4,8 ms (autotest, logs en parallèle) ✗ |
| Poids copiés en RAM (`weights_in_ram: true`) | 200–400 µs | **0,80 ms** (image seule), 0,88 ms (autotest), 0,66 ms (profilage) |
| Moteur en `-O2` (pragma), poids en RAM | 400–500 µs au profilage | **344 µs** au profilage ✓ (mieux que prévu) |

Bilan : de 3,7 ms à 0,34 ms, soit **×10,7**, sans changer un bit des résultats (toujours 24/24, 3,0e-7).

Images reçues par l'API, espacées de 0,5 s (`send.py`) : **500 à 620 µs** (prédiction 350–450 µs ✗).
Hypothèse : entre deux images, le traitement réseau (Wi-Fi, déchiffrement de l'API) chasse du cache
le code du moteur et d'`expf`, relu ensuite en flash. Non vérifié ; piste : placer le moteur en IRAM.

### 4.2 Coûts unitaires (cycles à 240 MHz, bouton « Profilage »)

| Opération | Prédiction | Mesure |
|---|---|---|
| Division float (pas d'instruction matérielle) | 30–100 | **63** ✓ |
| `expf` | 100–300 | **302** ✓ |
| Multiplication-addition, poids en RAM, code `-Os` | 5–8 | **13,3** ✗ |
| Multiplication-addition, poids en RAM, code `-O2` | 5–6 | **5,5** ✓ |
| Multiplication-addition, poids en flash hors cache | ≈ 90 | **155** ✗ |
| Multiplication-addition, poids en flash déjà en cache | ≈ RAM | **14,2** ✓ |

### 4.3 Où va le temps (poids en RAM)

| Étage | Code `-Os` (661 µs) | Code `-O2` (344 µs) | Part en `-O2` |
|---|---|---|---|
| Couche 0, produit (5 880 multiplications-additions) | 428 µs (17,5 cycles chacune) | **128 µs (5,2 cycles)** | 37 % |
| Prétraitement (196 divisions) | 122 µs | 98 µs | 28 % |
| Couche 0, sigmoïde (30 neurones) | 49 µs (391 cycles/neurone) | 55 µs (436 cycles/neurone) | 16 % |
| Couche 1, produit + softmax | 38 µs | 26 µs | 8 % |
| Reste (mesure, recherche du maximum…) | 23 µs | 37 µs | 11 % |

### 4.4 Explications

- **Flash** : 23,5 Ko de poids ne tiennent pas dans le cache de la flash (~32 Ko, partagé avec le code).
  Chaque relecture coûte ~155 cycles par poids (flash en DIO à 40 MHz). Le code lui-même est lu en flash :
  une image isolée après une pause (797 µs) est plus lente que des images enchaînées (661 µs).
- **`-Os`** : désassemblage avec le compilateur d'ESP-IDF (`xtensa-esp32-elf-g++` 14.2.0, esp-14.2.0_20260121).
  En `-Os`, la boucle du produit scalaire fait 9 instructions avec un saut (≈ 13 cycles, conforme à la mesure).
  En `-O2`, elle en fait 5, dans une boucle matérielle Xtensa (`loop`) : ≈ 5–6 cycles attendus.
  ESPHome compile en `-Os` ; la directive `#pragma GCC optimize("O2")` ne s'applique qu'à `nn_engine.cpp`
  (+130 octets).
- Hypothèse écartée : une chaîne de dépendances entre additions. En `-O2`, cinq instructions séparent deux `madd.s`.
- Écart 17,5 / 13,3 cycles en `-Os` (moteur / microbanc) : la boucle du moteur a une instruction de plus
  (`extui`, qui recalcule l'indice sur 16 bits, probablement à cause du type `uint16_t` de `n_in`),
  et le calcul d'adresse en dépend aussitôt. En `-O2`, les deux tombent à 5,2–5,5 cycles : sans enjeu.

### 4.5 Décisions

- **`weights_in_ram: true`** : 25 Ko de RAM sur ~135 Ko libres, ×4,2 en vitesse, résultats identiques au bit près.
- **Moteur en `-O2`** par directive, le reste du firmware en `-Os` : produit matriciel ×3,4.
- **Pour SpectraSense** : une sigmoïde coûte autant qu'environ 30 multiplications-additions.
  Dans les couches cachées d'un CNN (des milliers d'activations), préférer **ReLU** (1 à 2 cycles).
- **Pas de quantification int8 pour l'instant** : la flash et le temps ne sont pas des contraintes
  (0,66 ms, soit < 0,1 % du temps processeur à une inférence par seconde). La puce a une unité flottante.
  À reconsidérer si un réseau ne tient plus en RAM ou devient trop lent.

---

## 5. Entraînement sur le PC (choix du moteur du banc d'essai)

`verif_gpu.py` : TITAN Xp (Pascal, `sm_61`) reconnue, calculs exacts (float64 = numpy à 2e-13), reproductible.
**PyTorch 2.14 est la dernière version qui prend en charge Pascal** (CUDA 12.6) → `torch==2.14.1+cu126`
figé dans `requirements.txt` ; ne pas faire `pip install -U torch`. Pilote NVIDIA : rester sur la branche 580.

Une époque de [196, 30, 10] sur MNIST 14×14 (`verif_gpu.py`, `mesures_moteur.py`) :

| Moteur | m = 10 | m = 100 |
|---|---|---|
| Nielsen (numpy, image par image) | ≈ 1,4–1,7 s | — |
| PyTorch, processeur, gradients automatiques | 1,93 s | 0,21 s |
| PyTorch, carte graphique, gradients automatiques | 4,33 s ✗ | 0,47 s |
| **numpy, processeur, rétropropagation écrite à la main** | **0,18 s** | 0,10 s |
| Carte graphique, exécution normale, 1 réseau | 2,04 s | — |
| Carte graphique, CUDA graph, 1 réseau | 0,40 s | — |

Plusieurs réseaux entraînés ensemble (m = 10), temps par réseau : 32 → 19 ms (CUDA graph) ;
128 → 7,9 ms (CUDA graph) ; 512 → 4,8 ms ; 2 048 → 3,9 ms. La carte sature vers 512 réseaux.

Explication : sous Windows, chaque petit calcul lancé sur la carte coûte ~40 µs. Avec un si petit réseau,
la carte attend plus qu'elle ne calcule, sauf si l'on entraîne des centaines de réseaux à la fois.

**Décisions**
- Banc **interactif** : numpy float32 sur le processeur, rétropropagation écrite à la main.
- Banc **en lot** : PyTorch sur la carte, paquets de 128 à 512 réseaux de même architecture
  (CUDA graph jusqu'à ~256). Ordre des images partagé dans un paquet ; η, λ, poids de départ propres à chaque réseau.
- Deux moteurs → chaîne de tests d'équivalence : numpy float64 = `network2` de Nielsen (au bit près ou 1e-12),
  puis numpy float32, puis PyTorch carte ≈ numpy float32 (≈ 1e-6 après une époque).
- La taille de mini-batch reste un hyperparamètre de qualité : m = 100 n'est que 2 fois plus rapide que m = 10.

**Banc d'essai réalisé** (`banc/`, voir `banc/README.md`) : un seul moteur écrit pour numpy et PyTorch
(plutôt que deux moteurs à tenir équivalents), R réseaux par paquet. Tests : Nielsen = numpy float64 à 2e-15
après 300 mises à jour ; gradients conv1d / maxpool = différences finies (≤ 6e-8) ; paquet de 3 = 3 réseaux
séparés (écart nul) ; PyTorch = numpy (1e-15 en float64, 1e-6 en float32). Coût ESP estimé
pour [196, 30, 10] : 348 µs (mesuré : 344 µs).

Signal 1D synthétique (motif à position aléatoire), 5 époques : conv1d 8 canaux + max-pooling,
3 988 poids → **98,5 %** ; perceptron 32 neurones, 4 260 poids → 94,3 %. Même résultat qu'avec ton 7 manuscrit :
la convolution reconnaît un motif où qu'il soit.

**Moteur C++ de l'ESP** : ReLU et tanh ajoutées. Chaîne complète vérifiée avec un réseau ReLU
entraîné par le banc : export, puis 24/24 au test du moteur (écart 1,9e-6, sous la tolérance de 1e-5).

---

## 6. Choix du réseau déployé — lots d'hyperparamètres (11 octobre)

Tâche MNIST 14×14 (facteur 2), une couche cachée, 30 époques, m = 10, planning cosinus.
Sélection sur la **validation** (10 000 images), 3 graines par réglage ; moyenne ± écart-type entre graines.

**Lot 1** — 4 tailles (10, 20, 30, 50) × sigmoïde / ReLU × η ∈ {0,5 ; 1 ; 2 ; 3} × λ ∈ {0 ; 2 ; 5},
288 réseaux, 8 paquets de 36 sur la TITAN (≈ 4,4 s par époque et par paquet, ≈ 18 min en tout).

| Réseau | Meilleur réglage | Validation | ESP estimé | Poids |
|---|---|---|---|---|
| D50 sigmoïde | η = 2, λ = 2 | **98,05 ± 0,05** | 475 µs | 10 360 (40 Ko) |
| D50 ReLU | η = 0,5, λ = 5 | 97,90 ± 0,05 | 385 µs | 10 360 (40 Ko) |
| D30 sigmoïde | η = 3, λ = 2 | 97,59 ± 0,08 | 348 µs | 6 220 (24 Ko) |
| D30 ReLU | η = 0,5, λ = 5 | 97,27 ± 0,17 | 294 µs | 6 220 (24 Ko) |

- Sigmoïde : optimum à l'intérieur de la grille. ReLU : optimum **au bord** (plus petit η, plus grand λ),
  et réseaux morts dès η ≥ 2 (10,6 % = toujours la même réponse) : dérivée 1 au lieu de ≤ 0,25,
  donc des pas ~4 fois plus grands à η égal. → lot 2.
- Vitesse : ReLU ne gagne que ~20 % (les 50 sigmoïdes ≈ 91 µs ; le produit matrice × vecteur,
  9 800 mult.-add. ≈ 216 µs, ne change pas).

**Lot 2** — ReLU seulement, D30 et D50, η ∈ {0,1 ; 0,2 ; 0,3 ; 0,5} × λ ∈ {5 ; 10 ; 20}, 66 réseaux nouveaux.
Prédiction : optimum ReLU vers η ≈ 0,2–0,3 avec λ plus grand (la décroissance des poids par pas vaut ηλ/n).
Mesure : **plateau**. D50 ReLU entre 97,54 et 97,90 sur toute la zone, η = 0,2–0,5 et λ = 5–10 dans
0,1 point (l'écart entre graines) ; D30 ReLU plat vers 97,0–97,3. Le réglage η = 0,5, λ = 5 était déjà le bon.

**Décision** : D50 ReLU, η = 0,5, λ = 5, graine 3 (`c5ccb25201`, validation 97,96 %).
0,15 point de moins que la sigmoïde (~15 images sur 10 000), accepté : ReLU est l'activation des
couches de convolution de SpectraSense, et c'est sa première exécution sur l'ESP.
**Test (mesuré une seule fois) : 97,80 %**, contre 96,86 % pour le [196, 30, 10] sigmoïde déployé jusqu'ici.
Export : float32 = float64 sur les 10 000 images de test ; 24/24 au test du moteur sur PC
(écart max 1,1e-6), mutations détectées ; compilation Xtensa : 41,5 Ko de poids et d'images de test.

**Sur la carte** (bouton « Profilage », moyenne sur 24 images, poids en RAM). Prédiction : ≈ 385 µs.

| Étage | Mesuré | Estimé par le banc |
|---|---|---|
| Prétraitement | 94,8 µs | 100,5 µs |
| Couche 0, produit (9 800 mult.-add.) | 210,1 µs, 5,1 cycles / mult.-add. | 216,4 µs |
| Couche 0, ReLU (50 neurones) | 9,5 µs, **46 cycles / neurone** | 0,8 µs (4 cycles) |
| Couche 1, produit (500 mult.-add.) | 11,5 µs, 5,5 cycles / mult.-add. | 11,0 µs |
| Couche 1, softmax (10 neurones) | 22,2 µs, 532 cycles / neurone | 18,9 µs |
| **Total** | **377,8 µs** (somme des étages 348,0) | 385,1 µs |

- Prédiction tenue à 2 % près ; +34 µs par rapport au [196, 30, 10] sigmoïde (344 µs) pour +0,94 point au test.
- ReLU coûte 46 cycles par neurone au lieu des 4 estimés, mais ne pèse que 9,5 µs contre ~91 µs pour
  50 sigmoïdes. Coûts unitaires du banc recalés (`banc/couts.py`) : ReLU 46, partie fixe 8 000 cycles
  (mesurée entre 7 150 et 9 000) → estimations 389,7 µs (mesuré 377,8) et 343,6 µs (mesuré 344).
- Le profilage affichait « softmax » pour la couche ReLU : étiquette corrigée.

**Autotest sur la carte** : 24/24 conformes, écart max 1,4e-6 (PC : 1,1e-6), 627 µs en moyenne
(chemin « froid », comme par l'API ; 378 µs au profilage). Première exécution de ReLU sur l'ESP validée.

**Confiance** (même réseau sur le PC ; « Python » dans l'autotest = prédiction du PC, pas la vraie classe).
Les 4 images difficiles : 9009 (un 7) → 2 à 100 % ; 3558 (un 5) → 0 à 100 % ; 1754 (un 7) → 7 à 44 %
(3 à 44 %) ; 5870 (un 0) → 0 à 50 % (9 à 48 %). Deux justes mais hésitantes, deux fausses et sûres d'elles.
Sur la validation, en ne répondant qu'aux exemples les plus sûrs (même couverture pour tous) :

| Réseau | Erreurs sur 10 000 | Fausses à 95 % de couverture | à 90 % | à 80 % |
|---|---|---|---|---|
| D30 sigmoïde (déployé avant) | 311 | 103 | 38 | 14 |
| D50 sigmoïde | 197 | 47 | 17 | 5 |
| D50 ReLU (déployé) | 204 | 54 | 19 | 11 |

- La taille compte le plus : à 90 % de couverture, D50 fait 2 fois moins d'erreurs que D30.
- ReLU est plus sûr de lui que la sigmoïde, y compris quand il se trompe (48 erreurs au-dessus de
  90 % de confiance, contre 27) : à couverture égale, il lui faut un seuil plus haut (0,966 contre 0,938
  pour 90 %). Pour SpectraSense : le seuil de refus se règle sur la validation, réseau par réseau,
  jamais à une valeur fixe.

---

## 7. Questions ouvertes

- ReLU : 46 cycles par neurone pour une comparaison et une copie (≈ 8 instructions). Hypothèse : le code
  d'`activate` est relu en flash à chaque image (défauts de cache, ~155 cycles chacun), comme la
  softmax passée de 453 à 532 cycles. À vérifier en plaçant le moteur en IRAM. Enjeu : < 10 µs.

- Les activations sont un peu plus lentes en `-O2` (sigmoïde 391 → 436 cycles par neurone, softmax 379 → 453),
  alors que `expf` coûte toujours ~300 cycles. Non expliqué (code de `expf` relu en flash ?). Enjeu : ~10 µs.
- Flash en QIO à 80 MHz (≈ 4 fois plus de débit) avec les poids laissés en flash : prédiction ≈ 1 ms. Non testé.
- Prochain poste : le prétraitement (98 µs, dont ~51 µs de divisions). Multiplier par l'inverse :
  −45 µs environ, mais il faut changer la spécification des deux côtés (numpy et C)
  pour garder l'égalité au bit près. Puis les 40 `expf` (~50 µs).

---

## 8. SpectraSense sur l'ESP (14 au 17 octobre 2026)

Réseau `162dcb9fbf` (C8k3-P2-C16k5-P2-C16k5-P2-D24-12). Détails des choix d'apprentissage :
`spectrasense/tools/evaluation/RESULTATS_TEST.md`.

### 8.1 Inférence

| Mesure | Valeur |
|---|---|
| Autotest sur la carte | 16/16 fenêtres conformes, écart relatif max 4,1e-7 |
| Temps par inférence (poids en RAM) | 3,4 ms mesurés (somme des couches 3,4 ms), 3,7 ms en autotest |
| Convolution, noyau 5 | ≈ 8,5 cycles par multiplication-addition |
| Convolution, noyau 3 (première couche) | ≈ 20 cycles par opération |
| Max-pooling | ≈ 20 cycles par comparaison |
| Dense | ≈ 5,2 cycles par opération |

Décision : pas d'optimisation du moteur (réordonner les boucles de conv1d ferait gagner ≈ 1,4 ms,
inutile à une inférence par seconde). `banc/couts.py` est recalé sur ces mesures.

### 8.2 Chaîne en direct

| Choix | Pourquoi | Mesure |
|---|---|---|
| Rafales HA re-datées en direct (`EntreeHa`) | ≈ 2 % des lectures arrivent groupées à quelques ms ; l'entraînement les a re-datées (`debrouiller_rafales`) | sans re-datage, un pic de maintien de 1 s disparaît |
| Horloge en retard de 2 s (`clock_delay`) | une rafale tardive doit retrouver sa place | 15 jours rejoués : 5 s d'état différentes de Python au total ; 99 % des publications < 2,1 s après la seconde concernée (0 différence à 5 s, mais 99 % à 5 s) |
| Buffer de sortie de l'adaptateur supprimé (récepteur appelé case par case) | un trou de plus de 64 s perdait des secondes | journée de validation 09/04 corrigée |
| État long gardé en flash avec l'heure de HA | après un redémarrage, arrêt et veille sont indiscernables (0 W) | 14/10 : allumage publié « broyage » après un flash |
| Cycle commencé dans l'état inconnu : changement de famille allumage ↔ café | le réseau ne distingue un allumage d'un café qu'après ≈ 25 s | validation inchangée |
| Appareils simples : « arrêt » après 8 s (au lieu de 12) | 12 s fusionnait deux utilisations du consommateur inconnu | validation : inconnu 33/34 (au lieu de 30/34), 0 coupure, 0 fausse alerte ; à 3 et 5 s, coupures sur l'inconnu |
| Compteur de cafés : cycle café terminé avec ≥ 5 s de broyage et ≥ 5 s d'extraction publiés | écarte les fins d'allumage et les consommateurs plats | validation : 48/48, 0 en trop (plateau parfait de 5 à 10 s) |

### 8.3 Ce que montrent les poids

- Filtres morts (sortie toujours nulle) : couche 1, filtres 1 et 5 ; couche 2, filtres 1, 3, 4 et 14 ;
  couche 3, filtre 0. Piste pour le prochain entraînement : Leaky ReLU ou petit biais positif au départ.
- Couche 1 : niveau, présence, forte puissance, chute, calme. Couche 2 : filtre 2 « repos »,
  filtre 9 « long plateau » (grille-pain), filtre 13 « fin de forte puissance ».
- Plus des deux tiers des paramètres sont dans la couche dense (5 016 sur 7 324), qui encode la position
  dans la fenêtre de 130 s.

---

## 9. Micro INMP441 et son en UDP (21 octobre 2026)

But : enregistrer le son des machines (café, grille-pain) pour des frontières de phase physiques
(dataset), pas pour la production. Composant `audio_udp`, config `cnn-test-micro.yaml` (copie de
`cnn-test.yaml` + micro : retour arrière = recoller `cnn-test.yaml`). Le son part brut au PC
(`spectrasense/tools/audio/recepteur.py`), en plus des deux réseaux qui tournent sur la carte.

### 9.1 Choix

| Choix | Raison |
|---|---|
| Câblage SCK GPIO26, WS GPIO25, SD GPIO33, L/R à la masse, VDD 3V3 | broches libres, aucune ne sert au démarrage ; `channel: left` fonctionne |
| Lecture 16 kHz, 32 bits, mono ; envoi brut en UDP, aucun traitement sur la carte | la carte reste libre pour les réseaux ; tout se fait sur le PC, sans limite de mémoire |
| Paquets de 45 ms (720 échantillons, 1 468 octets ; 30 ms au départ), en-tête : session, seq, index, t_us | seq = pertes visibles ; index = position exacte de chaque échantillon ; l'heure vient du PC (droite heure/index), pas d'horloge à régler sur la carte ; 720 : un tiers de paquets en moins, la file d'envoi de la carte (~32 paquets) couvre ~1,5 s au lieu de 1 s |
| **16 bits** (au départ 24) | bruit propre du micro ≈ -87 dBFS, au-dessus du pas de 16 bits : 24 bits n'apportaient rien ; 32 ko/s au lieu de 48 |
| `power_save_mode: none` | obligatoire pour un flux continu (9.2) |
| `post_connect_roaming: false` | la carte ne bouge pas ; chaque recherche de point d'accès (toutes les 5 min) coûtait ~40 paquets |
| Pause de l'envoi après 3 échecs de suite (0,5 s, doublée jusqu'à 8 s) | une liaison saturée ne doit pas priver l'API et les mesures de HA (9.2) |
| `output_power: 15dB` (défaut 20) | moins de pics de courant à l'émission ; à -55 dBm la marge reste large (9.2, hypothèse de l'alimentation) |
| `api: reboot_timeout: 5min` (défaut 15) | sortir plus vite d'une liaison « zombie » (9.2) |
| Niveau publié après un passe-haut Butterworth 2e ordre à 100 Hz | sous 100 Hz, le micro capte des fluctuations de pression (souffle, -31 dBFS vers 1-2 Hz au bureau) qui masquaient le son audible (≈ -65 dBFS au bureau) ; le son envoyé et la crête restent bruts |
| Témoin d'état sur la LED bleue, décidé par la carte seule | voir un problème sans HA : éclair bref toutes les 5 s = OK ; 2 / 3 / 4 clignotements puis 2 s = son saturé (pause < 30 s) / pas de connexion à l'API / Wi-Fi perdu ; éteinte = carte figée ou sans courant |
| Micro coupé quand le PC est éteint (automatisation HA) | rien ne reçoit le son ; la carte arrête d'émettre dans le vide |

### 9.2 Problèmes rencontrés

| Situation | Paquets perdus (PC) | Retard médian / p95 | Échecs d'envoi (carte) | Secondes invalides SpectraSense |
|---|---|---|---|---|
| Wi-Fi `power_save_mode: light` (défaut ESP32), bureau | 0 → 94 % en 4 min | 18 ms → 0,3 s / 3,9 s | +16 /s, puis déconnexion de l'API | — |
| `power_save_mode: none`, bureau, alimentation par le PC (8 min) | 0,07 % | 18 / 61 ms | 0 | 0 |
| `none`, arrière-cuisine, chargeur Samsung + **câble douteux** (-60 dBm) | 35 à 60 % | 250 ms / 0,5-1,9 s | +12 /s | 161 s en 7 min |
| `none`, arrière-cuisine, même chargeur, **autre câble** (-57 dBm, 3 min) | 0,03 % | 24 / 62 ms | 0 | 0 |

1. **Économie d'énergie du Wi-Fi** : en « light », la radio dort entre les balises, la file d'envoi
   déborde, puis l'API se déconnecte. Réglé par `power_save_mode: none`.
2. **Câble d'alimentation** : avec un mauvais câble, rien ne redémarre et le signal reçu reste bon,
   mais les pics d'émission (300-400 mA) font chuter la tension ; les trames partent mal et sont
   réémises jusqu'à saturer la liaison. Le flux audio saturé prive aussi la chaîne SpectraSense des
   mesures de HA (161 s invalides). Un autre câble a tout réglé… pendant un temps.
3. **Décrochages restants** (même chargeur, nouveau câble) : à 20:38, 21:25 et 22:01, après 3 à 28 min
   sans problème, les échecs montent pendant ~2 min, puis la carte devient injoignable (`Errno 113`,
   pas de réponse ARP). Elle **se croit pourtant connectée** (témoin à 2 clignotements, pas 4), laisse
   passer un paquet de temps en temps, et ne s'en sort que par le redémarrage automatique (API sans HA
   pendant 15 min). Signal -53 à -60 dBm, mémoire stable, aucune machine en marche, pas de micro-ondes.
   La carte entend le point d'accès mais n'arrive plus à émettre : **piste de l'alimentation**
   (chargeur), jamais vue au bureau sur le PC. Réponse : `output_power: 15dB`, redémarrage après 5 min
   sans HA, pause de l'envoi. **À confirmer** : une heure dans l'arrière-cuisine sur une autre
   alimentation (batterie ou autre chargeur). Pour la carte de prod : alimentation de qualité, et un
   condensateur de 470 µF entre 5 V et GND en assurance.
4. **Premier flash « annulé »** : la carte débranchée moins de 60 s après le flash est repartie sur
   l'ancien firmware (`OTA rollback detected`) : un nouveau firmware n'est validé qu'après 60 s sans
   redémarrage. Laisser la carte tranquille au moins une minute après chaque flash.

5. **Coupures périodiques sous charge** (22 et 23/10, émission à 15 dB, plus aucun redémarrage ni
   décrochage « zombie » : une seule session de la carte pendant plus de 30 h) :
   - pertes de 3,7 à 5,7 % ; hors coupures, retard de 40 ms en médiane ;
   - **toutes les ~76 s, très régulièrement, la liaison s'arrête 3 à 4 s** quand le micro diffuse :
     ~60 % des paquets perdus sont partis de la carte sans arriver (perdus en radio), ~40 % refusés
     ou abandonnés par la carte quand sa file déborde ;
   - au repos (micro coupé), pas de rythme de 76 s (pings du 22/10, 11 ms en médiane) ; micro allumé
     (pings du 23/10), la carte perd 3-4 pings à chacune des 24 coupures sur 30 min, et un autre
     appareil en 2,4 GHz **sur le même Deco** (bureau) est touché aussi, plus légèrement
     (14 pings perdus dans ces fenêtres, ~5 attendus au hasard) : phénomène radio côté Wi-Fi sous
     charge, pas côté carte (micro, calcul et production des paquets continus : seq et index sans trou) ;
   - coupures longues (9 à 30 s, au repos) **communes à cnn-test (arrière-cuisine) et à un appareil du
     garage sur un autre Deco**, pas à l'appareil du bureau sur le même Deco que cnn-test : cause locale
     à ces pièces (électricité ou source radio), non élucidée.

   **Décision (23/10)** : on arrête l'enquête. Le flux audio ne sert qu'au dataset, les trous sont
   connus (journal) et comblés par du silence, et quelques pourcents de pertes ne gênent pas
   l'entraînement. Sur la carte de prod, pas de flux continu : le réseau tourne sur la carte, le Wi-Fi
   ne sert qu'à publier dans HA (une coupure retarde la publication, pas la détection).
   Dernier réglage : paquets de 720 échantillons.

- Mémoire : ~100 Ko libres en régime normal (le micro coûte ~10 Ko), minimum 78-85 Ko ; elle tombe
  vers 52 Ko quand les paquets s'accumulent.
- À surveiller sur la carte de prod (PZEM + sondes) : « Micro échecs d'envoi », « Micro paquets
  abandonnés » et « SpectraSense secondes invalides ».

## Matériel de référence

- Carte `cnn-test` : ESP-WROOM-32, puce **ESP32-D0WD-V3 rév. 3.1**, deux cœurs à **240 MHz**,
  flash **4 Mo en DIO à 40 MHz**, USB-série CH340, LED « D2 » sur GPIO2 (active à l'état haut).
- ESPHome 2026.8.2 (HA 2026.9.2), ESP-IDF 5.5.5, GCC Xtensa 14.2.0.
- PC : Windows, Python 3.12.10, numpy 2.5.3, PyTorch 2.14.1+cu126, NVIDIA TITAN Xp 12 Go, pilote 581.57.

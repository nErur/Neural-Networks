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

---

## 6. Questions ouvertes

- Les activations sont un peu plus lentes en `-O2` (sigmoïde 391 → 436 cycles par neurone, softmax 379 → 453),
  alors que `expf` coûte toujours ~300 cycles. Non expliqué (code de `expf` relu en flash ?). Enjeu : ~10 µs.
- Flash en QIO à 80 MHz (≈ 4 fois plus de débit) avec les poids laissés en flash : prédiction ≈ 1 ms. Non testé.
- Prochain poste : le prétraitement (98 µs, dont ~51 µs de divisions). Multiplier par l'inverse :
  −45 µs environ, mais il faut changer la spécification des deux côtés (numpy et C)
  pour garder l'égalité au bit près. Puis les 40 `expf` (~50 µs).

---

## Matériel de référence

- Carte `cnn-test` : ESP-WROOM-32, puce **ESP32-D0WD-V3 rév. 3.1**, deux cœurs à **240 MHz**,
  flash **4 Mo en DIO à 40 MHz**, USB-série CH340, LED « D2 » sur GPIO2 (active à l'état haut).
- ESPHome 2026.8.2 (HA 2026.9.2), ESP-IDF 5.5.5, GCC Xtensa 14.2.0.
- PC : Windows, Python 3.12.10, numpy 2.5.3, PyTorch 2.14.1+cu126, NVIDIA TITAN Xp 12 Go, pilote 581.57.

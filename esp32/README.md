# ESP32 — MNIST puis SpectraSense via ESPHome

Plan initial (MNIST, historique) : `PLAN.md`. Journal des essais et des choix : `ESSAIS.md`. Carte de test `cnn-test` : ESP-WROOM-32, puce ESP32-D0WD-V3 rév. 3.1, flash 4 Mo,
USB-série CH340, MAC b0:a7:32:27:da:20 (`board: esp32dev`, framework ESP-IDF).

## Organisation

- La carte est gérée par l'ESPHome de Home Assistant (mises à jour, OTA, secrets).
- Les composants `nn_mnist` et `nn_spectrasense` sont lus par HA dans ce dépôt (GitHub privé `nErur/Neural-Networks`,
  dossier `esp32/esphome/components`). Les headers générés par `export.py` sont commités :
  chaque modèle déployé correspond à un commit.
- `esphome/cnn-test.yaml` = référence de la config saisie dans HA.
- `esphome/cnn-test-local.yaml` = variante de secours compilée sur le PC (CLI ESPHome).
  **Pas à jour** (sans `nn_spectrasense`) : Antoine décide de son sort ; ne pas la modifier sans lui.

## Cycle de travail

1. Modifier le composant (ou régénérer les poids), commit, push.
2. Dans l'ESPHome de HA : « Install » sur `cnn-test` (OTA). Avec `refresh: 0s`,
   HA récupère le dernier commit de `master` à chaque compilation.

## Secrets côté HA (`/config/esphome/secrets.yaml`)

    github_user: nErur
    github_token: "github_pat_..."   # jeton fine-grained, lecture seule, ce dépôt uniquement

ESPHome stocke ces identifiants en clair dans son dossier de travail : jeton en lecture seule,
limité à ce dépôt. Noter sa date d'expiration.

## Étape 0 — validation

- Logs au démarrage : `[nn_mnist] hello depuis nn_mnist` ; bloc de config : `... (étape 0), v1`.
- Changer `v1` en `v2` dans `components/nn_mnist/nn_mnist.cpp`, commit, push, Install (OTA) :
  `v2` doit apparaître.

## Étape 1 — entraînement et export (PC)

Depuis la racine du dépôt, venv Nielsen activé :

    python esp32\python\train.py              # [196, 30, 10] sur 14×14 -> esp32\modeles\*.npz
    python esp32\python\export.py             # dernier modèle 14×14 -> headers C++

- `python/donnees.py` : chargement MNIST en uint8 + prétraitement de référence (la spécification).
- `train.py --facteur 1` (28×28) ou `--facteur 4` (7×7) : modèles de comparaison.
- `export.py` écrit `model_weights.h` et `test_vectors.h` dans `esphome/components/nn_mnist/`
  et s'arrête si un autocontrôle échoue (float32 vs float64, disposition des poids).

## Étape 3 — moteur C++ testé sur le PC

    python esp32\tests\lancer_tests.py

- Compile `tests/test_engine.cpp` avec le `nn_engine.cpp` du composant (pas une copie) et compare,
  pour les 24 images de `test_vectors.h` : entrée réduite (identique au bit près), sortie de chaque
  couche (écart < 1e-5), prédiction (identique à Python).
- Puis « teste le test » : trois bugs introduits dans une copie du moteur (poids transposés,
  division par 256, biais oublié) doivent chacun faire échouer le test au bon étage.
- Demande g++ ou clang++ dans le PATH. Sous Windows : MSYS2 (`pacman -S mingw-w64-ucrt-x86_64-gcc`),
  puis ajouter `C:\msys64\ucrt64\bin` au PATH.

## Étape 4 — le réseau sur l'ESP

Après push, recopier `esphome/cnn-test.yaml` dans la config de la carte dans HA, puis Install (OTA).

- Boutons : « Image de test suivante » (parcourt les 24 images embarquées), « Lancer l'autotest »
  (les 24 d'un coup ; critère : « 24/24 conformes »).
- Capteurs : chiffre prédit, confiance (%), marge (%), temps d'inférence (µs), étiquette attendue
  et écart max vs Python (images de test seulement), modèle, bilan de l'autotest.
- Action API `classify` (784 entiers 0..255) : utilisée par `send.py` à l'étape 5.
- Détail étage par étage dans les logs (niveau DEBUG).

## Étape 5 — envoyer des images depuis le PC

Une fois : `pip install aioesphomeapi` (venv activé), et la clé API de la carte en local, au choix :
ligne `cnn_test_api_key: "..."` dans `esphome\secrets.yaml` (ignoré par git, même valeur que dans le
`secrets.yaml` de HA), ou variable d'environnement `CNN_TEST_API_KEY`.

    python esp32\python\send.py                       # 10 images de test au hasard
    python esp32\python\send.py --index 9729 4601     # images précises
    python esp32\python\send.py --png chiffre.png     # tes propres images, cadrées comme MNIST
    python esp32\python\send.py --png chiffre.png --sans-carte --apercu apercu.png   # vérifier le cadrage
    python esp32\python\send.py --host 192.168.68.124 # si cnn-test.local ne répond pas

Exemples d'images manuscrites et d'aperçu : `esp32/exemples/`.

L'image brute (784 entiers) passe par l'action API `classify` ; l'ESP fait prétraitement et inférence.
Le script compare la prédiction de l'ESP à celle de Python (float32, même modèle).

## Variante de secours : compilation sur le PC

    cd C:\AItools\Projets\Neural-Networks\esp32
    py -3.12 -m venv .venv-esphome
    .\.venv-esphome\Scripts\Activate.ps1
    pip install esphome
    cd esphome
    esphome run cnn-test-local.yaml

Secrets dans `esphome\secrets.yaml` (ignoré par git).

## SpectraSense sur l'ESP

Le moteur (`esphome/components/nn_mnist/nn_engine.*`) exécute aussi `conv1d` et `maxpool1d`
(signaux rangés canal par canal, comme le banc). Le composant `nn_spectrasense` le réutilise : une
seule copie du moteur dans le firmware, donc `components: [nn_mnist, nn_spectrasense]`.

### Fichiers du composant `nn_spectrasense`

| Fichier | Rôle |
|---|---|
| `chaine.h/.cpp` | la chaîne complète en C++ pur (sans ESPHome) : `EntreeHa` (rafales re-datées, horloge), `Adaptateur` (grille 1 s), porte, fenêtre de 130 s, réseau, `AutomateMachine`, `EntiteSimple`, compteur de cafés. **Miroir exact** de `adaptateur.py`, `cibles.py` (porte) et `automate.py` du dépôt spectrasense |
| `nn_spectrasense.h/.cpp` | liaison ESPHome : capteur de puissance, publication des entités, mémoire en flash, autotest, profilage |
| `__init__.py` | schéma YAML et génération du code |
| `model_weights.h`, `test_vectors.h` | générés par `export_conv.py` (espace `nn_ss`) ; ne pas éditer |

### Chaîne en direct

Lectures de puissance (capteur `power`) → `EntreeHa` : les rafales de HA (lectures arrivées à quelques ms
d'écart) sont re-datées comme `debrouiller_rafales` → grille à 1 s → porte (P > 20 W dans les 5 dernières
secondes) → réseau, seulement porte ouverte → automate → entités. Une horloge ferme les secondes même sans
lecture, avec `clock_delay` de retard pour qu'une rafale tardive retrouve sa place (2 s via HA ; 0 s prévu
sur la carte PZEM, qui date elle-même ses lectures).

### Options YAML

| Option | Défaut | Rôle |
|---|---|---|
| `power` | — | capteur de puissance (W) ; sans lui, autotest et profilage seulement |
| `time_id` | — | horloge réelle (HA) : l'état long arrêt / veille / maintien survit aux redémarrages (flash) |
| `clock_delay` | 2s | retard de l'horloge qui ferme les secondes dans les silences |
| `threshold` | 0.95 | probabilité minimale pour publier un changement |
| `hold` | 1 | secondes consécutives au-dessus du seuil |
| `peak_threshold` | 0.5 | seuil propre aux pics de maintien |
| `off_hold` | 8s | repos continu avant « arrêt » d'un appareil simple |
| `weights_in_ram` | true | poids copiés en RAM au démarrage (plus rapide) |

Entités : `machine` (texte, 8 états : arrêt, veille, maintien, chauffe, régulation, broyage, extraction, fin),
`toaster`, `kettle`, `unknown_load` (binaires marche / arrêt), `coffee_count` (total, gardé en flash),
et en diagnostic `gate`, `network_class`, `network_confidence`, `class_probabilities` (12 capteurs,
probabilités brutes avant l'automate), `invalid_seconds`, `inference_time`, `max_error`, `model_id`,
`autotest`. Boutons « SpectraSense autotest » et « SpectraSense profilage » dans `cnn-test.yaml`.

### Ajouter une option ou une règle

1. Règle de l'automate ou de l'adaptateur : la modifier **dans `spectrasense` (Python) et dans
   `chaine.*` (C++)**, avec la même valeur par défaut des deux côtés.
2. Relancer la parité dans spectrasense : `python tools\evaluation\parite_cpp.py --jours 6`, puis
   `--jours 15 --flux 2`. Attendu : « SUCCÈS ».
3. Exposer le réglage dans `__init__.py` (schéma + `to_code`) et `nn_spectrasense.*`, l'ajouter à
   `cnn-test.yaml` avec un commentaire.
4. Commit, Antoine pousse, puis « Install » dans HA.

### Nouveau modèle

    python esp32\python\export_conv.py <id> --sortie esp32\esphome\components\nn_spectrasense --espace nn_ss --inclure-moteur esphome/components/nn_mnist/nn_engine.h --par-classe 1
    python esp32\python\export_conv.py <id>                       (headers de test dans esp32\tests\spectrasense)
    python esp32\tests\lancer_tests_conv.py                      (couche par couche + mutations)

Puis copier le `.npz` dans `esp32\modeles\`, relancer la parité, flasher, vérifier l'autotest (16/16)
et le profilage. Modèle actuel : `162dcb9fbf` — 28/28 fenêtres conformes sur PC (écart ≤ 4,4e-7),
16/16 sur la carte, 3,4 ms par inférence (profilage du 14/10/2026).

### Rejouer une journée en C++ sur le PC

`tests/spectrasense/rejouer.cpp` fait tourner `chaine.cpp` sur un fichier « t,p » et écrit chaque seconde
(grille, porte, probabilités, états, cafés). `parite_cpp.py` (dépôt spectrasense) le compile et le compare
à Python. Arguments : `rejouer mesures.csv sortie.csv [seuil tenue seuil_pic [retard [etat_initial]]]`.

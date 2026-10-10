# ESP32 — MNIST via ESPHome

Plan complet : `PLAN.md`. Carte de test : ESP-WROOM-32 (`board: esp32dev`, framework ESP-IDF).

## Organisation

- La carte est gérée par l'ESPHome de Home Assistant (mises à jour, OTA, secrets).
- Le composant `nn_mnist` est lu par HA dans ce dépôt (GitHub privé `nErur/Neural-Networks`,
  dossier `esp32/esphome/components`). Les headers générés par `export.py` sont commités :
  chaque modèle déployé correspond à un commit.
- `esphome/cnn-test.yaml` = référence de la config saisie dans HA.
- `esphome/cnn-test-local.yaml` = variante de secours compilée sur le PC (CLI ESPHome).

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

## Variante de secours : compilation sur le PC

    cd C:\AItools\Projets\Neural-Networks\esp32
    py -3.12 -m venv .venv-esphome
    .\.venv-esphome\Scripts\Activate.ps1
    pip install esphome
    cd esphome
    esphome run cnn-test-local.yaml

Secrets dans `esphome\secrets.yaml` (ignoré par git).

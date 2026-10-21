# CLAUDE.md — Neural-Networks

Instructions pour les agents qui travaillent dans ce dépôt. À lire en entier avant toute modification.

## Ce qu'est ce dépôt

Le laboratoire d'apprentissage d'Antoine sur les réseaux de neurones, et la partie embarquée de
SpectraSense :

| Dossier | Contenu | État |
|---|---|---|
| `nielsen/` | code du livre de Nielsen porté en Python 3 (MNIST) | historique, ne plus modifier |
| `experiences/` | notebooks d'apprentissage (essais MNIST, `convolution_pas_a_pas.ipynb`) | vivant ; `donnees/` et `resultats/` non versionnés |
| `banc/` | banc d'essai : entraîner, comparer, choisir des réseaux pour l'ESP32 | vivant — voir `banc/README.md` |
| `esp32/` | moteur d'inférence C++, composants ESPHome (dont `audio_udp`, micro INMP441), outils d'export et de test | vivant — voir `esp32/README.md` |
| `outils/`, `mesures_moteur.py`, `verif_gpu.py` | mesures qui ont servi à choisir le moteur du banc | historique |

Le dépôt voisin `spectrasense` (`C:\AItools\Projets\spectrasense`) prépare les données, évalue les
réseaux et contient la chaîne Python de référence. Les deux dépôts sont liés : voir « Parité » ci-dessous.

## Règles qui ne se discutent pas

1. **Ne jamais entraîner de réseau.** Antoine entraîne lui-même dans le banc (`banc/banc_esp.ipynb`).
   Un agent prépare tout jusqu'à l'entraînement (données, configurations, lots, analyses), puis s'arrête.
2. **Le jeu de test ne sert qu'une fois.** On choisit sur la validation. La mesure unique du test de
   SpectraSense a été faite le 14/10/2026 (`spectrasense/tools/evaluation/RESULTATS_TEST.md`) :
   ne plus rien régler sur ces données.
3. **Parité Python ↔ C++.** La chaîne de l'ESP (`esp32/esphome/components/nn_spectrasense/chaine.*`)
   doit rester identique, seconde par seconde, à la chaîne Python de `spectrasense`. Toute modification
   d'un côté se fait aussi de l'autre, puis on relance la parité (commandes plus bas). Un écart est un bug.
4. **Ce qui est commité est ce qui tourne.** HA compile le composant depuis GitHub (`master`, `refresh: 0s`).
   Les headers de poids générés sont commités : un modèle déployé = un commit.
5. **Proposer un plan avant les gros chantiers** (nouvelle architecture, refonte d'un module, changement
   de la chaîne embarquée). Les petites corrections se font directement.

## Conventions

- Langue : français partout (code, commentaires, documents, messages). Réponses concises.
- Commits : l'agent commite, Antoine pousse. Auteur `nErur <antoine.le-comte@hotmail.fr>`,
  `git commit --no-verify`. Un commit = une idée.
- Windows + PowerShell. Venv du dépôt : `.\.venv\Scripts\Activate.ps1` (Python 3.12).
  Compilateur C++ pour les tests sur PC : MSYS2 g++ (`C:\msys64\ucrt64\bin` dans le PATH).
- Ne pas modifier `nielsen/` ni les résultats d'essais historiques.

## Le modèle déployé

`162dcb9fbf` : C8k3-P2-C16k5-P2-C16k5-P2-D24-12, ReLU, biais initialisés à 0, fenêtres de 130 s
(canaux log + lin), jeu `L130_log+lin_c2_s22_aucun`. 7 324 paramètres, 3,4 à 3,7 ms par inférence sur l'ESP32.

- Poids : `banc/resultats/modeles/162dcb9fbf.npz` (journal du banc, non versionné) et une copie
  versionnée dans `esp32/modeles/spectrasense_162dcb9fbf.npz`.
- Headers embarqués : `esp32/esphome/components/nn_spectrasense/model_weights.h` et `test_vectors.h`.
- Les options par défaut de la tâche `spectrasense` dans `banc/config.py` (longueur 128, synth 0.10,
  équilibrage racine) **ne sont pas** celles de ce modèle : passer les options explicitement.

## Vérifications à lancer selon ce qu'on touche

| On modifie… | On relance |
|---|---|
| `banc/moteur.py` | `python -m banc.tests.test_equivalence` |
| `nn_engine.*` (moteur C++) | `python esp32\tests\lancer_tests.py` et `python esp32\tests\lancer_tests_conv.py` |
| `chaine.*`, ou côté spectrasense `adaptateur.py`, `cibles.py` (porte), `automate.py`, `reseau.py` | dans `spectrasense` : `python tools\evaluation\parite_cpp.py --jours 6` puis `--jours 15 --flux 2` |
| le composant ESPHome (`__init__.py`, `nn_spectrasense.*`) | compilation dans HA ; si possible avant, `esphome compile --only-generate` sur une copie locale de `cnn-test.yaml` |
| un nouveau modèle | `export_conv.py`, `lancer_tests_conv.py`, autotest sur la carte (16/16), puis évaluation dans spectrasense |

## Sur la carte de test

- Carte `cnn-test` (ESP-WROOM-32), gérée par l'ESPHome de Home Assistant. La configuration de référence
  est `esp32/esphome/cnn-test.yaml` ; Antoine la recopie dans HA.
- Elle lit la puissance de la ligne arrière-cuisine **via HA** et fait tourner toute la chaîne
  (rafales re-datées, réseau, automate). La carte PZEM de production n'est pas encore touchée.
- Piège connu : l'éditeur d'ESPHome valide la YAML avec une copie en cache du composant. Une option
  nouvelle peut y être refusée alors que « Install » (qui récupère GitHub) passe.

## Documents

- `banc/README.md` — le banc d'essai ; `esp32/README.md` — moteur, composants, déploiement.
- `esp32/ESSAIS.md` — journal des mesures et des décisions (à compléter à chaque décision mesurée).
- `esp32/PLAN.md` — plan initial MNIST (historique).

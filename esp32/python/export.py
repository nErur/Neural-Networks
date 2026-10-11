"""Étape 1 du pipeline ESP32 : traduit un modèle .npz en headers C++.

    python esp32\\python\\export.py                              # dernier modèle 14×14
    python esp32\\python\\export.py esp32\\modeles\\<nom>.npz

Produit, dans esphome/components/nn_mnist/ :
  model_weights.h  poids et biais float32, table des couches, identifiant
  test_vectors.h   images de test choisies + sorties de référence de chaque étage

Les références sont calculées en numpy float32 avec les poids déjà arrondis en
float32 : c'est exactement ce que l'ESP doit retrouver. Le script s'arrête si un
de ses autocontrôles échoue.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
sys.path.insert(0, ICI)
from donnees import charger_u8, pretraiter            # noqa: E402

DOSSIER_COMPOSANT = os.path.join(RACINE, "esp32", "esphome", "components", "nn_mnist")
TOLERANCE = 1e-5
F32 = np.float32


# ---------------------------------------------------------------- calcul de référence (float32)

def sigmoide32(z):
    return (F32(1) / (F32(1) + np.exp(-z))).astype(F32)


def softmax32(z):
    """Softmax stable sur la dernière dimension (on soustrait le maximum)."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(F32)


def relu32(z):
    return np.maximum(z, F32(0)).astype(F32)


def tanh32(z):
    return np.tanh(z).astype(F32)


ACTIV = {"sigmoide": sigmoide32, "softmax": softmax32, "relu": relu32, "tanh": tanh32}


def forward32(couches, x):
    """x : (N, n_in) float32 -> liste des sorties de chaque couche, (N, n_out) float32."""
    sorties, a = [], x
    for W, b, act in couches:
        a = ACTIV[act]((a @ W.T + b).astype(F32))
        sorties.append(a)
    return sorties


def forward_boucles(couches_plates, x):
    """Même calcul avec des boucles et les tableaux APLATIS, indice j*n_in + k,
    exactement comme la boucle C. Sert à attraper une erreur de disposition."""
    sorties, a = [], [F32(v) for v in x]
    for W_plat, b, n_in, n_out, act in couches_plates:
        z = []
        for j in range(n_out):
            s = F32(b[j])
            for k in range(n_in):
                s = F32(s + F32(W_plat[j * n_in + k]) * a[k])
            z.append(s)
        z = np.array(z, dtype=F32)
        a = list(ACTIV[act](z[None, :])[0])
        sorties.append(np.array(a, dtype=F32))
    return sorties


# ---------------------------------------------------------------- écriture C++

def litteral(v):
    """float32 -> littéral C exact : 9 chiffres significatifs suffisent à relire
    exactement un float32 ; suffixe f pour rester en simple précision."""
    s = "%.9g" % float(v)
    if "." not in s and "e" not in s and "n" not in s:
        s += ".0"
    return s + "f"


def tableau_float(nom, valeurs, par_ligne=8):
    v = np.asarray(valeurs, dtype=F32).ravel()
    for x in v:                                        # relecture exacte garantie
        assert F32(float(litteral(x)[:-1])) == x, "littéral non exact : %r" % x
    lignes = [", ".join(litteral(x) for x in v[i:i + par_ligne])
              for i in range(0, len(v), par_ligne)]
    return "static const float %s[%d] = {\n    %s};\n" % (nom, len(v), ",\n    ".join(lignes))


def tableau_u8(nom, valeurs, par_ligne=28):
    v = np.asarray(valeurs, dtype=np.uint8).ravel()
    lignes = [", ".join("%3d" % x for x in v[i:i + par_ligne])
              for i in range(0, len(v), par_ligne)]
    return "static const uint8_t %s[%d] = {\n    %s};\n" % (nom, len(v), ",\n    ".join(lignes))


ENTETE = ("// Généré par esp32/python/export.py le %s à partir de %s.\n"
          "// NE PAS MODIFIER À LA MAIN : relancer export.py.\n")


# ---------------------------------------------------------------- choix des images de test

def choisir_images(pred, conf, marge, y, par_chiffre=2, n_faux=2, n_douteux=2):
    """Indices du jeu de test : deux images sûres et justes par chiffre, puis les
    erreurs les plus assurées et les bonnes réponses les plus hésitantes."""
    choix = []
    for c in range(10):
        ok = np.where((y == c) & (pred == c))[0]
        surs = [i for i in ok if conf[i] > 0.9][:par_chiffre]       # dans l'ordre du jeu de test
        reste = [i for i in ok[np.argsort(-conf[ok])] if i not in surs]
        choix += (surs + reste)[:par_chiffre]
    faux = np.where(pred != y)[0]
    choix += list(faux[np.argsort(-conf[faux])][:n_faux])
    justes = np.where(pred == y)[0]
    choix += list(justes[np.argsort(marge[justes])][:n_douteux])
    return choix


# ---------------------------------------------------------------- programme

def dernier_modele():
    fichiers = glob.glob(os.path.join(RACINE, "esp32", "modeles", "mnist_14x14_*.npz"))
    if not fichiers:
        sys.exit("Aucun modèle 14×14 dans esp32/modeles : lancer train.py d'abord.")
    return max(fichiers, key=os.path.getmtime)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("modele", nargs="?", default=None)
    args = p.parse_args()
    chemin = os.path.abspath(args.modele or dernier_modele())
    z = np.load(chemin)
    meta = json.loads(str(z["meta"]))
    tailles, acts = meta["tailles"], meta["activations"]
    pre = meta["pretraitement"]
    n_couches = len(tailles) - 1
    couches = [(z["W%d" % i].astype(F32), z["b%d" % i].astype(F32), acts[i])
               for i in range(n_couches)]
    for i, (W, b, _) in enumerate(couches):
        assert W.shape == (tailles[i + 1], tailles[i]) and b.shape == (tailles[i + 1],)
    print("Modèle %s : %s, test %.2f %% (float64)" % (meta["nom"], tailles, meta["precision_test"]))

    # --- références float32 sur tout le jeu de test
    d = charger_u8()
    img_te, y_te = d["test"]
    x_te = pretraiter(img_te, pre["facteur"])
    sorties = forward32(couches, x_te)
    proba = sorties[-1]
    pred = proba.argmax(axis=1)
    tri = np.sort(proba, axis=1)
    conf, marge = tri[:, -1], tri[:, -1] - tri[:, -2]
    p32 = 100.0 * np.mean(pred == y_te)

    # contrôle 1 : float32 vs float64 (Python)
    a = x_te.astype(np.float64)
    for i in range(n_couches):
        zz = a @ z["W%d" % i].T + z["b%d" % i]
        a = {"sigmoide": lambda v: 1 / (1 + np.exp(-v)), "relu": lambda v: np.maximum(v, 0),
             "tanh": np.tanh}.get(acts[i], lambda v: v)(zz)
    desaccords = int(np.sum(a.argmax(axis=1) != pred))
    print("Contrôle float32 : précision %.2f %%, %d prédiction(s) différente(s) de float64 sur 10 000"
          % (p32, desaccords))

    # --- images de test
    choix = choisir_images(pred, conf, marge, y_te)
    print("Images de test : %d (indices %s)" % (len(choix), " ".join(str(int(i)) for i in choix)))

    # contrôle 2 : boucles sur tableaux aplatis (comme le C) vs numpy
    plates = [(W.ravel(), b, W.shape[1], W.shape[0], act) for W, b, act in couches]
    ecart_max = 0.0
    for i in choix:
        ref = [s[i] for s in sorties]
        boucle = forward_boucles(plates, x_te[i])
        for e, (r, bcl) in enumerate(zip(ref, boucle)):
            ecart_max = max(ecart_max, float(np.max(np.abs(r - bcl))))
        if int(np.argmax(boucle[-1])) != int(pred[i]):
            sys.exit("ÉCHEC : la boucle aplatie ne donne pas la même prédiction (image %d)" % i)
    if ecart_max > TOLERANCE:
        sys.exit("ÉCHEC : écart boucle aplatie / numpy = %.2e > %.0e" % (ecart_max, TOLERANCE))
    print("Contrôle disposition j*n_in+k : écart max %.2e (tolérance %.0e)" % (ecart_max, TOLERANCE))

    # --- identifiant du modèle
    h = hashlib.sha1()
    for W, b, _ in couches:
        h.update(W.tobytes()); h.update(b.tobytes())
    ident = "%s | %s | %.2f%% | %s" % (meta["nom"], h.hexdigest()[:8], p32, meta["date"][:10])
    quand = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    source = os.path.relpath(chemin, RACINE).replace("\\", "/")

    # --- model_weights.h
    act_c = {"sigmoide": "nn::Activation::kSigmoid", "softmax": "nn::Activation::kSoftmax",
             "relu": "nn::Activation::kReLU", "tanh": "nn::Activation::kTanh"}
    out = [ENTETE % (quand, source), "#pragma once\n\n#include \"nn_engine.h\"\n\nnamespace nn_model {\n\n"]
    out.append("constexpr int kNumLayers = %d;\n" % n_couches)
    out.append("constexpr int kInputSize = %d;   // %d×%d\n" % (tailles[0], 28 // pre["facteur"], 28 // pre["facteur"]))
    out.append("constexpr int kOutputSize = %d;\n" % tailles[-1])
    out.append("constexpr int kMaxWidth = %d;    // plus grande couche : taille des tampons\n\n" % max(tailles))
    for i, (W, b, act) in enumerate(couches):
        out.append("// couche %d : %d -> %d, %s\n" % (i, W.shape[1], W.shape[0], act))
        out.append(tableau_float("kW%d" % i, W))
        out.append(tableau_float("kB%d" % i, b) + "\n")
    out.append("static const nn::Layer kLayers[kNumLayers] = {\n")
    for i, (W, b, act) in enumerate(couches):
        out.append("    {%d, %d, kW%d, kB%d, %s},\n" % (W.shape[1], W.shape[0], i, i, act_c[act]))
    out.append("};\n\n")
    out.append("static const nn::Model kModel = {\n    \"%s\",\n    kNumLayers, kLayers, %d, %d, %d};\n\n"
               % (ident, pre["cote_brut"], pre["facteur"], pre["diviseur"]))
    out.append("}  // namespace nn_model\n")
    with open(os.path.join(DOSSIER_COMPOSANT, "model_weights.h"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(out))

    # --- test_vectors.h
    out = [ENTETE % (quand, source),
           "#pragma once\n\n#include <cstdint>\n\n#include \"model_weights.h\"\n\nnamespace nn_model {\n\n",
           "// Pour chaque image : image brute, entrée réduite, sortie de chaque couche,\n"
           "// toutes calculées en numpy float32 (référence à retrouver sur l'ESP).\n"
           "struct TestVector {\n  uint8_t label;          // étiquette MNIST\n"
           "  uint8_t py_pred;        // prédiction Python (le C doit donner la même)\n"
           "  uint16_t test_index;    // indice dans le jeu de test MNIST\n"
           "  const uint8_t *image;   // 28×28 brut\n  const float *input;     // après prétraitement\n"
           "  const float *outputs[kNumLayers];\n};\n\n"]
    for n, i in enumerate(choix):
        out.append("// image %d : test[%d], étiquette %d, Python %d (confiance %.3f)\n"
                   % (n, i, y_te[i], pred[i], conf[i]))
        out.append(tableau_u8("kTv%dImage" % n, img_te[i]))
        out.append(tableau_float("kTv%dInput" % n, x_te[i]))
        for c in range(n_couches):
            out.append(tableau_float("kTv%dL%d" % (n, c), sorties[c][i]))
        out.append("\n")
    out.append("static const TestVector kTestVectors[] = {\n")
    for n, i in enumerate(choix):
        lc = ", ".join("kTv%dL%d" % (n, c) for c in range(n_couches))
        out.append("    {%d, %d, %d, kTv%dImage, kTv%dInput, {%s}},\n"
                   % (y_te[i], pred[i], i, n, n, lc))
    out.append("};\nconstexpr int kNumTestVectors = %d;\n\n}  // namespace nn_model\n" % len(choix))
    with open(os.path.join(DOSSIER_COMPOSANT, "test_vectors.h"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(out))

    for nom in ("model_weights.h", "test_vectors.h"):
        taille = os.path.getsize(os.path.join(DOSSIER_COMPOSANT, nom))
        print("Écrit : esp32/esphome/components/nn_mnist/%s (%.0f Ko)" % (nom, taille / 1024))
    print("Identifiant : %s" % ident)


if __name__ == "__main__":
    main()

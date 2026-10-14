"""Réseau du banc (convolution 1D, SpectraSense) -> headers C++ pour le moteur de l'ESP, et vecteurs de test.

    python esp32\\python\\export_conv.py 162dcb9fbf [--sortie esp32\\tests\\spectrasense] [--par-classe 2]

Produit dans --sortie :
  model_weights.h  poids et biais float32, table des couches (dense / conv1d / maxpool1d)
  test_vectors.h   fenêtres de validation choisies + sortie de référence de chaque couche
Références : numpy float32 avec les poids arrondis en float32 (ce que l'ESP doit retrouver).
Autocontrôles (le script s'arrête s'ils échouent) :
  1. la propagation numpy float32 = celle du moteur du banc (Paquet.proba) ;
  2. une propagation en BOUCLES sur tableaux APLATIS, avec les indices du C, = numpy.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
sys.path.insert(0, RACINE)
sys.path.insert(0, ICI)
from banc.essai import charger_reseau                  # noqa: E402
from banc.moteur import plan_couches                   # noqa: E402
from export import litteral, tableau_float             # noqa: E402

F32 = np.float32
TOLERANCE_REL = 1e-5


def act32(z, act):
    if act == "relu":
        return np.maximum(z, F32(0)).astype(F32)
    if act == "sigmoide":
        return (F32(1) / (F32(1) + np.exp(-z))).astype(F32)
    if act == "tanh":
        return np.tanh(z).astype(F32)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(F32)


def forward32(plan, params, X):
    """X (N, C, L) -> sorties de chaque couche, aplaties (N, n_out), float32."""
    a, sorties = X.astype(F32), []
    for c, p in zip(plan, params):
        if c["type"] == "conv1d":
            W, b = p
            v = np.lib.stride_tricks.sliding_window_view(a, W.shape[2], axis=2)[:, :, ::c["pas"], :]
            a = act32(np.einsum("nclk,fck->nfl", v, W).astype(F32) + b[None, :, None], c["act"])
        elif c["type"] == "maxpool1d":
            t, L2 = c["taille"], a.shape[2] // c["taille"]
            a = a[:, :, :L2 * t].reshape(a.shape[0], a.shape[1], L2, t).max(-1)
        else:
            W, b = p
            a = act32((a.reshape(a.shape[0], -1) @ W.T + b).astype(F32), c["act"])
        sorties.append(a.reshape(a.shape[0], -1).copy())
    return sorties


def forward_boucles(plan, params, x):
    """Une fenêtre, en boucles et indices du C (canal par canal, W aplatis)."""
    C, L = x.shape
    a = [F32(v) for v in x.ravel()]
    sorties = []
    for c, p in zip(plan, params):
        if c["type"] == "conv1d":
            W, b = p
            F, _, k = W.shape
            s, Lo = c["pas"], c["sortie"][1]
            w = W.ravel()
            z = []
            for f in range(F):
                for o in range(Lo):
                    acc = F32(b[f])
                    for cc in range(C):
                        for j in range(k):
                            acc = F32(acc + F32(w[(f * C + cc) * k + j]) * a[cc * L + o * s + j])
                    z.append(acc)
            a = list(act32(np.array(z, dtype=F32), c["act"]))
            C, L = F, Lo
        elif c["type"] == "maxpool1d":
            t, Lo = c["taille"], c["sortie"][1]
            a = [max(a[cc * L + o * t + i] for i in range(t)) for cc in range(C) for o in range(Lo)]
            L = Lo
        else:
            W, b = p
            n_out, n_in = W.shape
            w = W.ravel()
            z = []
            for j in range(n_out):
                acc = F32(b[j])
                for kk in range(n_in):
                    acc = F32(acc + F32(w[j * n_in + kk]) * a[kk])
                z.append(acc)
            a = list(act32(np.array(z, dtype=F32)[None, :], c["act"])[0])
        sorties.append(np.array(a, dtype=F32))
    return sorties


def ecart_rel(a, b):
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("id")
    ap.add_argument("--sortie", default=os.path.join(RACINE, "esp32", "tests", "spectrasense"))
    ap.add_argument("--par-classe", type=int, default=2)
    ap.add_argument("--espace", default="nn_model", help="espace de noms C++ des headers")
    ap.add_argument("--inclure-moteur", default="nn_engine.h",
                    help="chemin d'inclusion du moteur (composant ESPHome : esphome/components/nn_mnist/nn_engine.h)")
    a = ap.parse_args()

    paquet, cfg, tache = charger_reseau(a.id)
    C, L = tache.forme_entree
    plan = plan_couches(cfg, tache.forme_entree, len(tache.classes))
    z = np.load(os.path.join(RACINE, "banc", "resultats", "modeles", a.id + ".npz"))
    params = [None if c["type"] == "maxpool1d" else (z["W%d" % i].astype(F32), z["b%d" % i].astype(F32))
              for i, c in enumerate(plan)]
    X, y, _ = tache.donnees("validation")
    sorties = forward32(plan, params, X)
    proba = sorties[-1]
    pred = proba.argmax(1)
    print("Réseau %s : %s ; validation %.2f %% (numpy float32)" % (a.id, cfg["couches"] and
          " → ".join(str(c["sortie"]) for c in plan), 100 * np.mean(pred == y)))

    # contrôle 1 : = moteur du banc
    ref_banc = paquet.proba(X[:2000])[0]
    e1 = float(np.max(np.abs(ref_banc - proba[:2000])))
    print("Contrôle 1 (banc) : écart max %.1e" % e1)
    if e1 > 1e-4:
        sys.exit("ÉCHEC : la propagation float32 ne reproduit pas le banc")

    # choix des fenêtres : par classe, les plus sûres et justes ; puis les erreurs les plus assurées
    conf = proba.max(1)
    choix = []
    for k in range(len(tache.classes)):
        ok = np.flatnonzero((y == k) & (pred == k))
        choix += list(ok[np.argsort(-conf[ok])][:a.par_classe])
    faux = np.flatnonzero(pred != y)
    choix += list(faux[np.argsort(-conf[faux])][:4])

    # contrôle 2 : boucles aplaties (indices du C) = numpy
    e2 = 0.0
    for i in choix[:6] + choix[-2:]:
        bcl = forward_boucles(plan, params, X[i])
        e2 = max(e2, max(ecart_rel(bcl[l], sorties[l][i]) for l in range(len(plan))))
    print("Contrôle 2 (boucles aplaties) : écart relatif max %.1e" % e2)
    if e2 > TOLERANCE_REL:
        sys.exit("ÉCHEC : la disposition aplatie ne donne pas le même calcul")

    # ---- écriture
    os.makedirs(a.sortie, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    h = hashlib.sha1(b"".join(p[0].tobytes() + p[1].tobytes() for p in params if p)).hexdigest()[:8]
    ident = "spectrasense_%s | %s | val %.2f%% | %s" % (a.id, h, 100 * np.mean(pred == y), date[:10])
    entete = ("// Généré par esp32/python/export_conv.py le %s à partir de banc/resultats/modeles/%s.npz.\n"
              "// NE PAS MODIFIER À LA MAIN : relancer export_conv.py.\n" % (date, a.id))
    tailles = [C * L] + [int(np.prod(c["sortie"])) for c in plan]
    out = [entete, "#pragma once\n\n#include \"%s\"\n\nnamespace %s {\n\n" % (a.inclure_moteur, a.espace),
           "constexpr int kNumLayers = %d;\n" % len(plan),
           "constexpr int kInputSize = %d;   // %d canaux × %d s\n" % (C * L, C, L),
           "constexpr int kOutputSize = %d;\n" % tailles[-1],
           "constexpr int kMaxWidth = %d;    // plus grande sortie : taille des tampons\n\n" % max(tailles)]
    lignes = []
    forme = (C, L)
    acts = {"relu": "kReLU", "sigmoide": "kSigmoid", "tanh": "kTanh", "softmax": "kSoftmax"}
    for i, (c, p) in enumerate(zip(plan, params)):
        n_in, n_out = int(np.prod(forme)), int(np.prod(c["sortie"]))
        if c["type"] == "maxpool1d":
            out.append("// couche %d : max-pooling %d, %s -> %s\n\n" % (i, c["taille"], forme, c["sortie"]))
            lignes.append("    {%d, %d, nullptr, nullptr, nn::Activation::kReLU, nn::LayerType::kMaxPool1d, %d, %d, %d, %d, %d, %d},"
                          % (n_in, n_out, forme[0], forme[1], c["sortie"][0], c["sortie"][1], c["taille"], c["taille"]))
        else:
            out.append("// couche %d : %s, %s -> %s, %s\n" % (i, c["type"], forme, c["sortie"], c["act"]))
            out.append(tableau_float("kW%d" % i, p[0]))
            out.append(tableau_float("kB%d" % i, p[1]) + "\n")
            if c["type"] == "conv1d":
                lignes.append("    {%d, %d, kW%d, kB%d, nn::Activation::%s, nn::LayerType::kConv1d, %d, %d, %d, %d, %d, %d},"
                              % (n_in, n_out, i, i, acts[c["act"]], forme[0], forme[1], c["sortie"][0], c["sortie"][1],
                                 p[0].shape[2], c["pas"]))
            else:
                lignes.append("    {%d, %d, kW%d, kB%d, nn::Activation::%s}," % (n_in, n_out, i, i, acts[c["act"]]))
        forme = c["sortie"]
    out.append("static const nn::Layer kLayers[kNumLayers] = {\n%s\n};\n\n" % "\n".join(lignes))
    out.append("static const nn::Model kModel = {\n    \"%s\",\n    kNumLayers, kLayers, 0, 1, 1, kInputSize};\n\n"
               % ident)
    out.append("static const char *const kClasses[kOutputSize] = {%s};\n\n}  // namespace %s\n"
               % (", ".join('"%s"' % c for c in tache.classes), a.espace))
    with open(os.path.join(a.sortie, "model_weights.h"), "w", encoding="utf-8") as f:
        f.write("".join(out))

    tv = [entete, "#pragma once\n\n#include <cstdint>\n\n#include \"model_weights.h\"\n\nnamespace %s {\n\n" % a.espace,
          "// Pour chaque fenêtre de validation : entrée (canaux × secondes) et sortie de chaque couche,\n"
          "// calculées en numpy float32 (référence à retrouver par le moteur C++).\n"
          "struct TestVector {\n  uint8_t label;\n  uint8_t py_pred;\n  uint32_t index;\n"
          "  const float *input;\n  const float *outputs[kNumLayers];\n};\n\n"]
    for t, i in enumerate(choix):
        tv.append("// fenêtre %d : validation[%d], classe %s, Python %s (confiance %.3f)\n"
                  % (t, i, tache.classes[y[i]], tache.classes[pred[i]], conf[i]))
        tv.append(tableau_float("kTv%dIn" % t, X[i]))
        for l in range(len(plan)):
            tv.append(tableau_float("kTv%dOut%d" % (t, l), sorties[l][i]))
        tv.append("\n")
    tv.append("constexpr int kNumTestVectors = %d;\nstatic const TestVector kTestVectors[kNumTestVectors] = {\n" % len(choix))
    for t, i in enumerate(choix):
        tv.append("    {%d, %d, %d, kTv%dIn, {%s}},\n" % (y[i], pred[i], i, t,
                                                         ", ".join("kTv%dOut%d" % (t, l) for l in range(len(plan)))))
    tv.append("};\n\n}  // namespace %s\n" % a.espace)
    with open(os.path.join(a.sortie, "test_vectors.h"), "w", encoding="utf-8") as f:
        f.write("".join(tv))
    print("Écrit : %s (%d couches, %d fenêtres de test) ; identifiant « %s »"
          % (a.sortie, len(plan), len(choix), ident))


if __name__ == "__main__":
    main()

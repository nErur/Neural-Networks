"""Moteur C++ avec convolution 1D (SpectraSense) : compile et lance le test sur le PC, puis « teste le test ».

    python esp32\\python\\export_conv.py <id>        (fabrique tests/spectrasense/*.h)
    python esp32\\tests\\lancer_tests_conv.py

1. Le moteur tel quel : toutes les fenêtres doivent être conformes, couche par couche.
2. Des copies du moteur avec un bug volontaire (« mutations ») : chacune doit faire échouer
   le test, au bon étage. Le test MNIST (lancer_tests.py) reste valable pour les couches denses.
"""
import os
import shutil
import subprocess
import sys
import tempfile

from lancer_tests import COMPOSANT, compilateur, etages_en_echec

ICI = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(ICI, "spectrasense")

MUTATIONS = [
    ("noyau lu à l'envers (vraie convolution au lieu d'une corrélation)",
     "for (int j = 0; j < k; j++) z += wc[j] * xc[j];", "for (int j = 0; j < k; j++) z += wc[k - 1 - j] * xc[j];",
     "couche 0"),
    ("sortie de convolution rangée seconde par seconde au lieu de canal par canal",
     "out[f * Lo + o] = z;", "out[o * F + f] = z;", "couche 0"),
    ("pooling qui ignore la dernière seconde de chaque paire",
     "for (int i = 1; i < t; i++) m = x[i] > m ? x[i] : m;", "for (int i = 1; i < t - 1; i++) m = x[i] > m ? x[i] : m;",
     "couche 1"),
    ("décalage d'une seconde dans la fenêtre de convolution",
     "const float *xc = in + c * Lin + o * s;", "const float *xc = in + c * Lin + o * s + (o + 1 < Lo ? 1 : 0);",
     "couche 0"),
]


def compiler_et_lancer(cxx, dossier_moteur, sortie):
    exe = os.path.join(sortie, "test_conv.exe" if os.name == "nt" else "test_conv")
    cmd = [cxx, "-O2", "-Wall", "-Wextra", "-std=c++17", "-I", dossier_moteur,
           os.path.join(TESTS, "test_engine_conv.cpp"), os.path.join(dossier_moteur, "nn_engine.cpp"), "-o", exe]
    comp = subprocess.run(cmd, capture_output=True, text=True)
    if comp.returncode:
        return None, comp.stderr
    run = subprocess.run([exe], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return run.returncode, run.stdout


def main():
    if not os.path.exists(os.path.join(TESTS, "test_vectors.h")):
        sys.exit("tests/spectrasense/test_vectors.h absent : lancer d'abord esp32/python/export_conv.py <id>")
    cxx = compilateur()
    with tempfile.TemporaryDirectory() as tmp:
        print("=== 1. Moteur tel quel (%s) ===" % cxx)
        code, sortie = compiler_et_lancer(cxx, COMPOSANT, tmp)
        print(sortie)
        if code != 0:
            sys.exit("Le moteur ne passe pas son test.")
        print("=== 2. Test du test : mutations ===")
        source = open(os.path.join(COMPOSANT, "nn_engine.cpp"), encoding="utf-8").read()
        bilan = True
        for nom, avant, apres, attendu in MUTATIONS:
            assert source.count(avant) == 1, "motif introuvable : %s" % avant
            mut = os.path.join(tmp, "mutant")
            os.makedirs(mut, exist_ok=True)
            shutil.copy(os.path.join(COMPOSANT, "nn_engine.h"), mut)
            with open(os.path.join(mut, "nn_engine.cpp"), "w", encoding="utf-8") as f:
                f.write(source.replace(avant, apres))
            code, sortie = compiler_et_lancer(cxx, mut, tmp)
            if code is None:
                print("  [ÉCHEC] %s : ne compile pas\n%s" % (nom, sortie))
                bilan = False
                continue
            etages = etages_en_echec(sortie)
            detecte = code != 0 and etages and etages[0] == attendu
            bilan &= bool(detecte)
            print("  [%s] %s -> premier étage en échec : %s (attendu : %s)"
                  % ("OK " if detecte else "ÉCHEC", nom, etages[0] if etages else "aucun", attendu))
        print("\n%s" % ("SUCCÈS : le moteur passe, et chaque bug est détecté au bon étage."
                        if bilan else "ÉCHEC : le test ne détecte pas tous les bugs."))
        return 0 if bilan else 1


if __name__ == "__main__":
    sys.exit(main())

"""Étape 3 : compile et lance le test du moteur C++ sur le PC, puis « teste le test ».

    python esp32\\tests\\lancer_tests.py            (il faut g++ ou clang++ dans le PATH)

1. Le moteur tel qu'il est : les 24 images doivent être conformes.
2. Des copies du moteur où l'on introduit volontairement un bug (« mutations ») :
   chacune doit faire échouer le test, et à l'étage attendu. Si une mutation
   passe, c'est le test qui est trop faible.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ICI = os.path.dirname(os.path.abspath(__file__))
COMPOSANT = os.path.join(os.path.dirname(ICI), "esphome", "components", "nn_mnist")

# (nom, texte à remplacer dans nn_engine.cpp, remplacement, étage qui doit échouer)
MUTATIONS = [
    ("poids transposés (k*n_out+j au lieu de j*n_in+k)",
     "z += w[k] * in[k];", "z += layer.W[k * layer.n_out + j] * in[k];", "couche 0"),
    ("division par 256 au lieu de 255",
     "(float) sum / (float) m.divisor;", "(float) sum / (float) (m.factor * m.factor * 256);", "entrée"),
    ("biais oublié",
     "float z = layer.b[j];", "float z = 0.0f;", "couche 0"),
]


def compilateur():
    for c in ("g++", "clang++"):
        if shutil.which(c):
            return c
    sys.exit("Aucun compilateur C++ trouvé (g++ ou clang++). Voir esp32/README.md.")


def compiler_et_lancer(cxx, dossier_moteur, sortie):
    exe = os.path.join(sortie, "test_engine.exe" if os.name == "nt" else "test_engine")
    cmd = [cxx, "-O2", "-Wall", "-Wextra", "-std=c++17", "-I", dossier_moteur,
           os.path.join(ICI, "test_engine.cpp"), os.path.join(dossier_moteur, "nn_engine.cpp"),
           "-o", exe]
    comp = subprocess.run(cmd, capture_output=True, text=True)
    if comp.returncode:
        return None, comp.stderr
    run = subprocess.run([exe], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return run.returncode, run.stdout


def etages_en_echec(sortie):
    m = re.search(r"Échecs par étage : (.*)", sortie)
    return [nom.strip() for nom, n in re.findall(r"([^,]+?) (\d+)(?:,|$)", m.group(1)) if int(n)]


def main():
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
            for nom_h in os.listdir(COMPOSANT):              # tous les headers, isolés de l'original
                if nom_h.endswith(".h"):
                    shutil.copy(os.path.join(COMPOSANT, nom_h), mut)
            with open(os.path.join(mut, "nn_engine.cpp"), "w", encoding="utf-8") as f:
                f.write(source.replace(avant, apres))
            code, sortie = compiler_et_lancer(cxx, mut, tmp)
            if code is None:
                print("  [ÉCHEC] %s : ne compile pas\n%s" % (nom, sortie)); bilan = False; continue
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

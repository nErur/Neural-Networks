// Rejoue une suite de mesures (t, P) dans la chaîne C++ de l'ESP et écrit, pour chaque seconde,
// la grille, la porte, les probabilités du réseau et les états publiés.
//   rejouer mesures.csv sortie.csv      (mesures : « t,p » par ligne, t en secondes)
// Comparé à la chaîne Python par spectrasense/tools/evaluation/parite_cpp.py.
#include <cstdio>
#include <cstdlib>

#include "esphome/components/nn_spectrasense/chaine.h"
#include "esphome/components/nn_spectrasense/model_weights.h"   // réseau embarqué (espace nn_ss)

static float buf_a[nn_ss::kMaxWidth], buf_b[nn_ss::kMaxWidth];

// Chaque case terminée passe dans la chaîne ; la sortie est écrite aussitôt.
struct Ecrivain : ss::Recepteur {
  ss::Chaine *chaine;
  FILE *out;
  void recevoir(const ss::Case &c) override {
    const ss::Sortie s = chaine->traiter(c);
    fprintf(out, "%.0f,%.17g,%d,%d,%d", s.t, s.p_moy, s.valide, s.porte, s.reseau);
    for (int k = 0; k < ss::kClasses; k++) fprintf(out, ",%.9g", s.q[k]);
    fprintf(out, ",%d,%d,%d,%d,%d\n", s.machine, s.activite, s.grille_pain, s.bouilloire, s.inconnu);
  }
};

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage : rejouer mesures.csv sortie.csv [seuil tenue seuil_pic]\n");
    return 2;
  }
  ss::Reglages r;
  if (argc >= 6) {
    r.seuil = atof(argv[3]);
    r.tenue = atoi(argv[4]);
    r.seuil_pic = atof(argv[5]);
  }
  FILE *in = fopen(argv[1], "r"), *out = fopen(argv[2], "w");
  if (!in || !out) return 2;
  ss::Adaptateur adapt;
  ss::Chaine chaine(nn_ss::kModel, buf_a, buf_b, r);
  Ecrivain ecrivain;
  ecrivain.chaine = &chaine;
  ecrivain.out = out;
  double t, p;
  fprintf(out, "t,p_moy,valide,porte,reseau");
  for (int k = 0; k < ss::kClasses; k++) fprintf(out, ",q%d", k);
  fprintf(out, ",machine,activite,grille_pain,bouilloire,inconnu\n");
  while (fscanf(in, "%lf,%lf", &t, &p) == 2) adapt.pousser(t, p, ecrivain);
  fclose(in);
  fclose(out);
  return 0;
}

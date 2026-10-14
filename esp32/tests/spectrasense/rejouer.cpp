// Rejoue une suite de mesures (t, P) dans la chaîne C++ de l'ESP et écrit, pour chaque seconde,
// la grille, la porte, les probabilités du réseau et les états publiés.
//   rejouer mesures.csv sortie.csv [seuil tenue seuil_pic [retard [etat_initial]]]
//     etat_initial : inconnu (défaut), arret, veille, maintien (état long de la machine au départ).
//     mesures : « t,p » par ligne, t en secondes.
//     sans retard : t déjà re-datés (debrouiller_rafales), remis directement à l'adaptateur ;
//     avec retard : t = heures d'ARRIVÉE brutes de HA, traitées comme sur la carte (ss::EntreeHa :
//     rafales re-datées en flux, horloge en retard de `retard` s, boucle simulée toutes les 0,1 s).
// Comparé à la chaîne Python par spectrasense/tools/evaluation/parite_cpp.py.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "esphome/components/nn_spectrasense/chaine.h"
#include "esphome/components/nn_spectrasense/model_weights.h"   // réseau embarqué (espace nn_ss)

static float buf_a[nn_ss::kMaxWidth], buf_b[nn_ss::kMaxWidth];

// Chaque case terminée passe dans la chaîne ; la sortie est écrite aussitôt.
struct Ecrivain : ss::Recepteur {
  ss::Chaine *chaine;
  FILE *out;
  double maintenant = 0;                                // flux : instant simulé de l'émission
  void recevoir(const ss::Case &c) override {
    const ss::Sortie s = chaine->traiter(c);
    fprintf(out, "%.0f,%.17g,%d,%d,%d", s.t, s.p_moy, s.valide, s.porte, s.reseau);
    for (int k = 0; k < ss::kClasses; k++) fprintf(out, ",%.9g", s.q[k]);
    fprintf(out, ",%d,%d,%d,%d,%d,%.3f\n", s.machine, s.activite, s.grille_pain, s.bouilloire, s.inconnu, maintenant);
  }
};

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage : rejouer mesures.csv sortie.csv [seuil tenue seuil_pic [retard]]\n");
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
  fprintf(out, ",machine,activite,grille_pain,bouilloire,inconnu,emis\n");
  if (argc >= 8) {
    const std::string e = argv[7];
    const ss::Long l = e == "arret" ? ss::Long::kArret : e == "veille" ? ss::Long::kVeille
                       : e == "maintien" ? ss::Long::kMaintien : ss::Long::kInconnu;
    long pos = ftell(in);
    if (fscanf(in, "%lf,%lf", &t, &p) == 2) chaine.machine().forcer_long(l, t);
    fseek(in, pos, SEEK_SET);
  }
  if (argc >= 7) {                                     // comme sur la carte
    ss::EntreeHa entree(atof(argv[6]));
    double tick = 0;
    bool premier = true;
    while (fscanf(in, "%lf,%lf", &t, &p) == 2) {
      if (premier) tick = t, premier = false;
      for (; tick < t; tick += 0.1) ecrivain.maintenant = tick, entree.horloge(tick, ecrivain);
      ecrivain.maintenant = t;
      entree.mesure(t, p, ecrivain);
    }
    entree.vider(ecrivain);
    fprintf(stderr, "lectures recalées (arrivées après fermeture de leur case) : %d\n", entree.recalees());
  } else {
    while (fscanf(in, "%lf,%lf", &t, &p) == 2) ecrivain.maintenant = t, adapt.pousser(t, p, ecrivain);
  }
  fclose(in);
  fclose(out);
  return 0;
}

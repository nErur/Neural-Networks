#pragma once
// Chaîne SpectraSense, en C++ pur (ni ESPHome ni Arduino, aucune allocation) : le même code est
// testé sur le PC contre la version Python (spectrasense/tools/evaluation/parite_cpp.py).
//
//   mesures PZEM (t, P) -> Adaptateur : une case par seconde (p_moy, x, valide)   [adaptateur.py]
//                       -> porte : P > 20 W dans les 5 dernières cases              [cibles.py]
//                       -> fenêtre des 130 dernières cases (log, lin) -> réseau     [reseau.py]
//                       -> automate : états HA de la machine et des appareils       [automate.py]
// Toutes les constantes et règles sont celles des fichiers Python indiqués.

#include <cstdint>

#include "esphome/components/nn_mnist/nn_engine.h"

namespace ss {

// ------------------------------------------------------------------ adaptateur (grille à 1 s)
struct Case {
  double t;        // début de la case (s)
  double p_moy;    // moyenne de la puissance tenue sur [t, t + 1)
  float x;         // ln(1 + max(p_moy, 0)) / ln(3001), en float32 comme le dataset
  bool valide;
};

// Reçoit les cases terminées, une par une (après une longue coupure, il peut y en avoir des milliers).
class Recepteur {
 public:
  virtual void recevoir(const Case &c) = 0;
  virtual ~Recepteur() = default;
};

class Adaptateur {
 public:
  // Ajoute une mesure ; les cases terminées sont remises au récepteur. Renvoie leur nombre.
  int pousser(double t, double p, Recepteur &r);
  // Ferme les cases jusqu'à t même sans mesure (lecture PZEM ratée).
  int avancer_horloge(double t, Recepteur &r);
  double curseur() const { return t_curseur_; }
  bool demarre() const { return demarre_; }

 private:
  int fermer_jusqua_(double t, Recepteur &r);
  bool demarre_{false};
  double t_mesure_{0}, p_{0}, t_case_{0}, t_curseur_{0}, acc_{0};
};

// ------------------------------------------------------------------ entrée depuis Home Assistant
// Les lectures relayées par HA sont datées à leur RÉCEPTION ; ~2 % arrivent en rafale (quelques ms
// d'écart) alors qu'elles ont été mesurées ~1 s l'une après l'autre. Même règle que
// adaptateur.py : debrouiller_rafales, appliquée en flux : une rafale est close quand 0,3 s passe
// sans nouvelle lecture, puis re-datée à rebours (1 s par lecture, sans repasser avant la lecture
// précédente) et remise à l'adaptateur.
// L'horloge (cases fermées sans nouvelle lecture) avance avec `retard` secondes de retard, pour
// qu'une rafale tardive retrouve sa place dans la grille. Pendant l'activité, ce sont les lectures
// qui ferment les cases : le retard ne joue que dans les silences (HA ne republie pas une valeur
// identique). Sur la carte PZEM (lectures datées sur place), retard = 0 et pas de rafale.
class EntreeHa {
 public:
  explicit EntreeHa(double retard) : retard_(retard) {}
  void mesure(double t, double p, Recepteur &r);   // lecture reçue à t (s)
  void horloge(double t, Recepteur &r);            // à appeler souvent (loop)
  void vider(Recepteur &r);                        // remet la rafale en attente (fin de fichier)
  int recalees() const { return recalees_; }       // lectures arrivées après fermeture de leur case

 private:
  static constexpr int kMax = 16;
  double retard_;
  Adaptateur a_;
  double t_[kMax]{}, p_[kMax]{};
  int n_{0};
  double prec_{0};
  bool a_prec_{false};
  int recalees_{0};
};

// ------------------------------------------------------------------ automate
enum class Long : uint8_t { kArret, kVeille, kMaintien, kInconnu };
enum class Cycle : uint8_t { kAucun, kAllumage, kCafe, kPic, kExtinction };
constexpr int kAct = 9;          // repos + 8 phases (ordre d'automate.py : ACT)
// états HA de la machine (ordre d'automate.py : HA_MACHINE)
enum Ha : uint8_t { kArret = 0, kVeilleHa, kMaintienHa, kChauffe, kRegulation, kBroyage, kExtraction, kFin };

struct Reglages {
  double seuil = 0.95, seuil_pic = 0.5;
  int tenue = 1;
};

class AutomateMachine {
 public:
  explicit AutomateMachine(const Reglages &r) : r_(r) {}
  uint8_t pas(double t, const double q[kAct], bool porte);   // renvoie l'état HA
  void annuler_cycle(double t);
  uint8_t ha() const;
  uint8_t activite() const { return act_; }

 private:
  void publier_(double t, int nouv);
  void minuterie_(double t);
  int cible_(int i) const;       // fusion : indice cible, ou -1 = activité en cours
  Reglages r_;
  Long long_{Long::kInconnu};
  int act_{0}, cand_{0}, n_{0};
  Cycle cycle_{Cycle::kAucun};
  bool a_activite_{false}, a_pic_{false}, a_pic_vu_{false}, porte_prec_{false}, a_debut_{false};
  double t_activite_{0}, t_pic_{0}, pic_vu_{0}, t_ouverture_{0};
  double debut_t_{0}, debut_t_activite_{0};
  Long debut_long_{Long::kInconnu};
  bool debut_a_activite_{false};
};

class EntiteSimple {
 public:
  explicit EntiteSimple(const Reglages &r) : r_(r) {}
  uint8_t pas(double q_marche, bool porte);
  uint8_t etat() const { return etat_; }

 private:
  Reglages r_;
  uint8_t etat_{0};
  int n_{0};
};

// ------------------------------------------------------------------ chaîne complète
constexpr int kLongueur = 130;   // fenêtre du réseau (s)
constexpr int kCanaux = 2;       // log, lin
constexpr int kClasses = 12;     // sortie du réseau (fenetres.py : CLASSES_LIGNE)

struct Sortie {
  double t;
  double p_moy;
  bool valide, porte, reseau;    // reseau : le réseau a tourné à cette seconde
  float q[kClasses];             // probabilités du réseau (repos = 1 si pas calculé)
  uint8_t machine, activite, grille_pain, bouilloire, inconnu;
};

class Chaine {
 public:
  // m : réseau ; buf_a, buf_b : tampons de travail du moteur (kMaxWidth floats chacun)
  Chaine(const nn::Model &m, float *buf_a, float *buf_b, const Reglages &r)
      : m_(m), buf_a_(buf_a), buf_b_(buf_b), machine_(r), gp_(r), bouilloire_(r), inconnu_(r) {}
  // Traite une case de la grille (une par seconde).
  Sortie traiter(const Case &c);

 private:
  const nn::Model &m_;
  float *buf_a_, *buf_b_;
  float log_[kLongueur]{}, lin_[kLongueur]{};      // fenêtre circulaire
  int tete_{0}, n_cases_{0};
  bool haut_[5]{};                                  // P > 20 W dans les 5 dernières cases
  int haut_tete_{0};
  float entree_[kCanaux * kLongueur]{};
  AutomateMachine machine_;
  EntiteSimple gp_, bouilloire_, inconnu_;
};

}  // namespace ss

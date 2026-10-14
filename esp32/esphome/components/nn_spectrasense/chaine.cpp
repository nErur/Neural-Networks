#include "chaine.h"

#include <cmath>

namespace ss {

// ------------------------------------------------------------------ adaptateur (adaptateur.py)
static const double kPas = 1.0, kSeuilActif = 60.0, kSilenceActif = 10.0, kSilenceMax = 30.0;
static const double kLnRef = std::log1p(3000.0);

static bool case_valide(double p_dernier, double age) {
  if (age > kSilenceMax) return false;
  return !(p_dernier > kSeuilActif && age > kSilenceActif);
}

int Adaptateur::fermer_jusqua_(double t, Recepteur &r) {
  int n = 0;
  while (t >= t_case_ + kPas) {
    const double fin = t_case_ + kPas;
    acc_ += p_ * (fin - t_curseur_);
    const double p_moy = acc_ / kPas;
    const Case c{t_case_, p_moy, (float) (std::log1p(p_moy > 0.0 ? p_moy : 0.0) / kLnRef),
                 case_valide(p_, fin - t_mesure_)};
    n++;
    t_case_ = fin;
    t_curseur_ = fin;
    acc_ = 0.0;
    r.recevoir(c);                                   // après la mise à jour : l'état est cohérent
  }
  return n;
}

int Adaptateur::pousser(double t, double p, Recepteur &r) {
  if (!demarre_) {                                   // la grille démarre à la seconde suivante
    demarre_ = true;
    t_case_ = std::ceil(t / kPas) * kPas;
    t_curseur_ = t_case_;
    t_mesure_ = t;
    p_ = p;
    return 0;
  }
  if (t < t_curseur_) return 0;                      // horodatage qui recule : ignoré
  const int n = fermer_jusqua_(t, r);
  if (t >= t_case_) {
    acc_ += p_ * (t - t_curseur_);
    t_curseur_ = t;
  }
  t_mesure_ = t;
  p_ = p;
  return n;
}

int Adaptateur::avancer_horloge(double t, Recepteur &r) {
  return demarre_ ? fermer_jusqua_(t, r) : 0;
}

// ------------------------------------------------------------------ entrée HA (debrouiller_rafales)
static const double kRafale = 0.3, kPeriodePzem = 1.0;

void EntreeHa::vider(Recepteur &r) {
  const int n = n_;
  if (n == 0) return;
  const double dernier = t_[n - 1];
  double pas = kPeriodePzem;
  if (a_prec_ && dernier - (n - 1) * kPeriodePzem <= prec_ + 0.05) pas = (dernier - prec_) / n;
  for (int i = 0; i < n; i++) {
    double t = n > 1 ? dernier - (n - 1 - i) * pas : t_[i];
    if (a_.demarre() && t < a_.curseur()) {          // sa case est déjà fermée par l'horloge
      t = a_.curseur();
      recalees_++;
    }
    a_.pousser(t, p_[i], r);
  }
  prec_ = dernier;
  a_prec_ = true;
  n_ = 0;
}

void EntreeHa::mesure(double t, double p, Recepteur &r) {
  if (n_ > 0 && (t - t_[n_ - 1] >= kRafale || n_ == kMax)) vider(r);
  t_[n_] = t;
  p_[n_] = p;
  n_++;
}

void EntreeHa::horloge(double t, Recepteur &r) {
  if (n_ > 0 && t - t_[n_ - 1] >= kRafale) vider(r);
  a_.avancer_horloge(t - retard_, r);
}

// ------------------------------------------------------------------ automate (automate.py)
enum : int { kRepos = 0, kChauffeA, kRegulationA, kFinAllumage, kBroyageA, kExtractionA, kFinCafe, kPicA, kExtinctionA };
static const double kFinMaintien = 2.5 * 222, kExtinctionSansPic = 140 * 60;
static const double kPicMax = 6, kTenuePorte = 5, kAnnulation = 60;
static const int kTenueFinCycle = 6, kTenueArret = 12;

static bool est_allumage(int a) { return a == kChauffeA || a == kRegulationA || a == kFinAllumage; }
static bool est_cafe(int a) { return a == kBroyageA || a == kExtractionA || a == kFinCafe; }

static uint8_t ha_de_act(int a) {
  static const uint8_t t[kAct] = {0, kChauffe, kRegulation, kFin, kBroyage, kExtraction, kFin, kMaintienHa, kArret};
  return t[a];
}
static uint8_t ha_de_long(Long l) {
  return l == Long::kVeille ? kVeilleHa : l == Long::kMaintien ? kMaintienHa : kArret;
}

int AutomateMachine::cible_(int i) const {
  switch (cycle_) {
    case Cycle::kAllumage:
      if (i == kBroyageA) return kRegulationA;
      if (i == kExtractionA || i == kPicA || i == kExtinctionA) return -1;
      if (i == kFinCafe) return kFinAllumage;
      return i;
    case Cycle::kCafe:
      if (i == kChauffeA || i == kRegulationA) return kBroyageA;
      if (i == kFinAllumage) return kFinCafe;
      if (i == kPicA || i == kExtinctionA) return -1;
      return i;
    case Cycle::kPic:
      return i == 0 ? 0 : -1;
    case Cycle::kExtinction:
      return (i == 0 || i == kExtinctionA) ? i : -1;
    default:
      break;
  }
  switch (long_) {
    case Long::kArret:
      if (i == kBroyageA) return kChauffeA;
      if (i == kExtractionA) return kRegulationA;
      if (i == kFinAllumage || i == kFinCafe || i == kExtinctionA || i == kPicA) return 0;
      return i;
    case Long::kVeille:
    case Long::kMaintien:
      if (i == kChauffeA || i == kRegulationA) return kBroyageA;
      if (i == kFinAllumage || i == kFinCafe) return 0;
      return i;
    default:                                         // inconnu
      if (i == kFinAllumage || i == kFinCafe) return 0;
      return i;
  }
}

void AutomateMachine::annuler_cycle(double t) {
  if ((cycle_ == Cycle::kAllumage || cycle_ == Cycle::kCafe) && a_debut_ && t - debut_t_ <= kAnnulation) {
    long_ = debut_long_;
    t_activite_ = debut_t_activite_;
    a_activite_ = debut_a_activite_;
    act_ = 0;
    cycle_ = Cycle::kAucun;
    a_debut_ = false;
    n_ = 0;
  }
}

void AutomateMachine::publier_(double t, int nouv) {
  const int anc = act_;
  act_ = nouv;
  if (nouv == 0) {
    if (cycle_ == Cycle::kAllumage || cycle_ == Cycle::kCafe) {
      long_ = Long::kVeille;
      t_activite_ = t;
      a_activite_ = true;
    } else if (cycle_ == Cycle::kExtinction) {
      long_ = Long::kArret;
    } else if (cycle_ == Cycle::kPic) {
      long_ = Long::kMaintien;
      t_pic_ = t;
      a_pic_ = true;
    }
    cycle_ = Cycle::kAucun;
  } else if (anc == 0) {
    cycle_ = est_allumage(nouv) ? Cycle::kAllumage : est_cafe(nouv) ? Cycle::kCafe
             : nouv == kPicA ? Cycle::kPic : Cycle::kExtinction;
    a_debut_ = true;
    debut_t_ = t;
    debut_long_ = long_;
    debut_t_activite_ = t_activite_;
    debut_a_activite_ = a_activite_;
    if (cycle_ == Cycle::kPic) {
      long_ = Long::kMaintien;
      t_pic_ = t;
      a_pic_ = true;
    }
    if (cycle_ == Cycle::kExtinction) long_ = Long::kArret;
  }
}

void AutomateMachine::minuterie_(double t) {
  if (act_ != 0) return;
  if (long_ == Long::kMaintien && a_pic_ && t - t_pic_ > kFinMaintien) {
    long_ = Long::kArret;
  } else if (long_ == Long::kVeille && a_activite_ && t - t_activite_ > kExtinctionSansPic) {
    long_ = Long::kArret;
  }
}

uint8_t AutomateMachine::ha() const { return act_ ? ha_de_act(act_) : ha_de_long(long_); }

uint8_t AutomateMachine::pas(double t, const double q[kAct], bool porte) {
  minuterie_(t);
  double q2[kAct] = {0};
  int c = 0;
  double pc = 1.0;
  if (porte) {
    for (int i = 0; i < kAct; i++) {
      const int j = cible_(i);
      q2[j < 0 ? act_ : j] += q[i];
    }
    for (int i = 1; i < kAct; i++)
      if (q2[i] > q2[c]) c = i;                      // premier maximum, comme max() en Python
    pc = q2[c];
  } else {
    q2[0] = 1.0;
  }
  if (c == kPicA) {                                  // un pic n'est jamais publié comme activité
    c = 0;
    pc = 1.0;
  }
  if (porte && !porte_prec_) t_ouverture_ = t;
  const double q_pic = porte ? q2[kPicA] : 0.0;
  double mx = q2[1];
  for (int i = 2; i < kAct; i++) mx = q2[i] > mx ? q2[i] : mx;
  if (cycle_ == Cycle::kAucun && q_pic >= r_.seuil_pic && q_pic >= mx) {
    if (!a_pic_vu_) {
      a_pic_vu_ = true;
      pic_vu_ = t;
    }
  }
  if (a_pic_vu_) {
    if (cycle_ != Cycle::kAucun || (porte && t - t_ouverture_ > kPicMax + kTenuePorte + 1)) {
      a_pic_vu_ = false;                             // c'était le début d'autre chose
    } else if (!porte && porte_prec_) {
      long_ = Long::kMaintien;
      t_pic_ = t;
      a_pic_ = true;
      a_pic_vu_ = false;
    }
  }
  porte_prec_ = porte;
  const int tenue = (c == 0 && (cycle_ == Cycle::kAllumage || cycle_ == Cycle::kCafe)) ? kTenueFinCycle : r_.tenue;
  if (c != act_ && pc >= r_.seuil) {
    n_ = (c == cand_) ? n_ + 1 : 1;
    cand_ = c;
    if (n_ >= (tenue > 1 ? tenue : 1)) {
      publier_(t, c);
      n_ = 0;
    }
  } else {
    n_ = 0;
  }
  return ha();
}

uint8_t EntiteSimple::pas(double q_marche, bool porte) {
  const double q = porte ? q_marche : 0.0;
  const int c = q >= 0.5 ? 1 : 0;
  const double pc = c ? q : 1.0 - q;
  if (c != etat_ && pc >= r_.seuil) {
    n_ += 1;
    const int besoin = c == 1 ? r_.tenue : (r_.tenue > kTenueArret ? r_.tenue : kTenueArret);
    if (n_ >= besoin) {
      etat_ = (uint8_t) c;
      n_ = 0;
    }
  } else {
    n_ = 0;
  }
  return etat_;
}

// ------------------------------------------------------------------ chaîne (reseau.py + executer)
Sortie Chaine::traiter(const Case &c) {
  Sortie s{};
  s.t = c.t;
  s.p_moy = c.p_moy;
  s.valide = c.valide;
  // fenêtre : log = x (float32), lin = expm1(x · ln 3001) / 3000, recalculé depuis x comme fenetres.py
  log_[tete_] = c.x;
  lin_[tete_] = (float) (std::expm1((double) c.x * kLnRef) / 3000.0);
  tete_ = (tete_ + 1) % kLongueur;
  if (n_cases_ < kLongueur) n_cases_++;
  // porte : p_moy > 20 W dans l'une des 5 dernières cases
  haut_[haut_tete_] = c.p_moy > 20.0;
  haut_tete_ = (haut_tete_ + 1) % 5;
  s.porte = haut_[0] || haut_[1] || haut_[2] || haut_[3] || haut_[4];

  for (int k = 0; k < kClasses; k++) s.q[k] = 0.0f;
  s.q[0] = 1.0f;
  if (s.porte && n_cases_ == kLongueur) {
    for (int i = 0; i < kLongueur; i++) {            // du plus ancien au plus récent, canal par canal
      const int j = (tete_ + i) % kLongueur;
      entree_[i] = log_[j];
      entree_[kLongueur + i] = lin_[j];
    }
    nn::forward(m_, entree_, buf_a_, buf_b_);
    const float *p = buf_b_;                           // dernière couche : tampon selon la parité
    if (m_.n_layers % 2 == 1) p = buf_a_;
    for (int k = 0; k < kClasses; k++) s.q[k] = p[k];
    s.reseau = true;
  }
  // machine : repos = repos + autres appareils (en float32, comme reseau.py)
  float autres = s.q[9] + s.q[10];
  autres += s.q[11];
  double qm[kAct];
  qm[0] = (double) (float) (s.q[0] + autres);
  for (int k = 1; k < kAct; k++) qm[k] = s.q[k];
  s.machine = machine_.pas(c.t, qm, s.porte);
  const uint8_t avant[3] = {gp_.etat(), bouilloire_.etat(), inconnu_.etat()};
  s.grille_pain = gp_.pas(s.q[9], s.porte);
  if (s.grille_pain && !avant[0]) { machine_.annuler_cycle(c.t); s.machine = machine_.ha(); }
  s.bouilloire = bouilloire_.pas(s.q[10], s.porte);
  if (s.bouilloire && !avant[1]) { machine_.annuler_cycle(c.t); s.machine = machine_.ha(); }
  s.inconnu = inconnu_.pas(s.q[11], s.porte);
  if (s.inconnu && !avant[2]) { machine_.annuler_cycle(c.t); s.machine = machine_.ha(); }
  s.activite = machine_.activite();
  return s;
}

}  // namespace ss

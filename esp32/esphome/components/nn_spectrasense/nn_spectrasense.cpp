#include "nn_spectrasense.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "esphome/core/hal.h"
#include "esphome/core/log.h"

#ifdef USE_ESP32
#include "esp_cpu.h"
#include "esp_timer.h"
#endif

#include "esphome/components/nn_mnist/nn_engine.h"   // moteur commun (compilé une fois, par nn_mnist)
#include "model_weights.h"
#include "test_vectors.h"

namespace esphome {
namespace nn_spectrasense {

static const char *const TAG = "nn_spectrasense";
namespace M = ::nn_ss;

static const nn::Model *g_model = &M::kModel;
static nn::Layer g_ram_layers[M::kNumLayers];
static nn::Model g_ram_model;
static const nn::Model &model() { return *g_model; }

static float input_[M::kInputSize];   // fenêtre en RAM, comme en usage réel (les références sont en flash)
static float buf_a_[M::kMaxWidth];
static float buf_b_[M::kMaxWidth];
static float layer_out_[M::kNumLayers][M::kMaxWidth];

static uint32_t cycles() {
#ifdef USE_ESP32
  return esp_cpu_get_cycle_count();
#else
  return micros();
#endif
}

// Écart relatif : écart maximal / max(1, plus grande valeur de la couche de référence).
static float ecart_relatif(const float *a, const float *ref, int n) {
  float d = 0.0f, m = 1.0f;
  for (int i = 0; i < n; i++) {
    d = fmaxf(d, fabsf(a[i] - ref[i]));
    m = fmaxf(m, fabsf(ref[i]));
  }
  return d / m;
}

static const char *nom_type(const nn::Layer &L) {
  return L.type == nn::LayerType::kConv1d ? "conv" : L.type == nn::LayerType::kMaxPool1d ? "pool" : "dense";
}

// Multiplications-additions d'une couche (comparaisons pour un pooling).
static uint32_t operations(const nn::Layer &L) {
  if (L.type == nn::LayerType::kConv1d) return (uint32_t) L.c_out * L.l_out * L.c_in * L.kernel;
  if (L.type == nn::LayerType::kMaxPool1d) return (uint32_t) L.c_out * L.l_out * (L.kernel - 1);
  return (uint32_t) L.n_in * L.n_out;
}

static const char *const kEtatsMachine[] = {"arrêt", "veille", "maintien", "chauffe",
                                            "régulation", "broyage", "extraction", "fin"};

void NnSpectrasense::setup() {
  if (weights_in_ram_ && !copy_weights_to_ram_()) ESP_LOGE(TAG, "RAM insuffisante : poids laissés en flash");
  if (model_id_ != nullptr) model_id_->publish_state(model().id);
  if (power_ != nullptr) {
    // une seule allocation, au démarrage ; la chaîne utilise le modèle définitif (RAM ou flash)
    entree_ = new ss::EntreeHa(clock_delay_);
    chaine_ = new ss::Chaine(model(), buf_a_, buf_b_, reglages_);
    power_->add_on_state_callback([this](float p) { this->mesure_(p); });
    pref_ = global_preferences->make_preference<Memoire>(0x53534D31UL /* "SSM1" */, true);
    pref_cafes_ = global_preferences->make_preference<uint32_t>(0x53534331UL /* "SSC1" */, true);
    if (!pref_cafes_.load(&n_cafes_)) n_cafes_ = 0;
    ESP_LOGI(TAG, "compteur de cafés repris : %u", (unsigned) n_cafes_);
    if (cafes_ != nullptr) cafes_->publish_state(n_cafes_);
  }
}

bool NnSpectrasense::epoch_(int64_t &e) const {
#ifdef USE_TIME
  if (time_ == nullptr) return false;
  const auto n = time_->now();
  if (!n.is_valid()) return false;
  e = (int64_t) n.timestamp;
  return true;
#else
  return false;
#endif
}

static const char *nom_long(ss::Long l) {
  return l == ss::Long::kArret ? "arrêt" : l == ss::Long::kVeille ? "veille" : l == ss::Long::kMaintien ? "maintien"
                                                                                                         : "inconnu";
}

// Au démarrage, dès que l'heure réelle est connue : reprend l'état long mémorisé, vieilli du temps
// écoulé (les minuteries de l'automate font le reste : veille -> arrêt après 140 min, etc.).
void NnSpectrasense::restaurer_() {
  int64_t e;
  if (!epoch_(e)) return;
  restaure_ = true;
  Memoire m{};
  if (!pref_.load(&m) || m.etat > (uint8_t) ss::Long::kMaintien) {
    ESP_LOGI(TAG, "pas d'état mémorisé : machine en état inconnu jusqu'au prochain cycle");
    return;
  }
  ss::AutomateMachine &a = chaine_->machine();
  mem_etat_ = m.etat;
  mem_ref_ = m.epoch_ref;
  if (a.etat_long() != ss::Long::kInconnu || a.activite() != 0) return;   // un cycle a déjà tranché
  const ss::Long l = (ss::Long) m.etat;
  const double t_ref = maintenant_() - (double) (e - m.epoch_ref);
  a.forcer_long(l, t_ref);
  ESP_LOGI(TAG, "état mémorisé repris : %s (référence il y a %.0f min)", nom_long(l), (e - m.epoch_ref) / 60.0);
}

// Après chaque case : mémorise l'état long s'il a changé (quelques écritures par jour).
void NnSpectrasense::memoriser_() {
  const ss::AutomateMachine &a = chaine_->machine();
  const ss::Long l = a.etat_long();
  int64_t e;
  if (!restaure_ || l == ss::Long::kInconnu || !epoch_(e)) return;
  const int64_t ref = l == ss::Long::kArret ? 0 : e - (int64_t) llround(maintenant_() - a.t_reference());
  if ((int) l == mem_etat_ && llabs(ref - mem_ref_) <= 2) return;
  Memoire m{(uint8_t) l, ref};
  if (pref_.save(&m)) {
    mem_etat_ = (int) l;
    mem_ref_ = ref;
    ESP_LOGD(TAG, "état long mémorisé : %s", nom_long(l));
  }
}

// Horloge monotone en secondes depuis le démarrage (64 bits : pas de retour à zéro).
double NnSpectrasense::maintenant_() {
#ifdef USE_ESP32
  return (double) esp_timer_get_time() * 1e-6;
#else
  return millis() * 1e-3;
#endif
}

void NnSpectrasense::mesure_(float p) {
  if (std::isnan(p)) return;                        // capteur indisponible : l'horloge invalidera les cases
  n_lectures_++;
  entree_->mesure(maintenant_(), p, *this);
}

void NnSpectrasense::recevoir(const ss::Case &c) {
  const uint32_t t0 = micros();
  const ss::Sortie s = chaine_->traiter(c);
  if (s.reseau) {
    inf_us_ += micros() - t0;
    inf_n_++;
  }
  n_cases_++;
  if (!c.valide) n_invalides_++;
  memoriser_();
  int k = 0;                                         // classe dominante du réseau (repos si porte fermée)
  for (int i = 1; i < ss::kClasses; i++)
    if (s.q[i] > s.q[k]) k = i;
  if (s.reseau)
    ESP_LOGD(TAG, "%6.0f W  %-20s %5.1f %%  machine %-10s%s%s%s", s.p_moy, M::kClasses[k], 100.0f * s.q[k],
             kEtatsMachine[s.machine], s.grille_pain ? "  grille-pain" : "", s.bouilloire ? "  bouilloire" : "",
             s.inconnu ? "  inconnu" : "");
  if (s.cafe) {                                      // café compté : cycle terminé, broyage et extraction vus
    n_cafes_++;
    pref_cafes_.save(&n_cafes_);
    ESP_LOGI(TAG, "café n° %u (broyage %d s, extraction %d s)", (unsigned) n_cafes_, s.cafe_broyage,
             s.cafe_extraction);
    if (cafes_ != nullptr) cafes_->publish_state(n_cafes_);
  } else if (chaine_->machine().cafe_termine()) {
    ESP_LOGI(TAG, "cycle café non compté (broyage %d s, extraction %d s : minimum %d s chacun)", s.cafe_broyage,
             s.cafe_extraction, ss::kCafeBroyageMin);
  }
  if (s.machine != pub_machine_) {
    if (pub_machine_ >= 0) ESP_LOGI(TAG, "machine : %s -> %s", kEtatsMachine[pub_machine_], kEtatsMachine[s.machine]);
    pub_machine_ = s.machine;
    if (machine_ != nullptr) machine_->publish_state(kEtatsMachine[s.machine]);
  }
  // confiance de la classe dominante (%) : publiée si la classe change ou si elle bouge d'au moins 2 points
  const float conf = 100.0f * s.q[k];
  if (confiance_ != nullptr && (k != pub_classe_ || fabsf(conf - pub_confiance_) >= 2.0f)) {
    pub_confiance_ = conf;
    confiance_->publish_state(conf);
  }
  if (k != pub_classe_) {
    pub_classe_ = k;
    if (classe_ != nullptr) classe_->publish_state(M::kClasses[k]);
  }
  struct { binary_sensor::BinarySensor *e; int *pub; int v; const char *nom; } simples[] = {
      {grille_pain_, &pub_gp_, s.grille_pain, "grille-pain"},
      {bouilloire_, &pub_bouilloire_, s.bouilloire, "bouilloire"},
      {inconnu_, &pub_inconnu_, s.inconnu, "consommateur inconnu"},
      {porte_, &pub_porte_, s.porte, nullptr},
  };
  for (auto &x : simples) {
    if (x.v == *x.pub) continue;
    if (*x.pub >= 0 && x.nom != nullptr) ESP_LOGI(TAG, "%s : %s", x.nom, x.v ? "marche" : "arrêt");
    *x.pub = x.v;
    if (x.e != nullptr) x.e->publish_state(x.v);
  }
}

// Toutes les minutes : secondes invalides (si elles ont changé) et temps d'inférence moyen.
void NnSpectrasense::bilan_minute_() {
  ESP_LOGD(TAG, "depuis le démarrage : %u lectures, %u cases dont %u invalides", (unsigned) n_lectures_,
           (unsigned) n_cases_, (unsigned) n_invalides_);
  if (invalides_ != nullptr && (int64_t) n_invalides_ != pub_invalides_) {
    pub_invalides_ = n_invalides_;
    invalides_->publish_state(n_invalides_);
  }
  if (inf_n_ > 0) {
    if (inference_time_ != nullptr) inference_time_->publish_state((float) inf_us_ / inf_n_);
    inf_us_ = inf_n_ = 0;
  }
}

bool NnSpectrasense::copy_weights_to_ram_() {
  const nn::Model &src = M::kModel;
  size_t n = 0;
  for (int l = 0; l < src.n_layers; l++)
    if (src.layers[l].W != nullptr) n += nn::weight_count(src.layers[l]) + src.layers[l].c_out + src.layers[l].n_out;
  float *p = new (std::nothrow) float[n];          // une seule allocation, au démarrage (majorant)
  if (p == nullptr) return false;
  for (int l = 0; l < src.n_layers; l++) {
    const nn::Layer &L = src.layers[l];
    g_ram_layers[l] = L;
    if (L.W == nullptr) continue;                    // pooling : ni poids ni biais
    const size_t nw = nn::weight_count(L);
    const size_t nb = L.type == nn::LayerType::kConv1d ? L.c_out : L.n_out;
    memcpy(p, L.W, nw * sizeof(float));
    g_ram_layers[l].W = p;
    p += nw;
    memcpy(p, L.b, nb * sizeof(float));
    g_ram_layers[l].b = p;
    p += nb;
  }
  g_ram_model = src;
  g_ram_model.layers = g_ram_layers;
  g_model = &g_ram_model;
  return true;
}

void NnSpectrasense::dump_config() {
  ESP_LOGCONFIG(TAG, "nn_spectrasense :");
  if (power_ != nullptr) {
    ESP_LOGCONFIG(TAG, "  Chaîne en direct : seuil %.2f, tenue %d, seuil pic %.2f, horloge en retard de %.1f s",
                  reglages_.seuil, reglages_.tenue, reglages_.seuil_pic, clock_delay_);
#ifdef USE_TIME
    ESP_LOGCONFIG(TAG, "  État long gardé en flash : %s", time_ != nullptr ? "oui (heure de HA)" : "non (pas de time_id)");
#endif
  }
  else
    ESP_LOGCONFIG(TAG, "  Pas de capteur de puissance : autotest et profilage seulement");
  ESP_LOGCONFIG(TAG, "  Modèle : %s", model().id);
  ESP_LOGCONFIG(TAG, "  Couches : %d, entrée %d, sortie %d, tampons %d floats", model().n_layers, M::kInputSize,
                M::kOutputSize, M::kMaxWidth);
  ESP_LOGCONFIG(TAG, "  Poids : %s ; fenêtres de test embarquées : %d", g_model == &g_ram_model ? "en RAM" : "en flash",
                M::kNumTestVectors);
}

void NnSpectrasense::run_autotest() {
  if (autotest_index_ >= 0 || profile_index_ >= 0) return;
  autotest_index_ = 0;
  autotest_ok_ = 0;
  autotest_worst_ = 0.0f;
  autotest_us_ = 0;
}

void NnSpectrasense::run_profile() {
  if (autotest_index_ >= 0 || profile_index_ >= 0) return;
  profile_index_ = 0;
  prof_total_ = 0;
  for (int l = 0; l < 8; l++) prof_calc_[l] = prof_act_[l] = 0;
  prof_c0_ = cycles();
  prof_u0_ = micros();
}

void NnSpectrasense::loop() {
  if (entree_ != nullptr) {
    if (!restaure_) restaurer_();
    entree_->horloge(maintenant_(), *this);         // ferme les cases (rafale close, silences)
    const uint32_t ms = millis();
    if (ms - t_bilan_ >= 60000) {
      t_bilan_ = ms;
      bilan_minute_();
    }
  }
  if (profile_index_ >= 0) {
    profile_step_();
    return;
  }
  if (autotest_index_ >= 0) autotest_step_();
}

void NnSpectrasense::autotest_step_() {
  const nn::Model &m = model();
  float *trace[M::kNumLayers];
  for (int l = 0; l < M::kNumLayers; l++) trace[l] = layer_out_[l];
  const int t = autotest_index_;
  const M::TestVector &v = M::kTestVectors[t];
  const uint32_t t0 = micros();
  const nn::Result r = nn::forward(m, v.input, buf_a_, buf_b_, trace);
  const uint32_t dt = micros() - t0;
  autotest_us_ += dt;
  float err = 0.0f;
  for (int l = 0; l < m.n_layers; l++) err = fmaxf(err, ecart_relatif(layer_out_[l], v.outputs[l], m.layers[l].n_out));
  const bool ok = err <= 1e-5f && r.prediction == v.py_pred;
  autotest_ok_ += ok;
  autotest_worst_ = fmaxf(autotest_worst_, err);
  ESP_LOGD(TAG, "  autotest %2d : %-20s Python %-20s ESP %-20s écart %.1e, %u µs %s", t, M::kClasses[v.label],
           M::kClasses[v.py_pred], M::kClasses[r.prediction], err, (unsigned) dt, ok ? "ok" : "ÉCHEC");
  if (++autotest_index_ < M::kNumTestVectors) return;
  autotest_index_ = -1;
  char bilan[120];
  snprintf(bilan, sizeof(bilan), "%d/%d conformes, écart relatif max %.1e, %u µs en moyenne, poids en %s",
           autotest_ok_, M::kNumTestVectors, autotest_worst_, (unsigned) (autotest_us_ / M::kNumTestVectors),
           g_model == &g_ram_model ? "RAM" : "flash");
  ESP_LOGI(TAG, "autotest : %s", bilan);
  if (autotest_ != nullptr) autotest_->publish_state(bilan);
  if (inference_time_ != nullptr) inference_time_->publish_state(autotest_us_ / M::kNumTestVectors);
  if (max_error_ != nullptr) max_error_->publish_state(autotest_worst_);
}

void NnSpectrasense::profile_step_() {
  const nn::Model &m = model();
  if (profile_index_ < M::kNumTestVectors) {
    nn::Profile prof{cycles, {0}, {0}};
    memcpy(input_, M::kTestVectors[profile_index_].input, sizeof(input_));
    const uint32_t t0 = cycles();
    nn::forward(m, input_, buf_a_, buf_b_, nullptr, &prof);
    prof_total_ += cycles() - t0;
    for (int l = 0; l < m.n_layers && l < 8; l++) {
      prof_calc_[l] += prof.matvec[l];
      prof_act_[l] += prof.activation[l];
    }
    profile_index_++;
    return;
  }
  profile_report_();
  profile_index_ = -1;
}

void NnSpectrasense::profile_report_() {
  const nn::Model &m = model();
  const int N = M::kNumTestVectors;
  const float mhz = (float) (cycles() - prof_c0_) / (float) (micros() - prof_u0_);
  auto us = [mhz](uint32_t c) { return c / mhz; };
  ESP_LOGI(TAG, "profilage : moyenne sur %d fenêtres, processeur à %.0f MHz, poids en %s", N, mhz,
           g_model == &g_ram_model ? "RAM" : "flash");
  uint32_t somme = 0;
  for (int l = 0; l < m.n_layers && l < 8; l++) {
    const nn::Layer &L = m.layers[l];
    const uint32_t c = prof_calc_[l] / N, a = prof_act_[l] / N, ops = operations(L);
    somme += c + a;
    ESP_LOGI(TAG, "  couche %d %-5s %4dx%-4d -> %3dx%-4d : calcul %7u cycles (%6.0f µs, %5.1f cycles/op. sur %6u), "
             "activation %6u cycles", l, nom_type(L), L.c_in ? L.c_in : L.n_in, L.c_in ? L.l_in : 1,
             L.c_out ? L.c_out : L.n_out, L.c_out ? L.l_out : 1, (unsigned) c, us(c), ops ? (float) c / ops : 0.0f,
             (unsigned) ops, (unsigned) a);
  }
  ESP_LOGI(TAG, "  total mesuré : %u cycles = %.0f µs (somme des couches : %.0f µs)", (unsigned) (prof_total_ / N),
           us(prof_total_ / N), us(somme));
}

}  // namespace nn_spectrasense
}  // namespace esphome

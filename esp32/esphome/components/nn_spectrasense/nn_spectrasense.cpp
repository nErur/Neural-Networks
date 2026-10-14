#include "nn_spectrasense.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#include "esphome/core/hal.h"
#include "esphome/core/log.h"

#ifdef USE_ESP32
#include "esp_cpu.h"
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

void NnSpectrasense::setup() {
  if (weights_in_ram_ && !copy_weights_to_ram_()) ESP_LOGE(TAG, "RAM insuffisante : poids laissés en flash");
  if (model_id_ != nullptr) model_id_->publish_state(model().id);
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

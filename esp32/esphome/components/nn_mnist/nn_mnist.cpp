#include "nn_mnist.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "esphome/core/hal.h"
#include "esphome/core/log.h"

#include "model_weights.h"
#include "nn_engine.h"

#ifdef USE_NN_MNIST_TEST_VECTORS
#include "test_vectors.h"
#endif

namespace esphome {
namespace nn_mnist {

static const char *const TAG = "nn_mnist";
static const nn::Model &model() { return ::nn_model::kModel; }

// Tampons de calcul, dimensionnés à la compilation d'après le modèle : pas de malloc.
static float input_[::nn_model::kInputSize];
static float buf_a_[::nn_model::kMaxWidth];
static float buf_b_[::nn_model::kMaxWidth];
static float layer_out_[::nn_model::kNumLayers][::nn_model::kMaxWidth];

// Écart maximal entre une sortie et sa référence (images de test seulement).
static float max_abs_diff(const float *a, const float *b, int n) {
  float d = 0.0f;
  for (int i = 0; i < n; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
  return d;
}

void NnMnist::setup() {
  if (model_id_ != nullptr) model_id_->publish_state(model().id);
  ESP_LOGI(TAG, "modèle chargé : %s", model().id);
}

void NnMnist::dump_config() {
  ESP_LOGCONFIG(TAG, "nn_mnist :");
  ESP_LOGCONFIG(TAG, "  Modèle : %s", model().id);
  ESP_LOGCONFIG(TAG, "  Couches : %d, entrée %d, sortie %d", model().n_layers,
                ::nn_model::kInputSize, ::nn_model::kOutputSize);
#ifdef USE_NN_MNIST_TEST_VECTORS
  ESP_LOGCONFIG(TAG, "  Images de test embarquées : %d", ::nn_model::kNumTestVectors);
#else
  ESP_LOGCONFIG(TAG, "  Images de test embarquées : non (test_vectors: false)");
#endif
}

void NnMnist::next_test_image() {
#ifdef USE_NN_MNIST_TEST_VECTORS
  pending_test_ = next_test_;
  next_test_ = (next_test_ + 1) % ::nn_model::kNumTestVectors;
  memcpy(image_, ::nn_model::kTestVectors[pending_test_].image, kRawSize);
  pending_ = Pending::kImage;
#else
  ESP_LOGW(TAG, "pas d'images de test embarquées (test_vectors: false)");
#endif
}

void NnMnist::run_autotest() {
#ifdef USE_NN_MNIST_TEST_VECTORS
  pending_ = Pending::kAutotest;
#else
  ESP_LOGW(TAG, "pas d'images de test embarquées (test_vectors: false)");
#endif
}

void NnMnist::loop() {
  const Pending p = pending_;
  pending_ = Pending::kNone;
  if (p == Pending::kNone) return;
  signal_activity_();
  if (p == Pending::kImage) classify_(pending_test_);
  else do_autotest_();
}

void NnMnist::signal_activity_() {
  if (activity_ == nullptr) return;
  activity_->turn_on();
  // Un nouveau set_timeout du même nom remplace le précédent : la LED reste
  // allumée tant que les inférences s'enchaînent.
  this->set_timeout("activity", activity_duration_, [this]() { activity_->turn_off(); });
}

void NnMnist::classify_(int test_index) {
  const nn::Model &m = model();
  float *trace[::nn_model::kNumLayers];
  for (int l = 0; l < ::nn_model::kNumLayers; l++) trace[l] = layer_out_[l];

  // Chronométrage : prétraitement + forward, comme en usage réel.
  const uint32_t t0 = micros();
  nn::preprocess(m, image_, input_);
  const nn::Result r = nn::forward(m, input_, buf_a_, buf_b_, trace);
  const uint32_t dt = micros() - t0;

  if (!first_done_) {
    ESP_LOGI(TAG, "première inférence : %u µs (cache de la flash encore froid)", (unsigned) dt);
    first_done_ = true;
  }

  float expected = NAN, error = NAN;
#ifdef USE_NN_MNIST_TEST_VECTORS
  if (test_index >= 0) {
    const ::nn_model::TestVector &v = ::nn_model::kTestVectors[test_index];
    const int n_in = (int) nn::input_size(m);
    int bits = 0;
    for (int i = 0; i < n_in; i++) bits += memcmp(&input_[i], &v.input[i], sizeof(float)) != 0;
    error = 0.0f;
    for (int l = 0; l < m.n_layers; l++) {
      const float d = max_abs_diff(layer_out_[l], v.outputs[l], m.layers[l].n_out);
      ESP_LOGD(TAG, "  couche %d : écart %.2e", l, d);
      error = fmaxf(error, d);
    }
    expected = v.label;
    ESP_LOGI(TAG, "image de test %d (MNIST %d) : étiquette %d, Python %d, ESP %d, entrée %s, écart %.2e",
             test_index, v.test_index, v.label, v.py_pred, r.prediction,
             bits ? "DIFFÉRENTE" : "identique", error);
  }
#endif
  ESP_LOGI(TAG, "prédiction %d, confiance %.1f %%, marge %.1f %%, %u µs", r.prediction,
           r.confidence * 100.0f, r.margin * 100.0f, (unsigned) dt);

  if (prediction_ != nullptr) prediction_->publish_state(r.prediction);
  if (confidence_ != nullptr) confidence_->publish_state(r.confidence * 100.0f);
  if (margin_ != nullptr) margin_->publish_state(r.margin * 100.0f);
  if (inference_time_ != nullptr) inference_time_->publish_state(dt);
  if (expected_label_ != nullptr) expected_label_->publish_state(expected);
  if (max_error_ != nullptr) max_error_->publish_state(error);
}

void NnMnist::do_autotest_() {
#ifdef USE_NN_MNIST_TEST_VECTORS
  // Les 24 images en une fois : ~24 × quelques centaines de µs, bien sous la
  // limite d'environ 30 ms au-delà de laquelle ESPHome signale un blocage.
  const nn::Model &m = model();
  const int n_in = (int) nn::input_size(m);
  float *trace[::nn_model::kNumLayers];
  for (int l = 0; l < ::nn_model::kNumLayers; l++) trace[l] = layer_out_[l];

  int ok_count = 0;
  float worst = 0.0f;
  uint32_t total_us = 0;
  for (int t = 0; t < ::nn_model::kNumTestVectors; t++) {
    const ::nn_model::TestVector &v = ::nn_model::kTestVectors[t];
    const uint32_t t0 = micros();
    nn::preprocess(m, v.image, input_);
    const nn::Result r = nn::forward(m, input_, buf_a_, buf_b_, trace);
    total_us += micros() - t0;

    int bits = 0;
    for (int i = 0; i < n_in; i++) bits += memcmp(&input_[i], &v.input[i], sizeof(float)) != 0;
    float err = 0.0f;
    for (int l = 0; l < m.n_layers; l++)
      err = fmaxf(err, max_abs_diff(layer_out_[l], v.outputs[l], m.layers[l].n_out));
    const bool ok = bits == 0 && err <= 1e-5f && r.prediction == v.py_pred;
    ok_count += ok;
    worst = fmaxf(worst, err);
    ESP_LOGD(TAG, "  autotest %2d : Python %d, ESP %d, entrée %s, écart %.2e %s", t, v.py_pred,
             r.prediction, bits ? "DIFFÉRENTE" : "identique", err, ok ? "ok" : "ÉCHEC");
  }
  char bilan[96];
  snprintf(bilan, sizeof(bilan), "%d/%d conformes, écart max %.1e, %u µs en moyenne", ok_count,
           ::nn_model::kNumTestVectors, worst, (unsigned) (total_us / ::nn_model::kNumTestVectors));
  ESP_LOGI(TAG, "autotest : %s", bilan);
  if (autotest_ != nullptr) autotest_->publish_state(bilan);
#endif
}

}  // namespace nn_mnist
}  // namespace esphome

// Vérifie le moteur C++ (convolution 1D, max-pooling, couches denses) sur le PC, couche par
// couche, avec les références numpy float32 générées par export_conv.py (test_vectors.h).
// Compile le moteur du composant ESPHome lui-même : voir ../lancer_tests_conv.py.

#include <cmath>
#include <cstdio>
#include <cstring>

#include "model_weights.h"
#include "nn_engine.h"
#include "test_vectors.h"

using namespace nn_model;

static const float kToleranceRel = 1e-5f;     // écart / max(1, plus grande valeur de la couche)

static uint32_t g_tick = 0;
static uint32_t fake_clock() { return ++g_tick; }

static float ecart_relatif(const float *a, const float *ref, int n) {
  float d = 0.0f, m = 1.0f;
  for (int i = 0; i < n; i++) {
    d = fmaxf(d, fabsf(a[i] - ref[i]));
    m = fmaxf(m, fabsf(ref[i]));
  }
  return d / m;
}

static const char *nom_type(nn::LayerType t) {
  return t == nn::LayerType::kConv1d ? "conv" : t == nn::LayerType::kMaxPool1d ? "pool" : "dense";
}

int main() {
  const nn::Model &m = kModel;
  printf("Modèle : %s\n", m.id);
  printf("%d couches, entrée %d, %d fenêtres de test, tolérance relative %.0e\n\n", m.n_layers,
         (int) nn::input_size(m), kNumTestVectors, kToleranceRel);
  printf(" fen  classe                py  C   ");
  for (int l = 0; l < m.n_layers; l++) printf("%d:%-5s   ", l, nom_type(m.layers[l].type));
  printf("verdict\n");

  static float buf_a[kMaxWidth], buf_b[kMaxWidth];
  static float layer_out[kNumLayers][kMaxWidth], out_p[kNumLayers][kMaxWidth];
  float *trace[kNumLayers], *trace_p[kNumLayers];
  for (int l = 0; l < kNumLayers; l++) {
    trace[l] = layer_out[l];
    trace_p[l] = out_p[l];
  }

  int failures = 0, stage_fail[kNumLayers + 1] = {0}, profile_diff = 0;
  float worst[kNumLayers] = {0};
  for (int t = 0; t < kNumTestVectors; t++) {
    const TestVector &v = kTestVectors[t];
    bool ok = true;
    nn::Result r = nn::forward(m, v.input, buf_a, buf_b, trace);
    float diff[kNumLayers];
    for (int l = 0; l < m.n_layers; l++) {
      diff[l] = ecart_relatif(layer_out[l], v.outputs[l], m.layers[l].n_out);
      worst[l] = fmaxf(worst[l], diff[l]);
      if (!(diff[l] <= kToleranceRel)) { ok = false; stage_fail[l]++; }
    }
    if (r.prediction != v.py_pred) { ok = false; stage_fail[kNumLayers]++; }

    nn::Profile prof{fake_clock, {0}, {0}};
    nn::forward(m, v.input, buf_a, buf_b, trace_p, &prof);
    for (int l = 0; l < m.n_layers; l++)
      if (memcmp(out_p[l], layer_out[l], m.layers[l].n_out * sizeof(float)) != 0) {
        ok = false;
        profile_diff++;
      }

    printf(" %3d  %-20s  %2d  %2d   ", t, kClasses[v.label], v.py_pred, r.prediction);
    for (int l = 0; l < m.n_layers; l++) printf("%.1e   ", diff[l]);
    printf("%s\n", ok ? "ok" : "ÉCHEC");
    failures += !ok;
  }

  printf("\nÉcart relatif maximal par couche :");
  for (int l = 0; l < m.n_layers; l++) printf(" %d:%s %.1e", l, nom_type(m.layers[l].type), worst[l]);
  printf("\nÉchecs par étage : ");
  for (int l = 0; l < m.n_layers; l++) printf("couche %d %d, ", l, stage_fail[l]);
  printf("prédiction %d\n", stage_fail[kNumLayers]);
  printf("Chemin profilé identique au bit près : %s\n", profile_diff ? "NON" : "oui");
  printf("%s : %d/%d fenêtres conformes\n", failures ? "ÉCHEC" : "SUCCÈS", kNumTestVectors - failures,
         kNumTestVectors);
  return failures ? 1 : 0;
}

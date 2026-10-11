// Étape 3 : vérifie le moteur C++ sur le PC, étage par étage, avec les
// références numpy float32 générées par export.py (test_vectors.h).
// Compile les fichiers du composant ESPHome eux-mêmes, pas une copie :
// voir lancer_tests.py.

#include <cmath>
#include <cstdio>
#include <cstring>

#include "nn_engine.h"
#include "test_vectors.h"

using namespace nn_model;

static const float kTolerance = 1e-5f;

static uint32_t g_tick = 0;
static uint32_t fake_clock() { return ++g_tick; }    // horloge factice : le profilage ne doit rien changer

static float max_abs_diff(const float *a, const float *b, int n) {
  float d = 0.0f;
  for (int i = 0; i < n; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
  return d;
}

int main() {
  const nn::Model &m = kModel;
  const int n_in = (int) nn::input_size(m);
  printf("Modèle : %s\n", m.id);
  printf("%d couches, entrée %d, %d images de test, tolérance %.0e\n\n", m.n_layers, n_in,
         kNumTestVectors, kTolerance);
  printf(" img  test  étiq  py  C   entrée    ");
  for (int l = 0; l < m.n_layers; l++) printf("couche %d   ", l);
  printf("verdict\n");

  float input[kInputSize], buf_a[kMaxWidth], buf_b[kMaxWidth];
  float layer_out[kNumLayers][kMaxWidth];
  float *trace[kNumLayers];
  for (int l = 0; l < kNumLayers; l++) trace[l] = layer_out[l];

  int failures = 0, stage_fail[kNumLayers + 2] = {0}, profile_diff = 0;
  float worst[kNumLayers] = {0};
  for (int t = 0; t < kNumTestVectors; t++) {
    const TestVector &v = kTestVectors[t];
    bool ok = true;

    // étage 0 : prétraitement, identique au bit près
    nn::preprocess(m, v.image, input);
    int bits = 0;
    for (int i = 0; i < n_in; i++) bits += memcmp(&input[i], &v.input[i], sizeof(float)) != 0;
    if (bits) { ok = false; stage_fail[0]++; }

    // couches : écart <= tolérance (on part de la vraie entrée, pour isoler chaque étage)
    nn::Result r = nn::forward(m, v.input, buf_a, buf_b, trace);
    float diff[kNumLayers];
    for (int l = 0; l < m.n_layers; l++) {
      diff[l] = max_abs_diff(layer_out[l], v.outputs[l], m.layers[l].n_out);
      worst[l] = fmaxf(worst[l], diff[l]);
      if (!(diff[l] <= kTolerance)) { ok = false; stage_fail[1 + l]++; }
    }
    if (r.prediction != v.py_pred) { ok = false; stage_fail[1 + kNumLayers]++; }

    // le chemin profilé doit donner exactement les mêmes sorties
    float out_p[kNumLayers][kMaxWidth];
    float *trace_p[kNumLayers];
    for (int l = 0; l < kNumLayers; l++) trace_p[l] = out_p[l];
    nn::Profile prof{fake_clock, {0}, {0}};
    nn::forward(m, v.input, buf_a, buf_b, trace_p, &prof);
    for (int l = 0; l < m.n_layers; l++)
      if (memcmp(out_p[l], layer_out[l], m.layers[l].n_out * sizeof(float)) != 0) {
        ok = false;
        profile_diff++;
      }

    printf(" %3d %5d   %d    %d   %d   %-9s ", t, v.test_index, v.label, v.py_pred, r.prediction,
           bits ? "DIFFÉRENT" : "identique");
    for (int l = 0; l < m.n_layers; l++) printf("%.1e    ", diff[l]);
    printf("%s\n", ok ? "ok" : "ÉCHEC");
    failures += !ok;
  }

  printf("\nÉcart maximal par couche :");
  for (int l = 0; l < m.n_layers; l++) printf(" couche %d %.1e", l, worst[l]);
  printf("\nÉchecs par étage : entrée %d", stage_fail[0]);
  for (int l = 0; l < m.n_layers; l++) printf(", couche %d %d", l, stage_fail[1 + l]);
  printf(", prédiction %d\n", stage_fail[1 + kNumLayers]);
  printf("Chemin profilé identique au bit près : %s\n", profile_diff ? "NON" : "oui");
  printf("%s : %d/%d images conformes\n", failures ? "ÉCHEC" : "SUCCÈS",
         kNumTestVectors - failures, kNumTestVectors);
  return failures ? 1 : 0;
}

// ESPHome compile le firmware en -Os (taille). Pour ce seul fichier, -O2 :
// la boucle du produit scalaire passe de 9 instructions avec un saut (~13
// cycles par multiplication-addition) à 5 instructions dans une boucle
// matérielle Xtensa (~5-6 cycles), pour ~130 octets de code en plus.
// Mesuré et désassemblé avec xtensa-esp32-elf-g++ 14.2.0 (esp-14.2.0_20260121).
#pragma GCC optimize("O2")

#include "nn_engine.h"

#include <cmath>
#include <cstring>

namespace nn {

size_t input_size(const Model &m) {
  const size_t side = m.raw_side / m.factor;
  return side * side;
}

void preprocess(const Model &m, const uint8_t *image, float *out) {
  const int f = m.factor;
  const int side = m.raw_side / f;
  for (int r = 0; r < side; r++) {
    for (int c = 0; c < side; c++) {
      uint32_t sum = 0;                                   // somme entière : exacte
      for (int i = 0; i < f; i++)
        for (int j = 0; j < f; j++)
          sum += image[(r * f + i) * m.raw_side + c * f + j];
      out[r * side + c] = (float) sum / (float) m.divisor;  // une seule division
    }
  }
}

static void sigmoid(float *v, int n) {
  // expf(-z) devient +inf pour z très négatif : 1 / (1 + inf) = 0, sans erreur.
  for (int j = 0; j < n; j++) v[j] = 1.0f / (1.0f + expf(-v[j]));
}

static void softmax(float *v, int n) {
  // On soustrait le maximum : tous les exposants sont <= 0, expf ne déborde pas
  // (il déborde vers z = 88 en float32), et le résultat mathématique est le même.
  float mx = v[0];
  for (int j = 1; j < n; j++) mx = v[j] > mx ? v[j] : mx;
  float sum = 0.0f;
  for (int j = 0; j < n; j++) {
    v[j] = expf(v[j] - mx);
    sum += v[j];
  }
  for (int j = 0; j < n; j++) v[j] /= sum;
}

void matvec(const Layer &layer, const float *in, float *out) {
  const float *w = layer.W;
  for (int j = 0; j < layer.n_out; j++) {
    float z = layer.b[j];
    for (int k = 0; k < layer.n_in; k++) z += w[k] * in[k];   // ligne j de W, contiguë
    out[j] = z;
    w += layer.n_in;                                         // ligne suivante : j * n_in
  }
}

void activate(const Layer &layer, float *v) {
  if (layer.activation == Activation::kSigmoid)
    sigmoid(v, layer.n_out);
  else
    softmax(v, layer.n_out);
}

void dense(const Layer &layer, const float *in, float *out) {
  matvec(layer, in, out);
  activate(layer, out);
}

Result forward(const Model &m, const float *input, float *buf_a, float *buf_b,
               float *const *trace, Profile *profile) {
  const float *in = input;
  float *out = buf_a;
  int n = 0;
  for (int l = 0; l < m.n_layers; l++) {
    const Layer &layer = m.layers[l];
    if (profile != nullptr && l < kMaxProfiledLayers) {
      const uint32_t t0 = profile->clock();
      matvec(layer, in, out);
      const uint32_t t1 = profile->clock();
      activate(layer, out);
      const uint32_t t2 = profile->clock();
      profile->matvec[l] += t1 - t0;
      profile->activation[l] += t2 - t1;
    } else {
      dense(layer, in, out);
    }
    n = layer.n_out;
    if (trace != nullptr && trace[l] != nullptr)
      memcpy(trace[l], out, n * sizeof(float));
    in = out;
    out = (out == buf_a) ? buf_b : buf_a;                    // tampons alternés
  }

  Result r{0, in[0], 0.0f};
  float second = -INFINITY;
  for (int j = 1; j < n; j++) {
    if (in[j] > r.confidence) {
      second = r.confidence;
      r.confidence = in[j];
      r.prediction = j;
    } else if (in[j] > second) {
      second = in[j];
    }
  }
  r.margin = r.confidence - second;
  return r;
}

}  // namespace nn

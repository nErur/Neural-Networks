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
  if (m.n_input != 0) return m.n_input;
  const size_t side = m.raw_side / m.factor;
  return side * side;
}

size_t weight_count(const Layer &layer) {
  switch (layer.type) {
    case LayerType::kConv1d:
      return (size_t) layer.c_out * layer.c_in * layer.kernel;
    case LayerType::kMaxPool1d:
      return 0;
    default:
      return (size_t) layer.n_in * layer.n_out;
  }
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

// Convolution 1D « valide » : pour chaque filtre f et chaque position o, le même petit
// produit scalaire (c_in × kernel poids) glisse le long du signal.
static void conv1d(const Layer &L, const float *in, float *out) {
  const int C = L.c_in, Lin = L.l_in, F = L.c_out, Lo = L.l_out, k = L.kernel, s = L.stride;
  for (int f = 0; f < F; f++) {
    const float *wf = L.W + f * C * k;                       // les C × k poids du filtre f
    for (int o = 0; o < Lo; o++) {
      float z = L.b[f];
      for (int c = 0; c < C; c++) {
        const float *wc = wf + c * k;
        const float *xc = in + c * Lin + o * s;              // fenêtre du canal c qui commence en o·s
        for (int j = 0; j < k; j++) z += wc[j] * xc[j];
      }
      out[f * Lo + o] = z;
    }
  }
}

// Max-pooling 1D : le plus grand de chaque groupe de `kernel` secondes, canal par canal.
static void maxpool1d(const Layer &L, const float *in, float *out) {
  const int C = L.c_in, Lin = L.l_in, Lo = L.l_out, t = L.kernel;
  for (int c = 0; c < C; c++) {
    for (int o = 0; o < Lo; o++) {
      const float *x = in + c * Lin + o * t;
      float m = x[0];
      for (int i = 1; i < t; i++) m = x[i] > m ? x[i] : m;
      out[c * Lo + o] = m;
    }
  }
}

void matvec(const Layer &layer, const float *in, float *out) {
  if (layer.type == LayerType::kConv1d) {
    conv1d(layer, in, out);
    return;
  }
  if (layer.type == LayerType::kMaxPool1d) {
    maxpool1d(layer, in, out);
    return;
  }
  const float *w = layer.W;
  for (int j = 0; j < layer.n_out; j++) {
    float z = layer.b[j];
    for (int k = 0; k < layer.n_in; k++) z += w[k] * in[k];   // ligne j de W, contiguë
    out[j] = z;
    w += layer.n_in;                                         // ligne suivante : j * n_in
  }
}

void activate(const Layer &layer, float *v) {
  if (layer.type == LayerType::kMaxPool1d) return;          // pas d'activation après un pooling
  switch (layer.activation) {
    case Activation::kSigmoid:
      sigmoid(v, layer.n_out);
      break;
    case Activation::kReLU:                                  // un maximum : quelques cycles
      for (int j = 0; j < layer.n_out; j++) v[j] = v[j] > 0.0f ? v[j] : 0.0f;
      break;
    case Activation::kTanh:
      for (int j = 0; j < layer.n_out; j++) v[j] = tanhf(v[j]);
      break;
    default:
      softmax(v, layer.n_out);
  }
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

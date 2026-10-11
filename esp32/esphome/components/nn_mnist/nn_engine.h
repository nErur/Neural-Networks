#pragma once
// Moteur d'inférence pur : forward pass d'un réseau dense, en float32.
// Aucune dépendance à ESPHome ni à Arduino, aucune allocation dynamique :
// le même code est compilé sur le PC (esp32/tests) et sur l'ESP.

#include <cstddef>
#include <cstdint>

namespace nn {

enum class Activation : uint8_t { kSigmoid, kSoftmax };

// Une couche dense : z = W·a + b, W rangé ligne par ligne,
// le poids de l'entrée k vers le neurone j est W[j * n_in + k].
struct Layer {
  uint16_t n_in;
  uint16_t n_out;
  const float *W;
  const float *b;
  Activation activation;
};

struct Model {
  const char *id;         // nom | hachage des poids | précision | date
  uint8_t n_layers;
  const Layer *layers;
  // prétraitement : image brute raw_side×raw_side uint8, somme entière des
  // blocs factor×factor, puis (float)somme / divisor
  uint8_t raw_side;
  uint8_t factor;
  uint16_t divisor;
};

struct Result {
  uint8_t prediction;     // indice de la plus grande sortie
  float confidence;       // plus grande sortie (probabilité si softmax)
  float margin;           // écart entre les deux plus grandes sorties
};

// Taille de l'entrée du réseau après prétraitement : (raw_side / factor)².
size_t input_size(const Model &m);

// Image brute (raw_side² octets, ligne par ligne) -> entrée du réseau.
// Somme entière exacte puis une seule division : identique au bit près à numpy.
void preprocess(const Model &m, const uint8_t *image, float *out);

// Une couche : out = activation(W·in + b). in et out ne doivent pas se chevaucher.
void dense(const Layer &layer, const float *in, float *out);

// Toutes les couches. buf_a et buf_b : tampons de travail d'au moins
// max(n_out) floats chacun. trace (facultatif, nullptr sur l'ESP) : un pointeur
// par couche, où recopier la sortie de cette couche pour la comparer aux références.
Result forward(const Model &m, const float *input, float *buf_a, float *buf_b,
               float *const *trace = nullptr);

}  // namespace nn

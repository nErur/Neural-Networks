#pragma once
// Moteur d'inférence pur (aucune dépendance à ESPHome) : types partagés.
// Étape 1 : seuls les types, utilisés par model_weights.h (généré par export.py).
// Étape 3 : ajout de preprocess(), dense() et forward().

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

}  // namespace nn

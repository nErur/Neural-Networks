#pragma once
// Moteur d'inférence pur : forward pass d'un réseau (couches denses, convolution 1D,
// max-pooling 1D), en float32.
// Aucune dépendance à ESPHome ni à Arduino, aucune allocation dynamique :
// le même code est compilé sur le PC (esp32/tests) et sur l'ESP.

#include <cstddef>
#include <cstdint>

namespace nn {

enum class Activation : uint8_t { kSigmoid, kSoftmax, kReLU, kTanh };
enum class LayerType : uint8_t { kDense, kConv1d, kMaxPool1d };

// Une couche. Les signaux 1D sont rangés canal par canal : l'élément (c, t) d'un
// signal de c_in canaux × l_in secondes est en [c * l_in + t] (comme numpy (C, L)).
//   dense     : z = W·a + b ; le poids de l'entrée k vers le neurone j est W[j * n_in + k].
//               Après une convolution, l'entrée est le signal (C, L) aplati tel quel.
//   conv1d    : convolution « valide » (corrélation, comme le banc) :
//               z[f, o] = b[f] + somme_c somme_j W[(f * c_in + c) * kernel + j] · a[c, o * stride + j]
//               l_out = (l_in - kernel) / stride + 1 ; sortie (c_out = filtres) × l_out.
//   maxpool1d : sortie[c, o] = max_t a[c, o * kernel + t], t < kernel ; l_out = l_in / kernel
//               (arrondi inférieur : les dernières secondes en trop sont ignorées, comme le banc).
//               Ni poids, ni biais, ni activation.
// n_in et n_out sont toujours les tailles totales (c × l pour un signal).
// Les champs ajoutés ont des valeurs par défaut : {n_in, n_out, W, b, act} reste une couche dense.
struct Layer {
  uint16_t n_in;
  uint16_t n_out;
  const float *W;
  const float *b;
  Activation activation;
  LayerType type = LayerType::kDense;
  uint16_t c_in = 0, l_in = 0, c_out = 0, l_out = 0;
  uint8_t kernel = 0, stride = 1;
};

// Nombre de poids (hors biais) d'une couche ; 0 pour le max-pooling.
size_t weight_count(const Layer &layer);

struct Model {
  const char *id;         // nom | hachage des poids | précision | date
  uint8_t n_layers;
  const Layer *layers;
  // prétraitement : image brute raw_side×raw_side uint8, somme entière des
  // blocs factor×factor, puis (float)somme / divisor
  uint8_t raw_side;
  uint8_t factor;
  uint16_t divisor;
  // taille de l'entrée quand elle ne vient pas d'une image (signal : canaux × longueur) ; 0 = image
  uint16_t n_input = 0;
};

struct Result {
  uint8_t prediction;     // indice de la plus grande sortie
  float confidence;       // plus grande sortie (probabilité si softmax)
  float margin;           // écart entre les deux plus grandes sorties
};

// Profilage (facultatif) : forward() note le temps du produit matriciel et de
// l'activation de chaque couche, mesuré avec l'horloge fournie (sur l'ESP : le
// compteur de cycles du processeur). Les durées s'ajoutent d'un appel à l'autre.
static const int kMaxProfiledLayers = 8;
struct Profile {
  uint32_t (*clock)();
  uint32_t matvec[kMaxProfiledLayers];
  uint32_t activation[kMaxProfiledLayers];
};

// Taille de l'entrée du réseau : n_input s'il est donné, sinon (raw_side / factor)².
size_t input_size(const Model &m);

// Image brute (raw_side² octets, ligne par ligne) -> entrée du réseau.
// Somme entière exacte puis une seule division : identique au bit près à numpy.
void preprocess(const Model &m, const uint8_t *image, float *out);

// Une couche : out = activation(W·in + b) (ou convolution, ou max-pooling selon layer.type).
// in et out ne doivent pas se chevaucher. matvec (le calcul linéaire, ou le pooling) puis
// activate, appelées par dense, sont séparées pour le profilage.
void matvec(const Layer &layer, const float *in, float *out);
void activate(const Layer &layer, float *v);
void dense(const Layer &layer, const float *in, float *out);

// Toutes les couches. buf_a et buf_b : tampons de travail d'au moins
// max(n_out) floats chacun (pour une convolution : c_out × l_out). trace (facultatif, nullptr sur l'ESP) : un pointeur
// par couche, où recopier la sortie de cette couche pour la comparer aux références.
// profile (facultatif) : durées par couche, voir Profile.
Result forward(const Model &m, const float *input, float *buf_a, float *buf_b,
               float *const *trace = nullptr, Profile *profile = nullptr);

}  // namespace nn

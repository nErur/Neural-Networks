#pragma once

#include "esphome/core/component.h"

namespace esphome {
namespace nn_mnist {

// Étape 0 : composant vide, sert seulement à valider la chaîne de compilation.
class NnMnist : public Component {
 public:
  void setup() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::LATE; }
};

}  // namespace nn_mnist
}  // namespace esphome

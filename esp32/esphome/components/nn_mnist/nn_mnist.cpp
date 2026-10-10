#include "nn_mnist.h"
#include "esphome/core/log.h"

namespace esphome {
namespace nn_mnist {

static const char *const TAG = "nn_mnist";

void NnMnist::setup() { ESP_LOGI(TAG, "hello depuis nn_mnist"); }

void NnMnist::dump_config() { ESP_LOGCONFIG(TAG, "nn_mnist : composant vide (étape 0), v1"); }

}  // namespace nn_mnist
}  // namespace esphome

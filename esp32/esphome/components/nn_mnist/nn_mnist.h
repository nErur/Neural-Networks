#pragma once
// Liaison ESPHome : reçoit les images, appelle le moteur (nn_engine), publie
// les résultats. Aucun calcul de réseau ici.

#include <cstdint>

#include "esphome/core/component.h"
#include "esphome/core/defines.h"
#include "esphome/core/log.h"
#include "esphome/components/output/binary_output.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/text_sensor/text_sensor.h"


namespace esphome {
namespace nn_mnist {

static const int kRawSize = 28 * 28;

class NnMnist : public Component {
 public:
  void setup() override;
  void loop() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::LATE; }

  // Entités (toutes facultatives)
  void set_prediction_sensor(sensor::Sensor *s) { prediction_ = s; }
  void set_confidence_sensor(sensor::Sensor *s) { confidence_ = s; }
  void set_margin_sensor(sensor::Sensor *s) { margin_ = s; }
  void set_inference_time_sensor(sensor::Sensor *s) { inference_time_ = s; }
  void set_expected_label_sensor(sensor::Sensor *s) { expected_label_ = s; }
  void set_max_error_sensor(sensor::Sensor *s) { max_error_ = s; }
  void set_model_id_text_sensor(text_sensor::TextSensor *s) { model_id_ = s; }
  void set_autotest_text_sensor(text_sensor::TextSensor *s) { autotest_ = s; }

  // Témoin d'activité (facultatif) : allumé à chaque inférence, éteint
  // activity_duration_ ms après la dernière (300 µs seraient invisibles).
  void set_activity_output(output::BinaryOutput *o) { activity_ = o; }
  void set_activity_duration(uint32_t ms) { activity_duration_ = ms; }

  // Les deux portes d'entrée : elles ne font que déposer une demande,
  // traitée par loop() (une seule chaîne de traitement).
  void next_test_image();
  void run_autotest();

  // Image brute 28×28, 0..255, ligne par ligne (action API « classify »).
  // Modèle de fonction : accepte le type de tableau fourni par l'API.
  template<typename T> void submit_image(const T &image) {
    if (image.size() != kRawSize) {
      ESP_LOGW("nn_mnist", "image refusée : %d valeurs au lieu de %d", (int) image.size(), kRawSize);
      return;
    }
    for (int i = 0; i < kRawSize; i++) {
      if (image[i] < 0 || image[i] > 255) {
        ESP_LOGW("nn_mnist", "image refusée : pixel %d = %d hors de 0..255", i, (int) image[i]);
        return;
      }
      image_[i] = (uint8_t) image[i];
    }
    pending_test_ = -1;
    pending_ = Pending::kImage;
  }

 protected:
  enum class Pending : uint8_t { kNone, kImage, kAutotest };

  // Classe image_ ; test_index >= 0 : image de test, comparée aux références.
  void classify_(int test_index);
  void do_autotest_();
  void signal_activity_();

  Pending pending_{Pending::kNone};
  int pending_test_{-1};
  int next_test_{0};
  bool first_done_{false};

  uint8_t image_[kRawSize];   // tampons de calcul : dans nn_mnist.cpp (taille fixée par le modèle)

  sensor::Sensor *prediction_{nullptr};
  sensor::Sensor *confidence_{nullptr};
  sensor::Sensor *margin_{nullptr};
  sensor::Sensor *inference_time_{nullptr};
  sensor::Sensor *expected_label_{nullptr};
  sensor::Sensor *max_error_{nullptr};
  text_sensor::TextSensor *model_id_{nullptr};
  text_sensor::TextSensor *autotest_{nullptr};
  output::BinaryOutput *activity_{nullptr};
  uint32_t activity_duration_{150};
};

}  // namespace nn_mnist
}  // namespace esphome

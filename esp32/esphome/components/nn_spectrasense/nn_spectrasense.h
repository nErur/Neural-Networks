#pragma once
// Liaison ESPHome du réseau convolutif SpectraSense (étape 1 : autotest et profilage sur la carte).

#include <cstdint>

#include "esphome/core/component.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/text_sensor/text_sensor.h"

namespace esphome {
namespace nn_spectrasense {

class NnSpectrasense : public Component {
 public:
  void setup() override;
  void loop() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::LATE; }

  void set_inference_time_sensor(sensor::Sensor *s) { inference_time_ = s; }
  void set_max_error_sensor(sensor::Sensor *s) { max_error_ = s; }
  void set_model_id_text_sensor(text_sensor::TextSensor *s) { model_id_ = s; }
  void set_autotest_text_sensor(text_sensor::TextSensor *s) { autotest_ = s; }
  void set_weights_in_ram(bool v) { weights_in_ram_ = v; }

  void run_autotest();     // les fenêtres embarquées, une par passage dans loop()
  void run_profile();      // cycles par couche, moyenne sur les fenêtres embarquées

 protected:
  void autotest_step_();
  void profile_step_();
  void profile_report_();
  bool copy_weights_to_ram_();

  bool weights_in_ram_{true};
  int autotest_index_{-1};
  int autotest_ok_{0};
  float autotest_worst_{0.0f};
  uint32_t autotest_us_{0};
  int profile_index_{-1};
  uint32_t prof_total_{0};
  uint32_t prof_calc_[8]{}, prof_act_[8]{};
  uint32_t prof_c0_{0}, prof_u0_{0};

  sensor::Sensor *inference_time_{nullptr};
  sensor::Sensor *max_error_{nullptr};
  text_sensor::TextSensor *model_id_{nullptr};
  text_sensor::TextSensor *autotest_{nullptr};
};

}  // namespace nn_spectrasense
}  // namespace esphome

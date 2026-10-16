#pragma once
// Liaison ESPHome du réseau convolutif SpectraSense : autotest, profilage, et chaîne complète
// (ss::EntreeHa -> ss::Chaine) alimentée par un capteur de puissance, avec publication des états.

#include <cstdint>

#include "esphome/core/component.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/text_sensor/text_sensor.h"
#include "esphome/components/binary_sensor/binary_sensor.h"

#include "esphome/core/preferences.h"
#ifdef USE_TIME
#include "esphome/components/time/real_time_clock.h"
#endif

#include "chaine.h"

namespace esphome {
namespace nn_spectrasense {

class NnSpectrasense : public Component, public ss::Recepteur {
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

  // chaîne en direct
  void set_power_sensor(sensor::Sensor *s) { power_ = s; }
  void set_clock_delay(float s) { clock_delay_ = s; }
  void set_reglages(float seuil, float seuil_pic, int tenue) {
    reglages_.seuil = seuil;
    reglages_.seuil_pic = seuil_pic;
    reglages_.tenue = tenue;
  }
  void set_machine_text_sensor(text_sensor::TextSensor *s) { machine_ = s; }
  void set_network_class_text_sensor(text_sensor::TextSensor *s) { classe_ = s; }
  void set_toaster_binary_sensor(binary_sensor::BinarySensor *s) { grille_pain_ = s; }
  void set_kettle_binary_sensor(binary_sensor::BinarySensor *s) { bouilloire_ = s; }
  void set_unknown_load_binary_sensor(binary_sensor::BinarySensor *s) { inconnu_ = s; }
  void set_gate_binary_sensor(binary_sensor::BinarySensor *s) { porte_ = s; }
  void set_invalid_seconds_sensor(sensor::Sensor *s) { invalides_ = s; }
  void set_coffee_count_sensor(sensor::Sensor *s) { cafes_ = s; }
  void set_network_confidence_sensor(sensor::Sensor *s) { confiance_ = s; }
  // probabilité brute de la classe k du réseau (%), avant l'automate
  void set_class_probability_sensor(int k, sensor::Sensor *s) { if (k >= 0 && k < ss::kClasses) proba_[k] = s; }
#ifdef USE_TIME
  // horloge réelle : l'état long de la machine survit aux redémarrages (mémoire en flash)
  void set_time(time::RealTimeClock *t) { time_ = t; }
#endif

  void recevoir(const ss::Case &c) override;   // une case de la grille (une par seconde)

  void run_autotest();     // les fenêtres embarquées, une par passage dans loop()
  void run_profile();      // cycles par couche, moyenne sur les fenêtres embarquées

 protected:
  void autotest_step_();
  void profile_step_();
  void profile_report_();
  bool copy_weights_to_ram_();
  void mesure_(float p);
  void bilan_minute_();
  bool epoch_(int64_t &e) const;   // heure réelle (s), si connue
  void restaurer_();
  void memoriser_();
  static double maintenant_();

  bool weights_in_ram_{true};
  int autotest_index_{-1};
  int autotest_ok_{0};
  float autotest_worst_{0.0f};
  uint32_t autotest_us_{0};
  int profile_index_{-1};
  uint32_t prof_total_{0};
  uint32_t prof_calc_[8]{}, prof_act_[8]{};
  uint32_t prof_c0_{0}, prof_u0_{0};

  // état long gardé en flash : {état, heure réelle de la dernière activité / du dernier pic}
  struct Memoire {
    uint8_t etat;
    int64_t epoch_ref;
  };
  ESPPreferenceObject pref_;
  ESPPreferenceObject pref_cafes_;   // compteur de cafés, gardé en flash
  uint32_t n_cafes_{0};
  bool restaure_{false};
  int mem_etat_{-1};
  int64_t mem_ref_{0};
#ifdef USE_TIME
  time::RealTimeClock *time_{nullptr};
#endif
  sensor::Sensor *power_{nullptr};
  float clock_delay_{2.0f};
  ss::Reglages reglages_;
  ss::EntreeHa *entree_{nullptr};
  ss::Chaine *chaine_{nullptr};
  int pub_machine_{-1}, pub_classe_{-1}, pub_gp_{-1}, pub_bouilloire_{-1}, pub_inconnu_{-1}, pub_porte_{-1};
  uint32_t n_cases_{0}, n_invalides_{0}, n_lectures_{0};
  int64_t pub_invalides_{-1};
  uint32_t inf_us_{0}, inf_n_{0}, t_bilan_{0};

  text_sensor::TextSensor *machine_{nullptr};
  text_sensor::TextSensor *classe_{nullptr};
  binary_sensor::BinarySensor *grille_pain_{nullptr};
  binary_sensor::BinarySensor *bouilloire_{nullptr};
  binary_sensor::BinarySensor *inconnu_{nullptr};
  binary_sensor::BinarySensor *porte_{nullptr};
  sensor::Sensor *invalides_{nullptr};
  sensor::Sensor *cafes_{nullptr};
  sensor::Sensor *confiance_{nullptr};
  float pub_confiance_{-100.0f};
  sensor::Sensor *proba_[ss::kClasses]{};
  float pub_proba_[ss::kClasses]{-100.0f, -100.0f, -100.0f, -100.0f, -100.0f, -100.0f,
                                 -100.0f, -100.0f, -100.0f, -100.0f, -100.0f, -100.0f};
  sensor::Sensor *inference_time_{nullptr};
  sensor::Sensor *max_error_{nullptr};
  text_sensor::TextSensor *model_id_{nullptr};
  text_sensor::TextSensor *autotest_{nullptr};
};

}  // namespace nn_spectrasense
}  // namespace esphome

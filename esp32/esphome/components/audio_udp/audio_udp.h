#pragma once
// Micro I2S (INMP441) -> PC en UDP, pour construire le dataset audio de SpectraSense.
// Le son brut (mono, 16 ou 24 bits) part par paquets ; le PC horodate, assemble et trie.
// Aucun traitement intelligent ici : l'ESP reste libre pour le réseau.

#include <cstdint>
#include <string>
#include <vector>

#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <lwip/sockets.h>

#include "esphome/core/component.h"
#include "esphome/components/microphone/microphone.h"
#include "esphome/components/sensor/sensor.h"

namespace esphome::audio_udp {

// En-tête de chaque paquet (petit-boutiste, 28 octets), suivi de N échantillons PCM signés
// petit-boutistes (la disposition d'un WAV) :
//   0  'SSA1' (24 bits, 3 octets par échantillon) ou 'SSA2' (16 bits, 2 octets)
//   4  uint32 session              tirée au démarrage : un changement = l'ESP a redémarré
//   8  uint32 seq                  numéro du paquet dans la session (trous = paquets perdus ou abandonnés)
//  12  uint64 index                rang du premier échantillon depuis le début de la session
//  20  uint64 t_us                 esp_timer (µs depuis le démarrage) à la lecture du premier morceau
static const size_t TAILLE_ENTETE = 28;

class AudioUdp : public PollingComponent {
 public:
  void setup() override;
  void update() override;
  void dump_config() override;
  // après le micro (DATA) : son sémaphore doit exister avant qu'on le démarre
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  void set_microphone(microphone::Microphone *m) { micro_ = m; }
  void set_destination(const std::string &hote, uint16_t port) {
    hote_ = hote;
    port_ = port;
  }
  void set_samples_per_packet(uint16_t n) { par_paquet_ = n; }
  void set_sample_bits(uint8_t bits) { octets_ = bits / 8; }
  void set_level_sensor(sensor::Sensor *s) { niveau_ = s; }
  void set_peak_sensor(sensor::Sensor *s) { crete_ = s; }
  void set_packets_sensor(sensor::Sensor *s) { paquets_ = s; }
  void set_errors_sensor(sensor::Sensor *s) { erreurs_ = s; }
  void set_dropped_sensor(sensor::Sensor *s) { abandons_capteur_ = s; }

  // Vrai si l'envoi est en pause ou s'est mis en pause dans les fenetre_ms dernières millisecondes
  // (liaison saturée). Sert au témoin lumineux, qui doit marcher même sans HA.
  bool liaison_saturee(uint32_t fenetre_ms) const {
    const uint32_t derniere = this->derniere_pause_ms_;
    if (derniere == 0)
      return false;
    // négatif pendant la pause (la fin est dans le futur), puis temps écoulé depuis la fin
    const int32_t depuis = (int32_t) ((uint32_t) (esp_timer_get_time() / 1000) - derniere);
    return depuis < (int32_t) fenetre_ms;
  }

 protected:
  void recevoir_(const std::vector<uint8_t> &donnees);  // tâche du micro (priorité 23)
  void envoyer_();                                       // idem

  microphone::Microphone *micro_{nullptr};
  std::string hote_;
  uint16_t port_{50005};
  uint16_t par_paquet_{480};  // 30 ms à 16 kHz
  uint8_t octets_{2};         // par échantillon : 2 (16 bits) ou 3 (24 bits)

  int sock_{-1};
  struct sockaddr_in dest_ {};

  // paquet en cours (écrit seulement par la tâche du micro)
  std::vector<uint8_t> paquet_;
  uint16_t remplis_{0};
  uint32_t session_{0};
  uint32_t seq_{0};
  uint64_t index_{0};
  int64_t t_dernier_morceau_{0};
  // passe-haut du niveau (coefficients calculés au démarrage, état tenu par la tâche du micro)
  float hp_b0_{1.0f}, hp_a1_{0.0f}, hp_a2_{0.0f};
  float hp_z1_{0.0f}, hp_z2_{0.0f};

  // pause de l'envoi quand la liaison sature (tâche du micro) : après ECHECS_AVANT_PAUSE échecs
  // de suite, on n'envoie plus rien pendant pause_us_ ; la pause double tant que ça échoue encore.
  uint8_t echecs_suite_{0};
  bool en_difficulte_{false};
  int64_t pause_jusqu_a_{0};
  int64_t pause_us_{0};
  volatile uint32_t derniere_pause_ms_{0};  // fin de la dernière pause décidée (ms depuis le démarrage)

  // statistiques partagées avec update() (boucle principale)
  portMUX_TYPE verrou_ = portMUX_INITIALIZER_UNLOCKED;
  double somme2_{0.0};  // somme des carrés du signal filtré (pleine échelle = 1)
  uint32_t n_stats_{0};
  int32_t max_abs_{0};
  uint32_t envoyes_{0};
  uint32_t echecs_{0};
  uint32_t abandons_{0};
  uint32_t pauses_{0};
  uint32_t abandons_publies_{0};

  sensor::Sensor *niveau_{nullptr};
  sensor::Sensor *crete_{nullptr};
  sensor::Sensor *paquets_{nullptr};
  sensor::Sensor *erreurs_{nullptr};
  sensor::Sensor *abandons_capteur_{nullptr};
};

}  // namespace esphome::audio_udp

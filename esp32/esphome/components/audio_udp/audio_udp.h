#pragma once
// Micro I2S (INMP441) -> PC en UDP, pour construire le dataset audio de SpectraSense.
// Le son brut (24 bits utiles, mono) part par paquets ; le PC horodate, assemble et trie.
// Aucun traitement intelligent ici : l'ESP reste libre pour le réseau.

#include <cstdint>
#include <string>
#include <vector>

#include <freertos/FreeRTOS.h>
#include <lwip/sockets.h>

#include "esphome/core/component.h"
#include "esphome/components/microphone/microphone.h"
#include "esphome/components/sensor/sensor.h"

namespace esphome::audio_udp {

// En-tête de chaque paquet (petit-boutiste, 28 octets), suivi de N échantillons de 3 octets
// (PCM 24 bits signé petit-boutiste, la disposition d'un WAV 24 bits) :
//   0  'SSA1'                      magique + version du format
//   4  uint32 session              tirée au démarrage : un changement = l'ESP a redémarré
//   8  uint32 seq                  numéro du paquet dans la session (trous = paquets perdus)
//  12  uint64 index                rang du premier échantillon depuis le début de la session
//  20  uint64 t_us                 esp_timer (µs depuis le démarrage) à la lecture du premier morceau
static const size_t TAILLE_ENTETE = 28;
static const size_t OCTETS_PAR_ECHANTILLON = 3;

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
  void set_level_sensor(sensor::Sensor *s) { niveau_ = s; }
  void set_peak_sensor(sensor::Sensor *s) { crete_ = s; }
  void set_packets_sensor(sensor::Sensor *s) { paquets_ = s; }
  void set_errors_sensor(sensor::Sensor *s) { erreurs_ = s; }

 protected:
  void recevoir_(const std::vector<uint8_t> &donnees);  // tâche du micro (priorité 23)
  void envoyer_();                                       // idem

  microphone::Microphone *micro_{nullptr};
  std::string hote_;
  uint16_t port_{50005};
  uint16_t par_paquet_{480};  // 30 ms à 16 kHz : 28 + 1 440 octets, sous la MTU

  int sock_{-1};
  struct sockaddr_in dest_ {};

  // paquet en cours (écrit seulement par la tâche du micro)
  std::vector<uint8_t> paquet_;
  uint16_t remplis_{0};
  uint32_t session_{0};
  uint32_t seq_{0};
  uint64_t index_{0};
  int64_t t_dernier_morceau_{0};

  // statistiques partagées avec update() (boucle principale)
  portMUX_TYPE verrou_ = portMUX_INITIALIZER_UNLOCKED;
  int64_t somme_{0};
  int64_t somme2_{0};
  uint32_t n_stats_{0};
  int32_t max_abs_{0};
  uint32_t envoyes_{0};
  uint32_t echecs_{0};

  sensor::Sensor *niveau_{nullptr};
  sensor::Sensor *crete_{nullptr};
  sensor::Sensor *paquets_{nullptr};
  sensor::Sensor *erreurs_{nullptr};
};

}  // namespace esphome::audio_udp

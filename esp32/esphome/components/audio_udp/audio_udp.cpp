#include "audio_udp.h"

#include <cerrno>
#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <esp_random.h>
#include <esp_timer.h>

#include "esphome/core/log.h"

namespace esphome::audio_udp {

static const char *const TAG = "audio_udp";
static const int64_t TROU_US = 200000;  // plus de 200 ms sans données : discontinuité (micro arrêté…)
static const float PLEINE_ECHELLE = 8388608.0f;  // 2^23
static const float COUPURE_NIVEAU_HZ = 100.0f;   // passe-haut du capteur de niveau (Butterworth, 2e ordre)
static const uint8_t ECHECS_AVANT_PAUSE = 3;     // échecs d'envoi de suite avant de faire une pause
static const int64_t PAUSE_MIN_US = 500000;      // première pause : 0,5 s
static const int64_t PAUSE_MAX_US = 8000000;     // la pause double jusqu'à 8 s si ça échoue encore

static void ecrire_u32(uint8_t *p, uint32_t v) {
  for (int i = 0; i < 4; i++)
    p[i] = (v >> (8 * i)) & 0xFF;
}
static void ecrire_u64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++)
    p[i] = (v >> (8 * i)) & 0xFF;
}

void AudioUdp::setup() {
  audio::AudioStreamInfo info = this->micro_->get_audio_stream_info();
  if (info.get_bits_per_sample() != 32 || info.get_channels() != 1) {
    ESP_LOGE(TAG, "Micro attendu en 32 bits mono (bits_per_sample: 32bit, channel: left ou right), reçu %u bits, %u canal(aux)",
             info.get_bits_per_sample(), info.get_channels());
    this->mark_failed();
    return;
  }

  this->sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  if (this->sock_ < 0) {
    ESP_LOGE(TAG, "Création du socket UDP impossible (errno %d)", errno);
    this->mark_failed();
    return;
  }
  this->dest_.sin_family = AF_INET;
  this->dest_.sin_port = htons(this->port_);
  if (inet_aton(this->hote_.c_str(), &this->dest_.sin_addr) == 0) {
    ESP_LOGE(TAG, "Adresse invalide : %s", this->hote_.c_str());
    this->mark_failed();
    return;
  }

  // passe-haut du niveau : Butterworth du 2e ordre par transformation bilinéaire
  const float k = tanf((float) M_PI * COUPURE_NIVEAU_HZ / (float) info.get_sample_rate());
  const float q = (float) M_SQRT1_2;
  const float n = 1.0f / (1.0f + k / q + k * k);
  this->hp_b0_ = n;
  this->hp_a1_ = 2.0f * (k * k - 1.0f) * n;
  this->hp_a2_ = (1.0f - k / q + k * k) * n;

  this->paquet_.assign(TAILLE_ENTETE + this->octets_ * this->par_paquet_, 0);
  this->session_ = esp_random();
  this->pause_us_ = PAUSE_MIN_US;
  memcpy(this->paquet_.data(), this->octets_ == 3 ? "SSA1" : "SSA2", 4);
  ecrire_u32(this->paquet_.data() + 4, this->session_);

  this->micro_->add_data_callback([this](const std::vector<uint8_t> &d) { this->recevoir_(d); });
  this->micro_->start();  // le commutateur « Micro » peut l'arrêter ensuite
}

void AudioUdp::recevoir_(const std::vector<uint8_t> &donnees) {
  const int64_t maintenant = esp_timer_get_time();
  if (this->remplis_ > 0 && maintenant - this->t_dernier_morceau_ > TROU_US)
    this->envoyer_();  // ne pas coller deux morceaux séparés par un trou dans le même paquet
  this->t_dernier_morceau_ = maintenant;

  const size_t n = donnees.size() / 4;
  const uint8_t *src = donnees.data();
  float somme2 = 0.0f;  // un morceau (16 ms) : pas de perte de précision en float
  int32_t max_abs = 0;

  for (size_t i = 0; i < n; i++, src += 4) {
    if (this->remplis_ == 0) {
      ecrire_u32(this->paquet_.data() + 8, this->seq_);
      ecrire_u64(this->paquet_.data() + 12, this->index_);
      ecrire_u64(this->paquet_.data() + 20, (uint64_t) maintenant);
    }
    // échantillon 32 bits petit-boutiste, 24 bits utiles en haut : on garde les octets 1 à 3 (24 bits)
    // ou 2 à 3 (16 bits ; le bruit propre du micro, ~-87 dBFS, reste au-dessus du pas de 16 bits)
    uint8_t *dst = this->paquet_.data() + TAILLE_ENTETE + this->octets_ * this->remplis_;
    if (this->octets_ == 3) {
      dst[0] = src[1];
      dst[1] = src[2];
      dst[2] = src[3];
    } else {
      dst[0] = src[2];
      dst[1] = src[3];
    }

    int32_t x = ((int32_t) ((uint32_t) src[1] << 8 | (uint32_t) src[2] << 16 | (uint32_t) src[3] << 24)) >> 8;
    // niveau : après un passe-haut à 100 Hz (forme directe II transposée, b1 = -2 b0, b2 = b0).
    // Sous 100 Hz le micro capte surtout des fluctuations de pression (souffle, ~-30 dBFS vers 1-2 Hz)
    // qui masqueraient le son audible (~-65 dBFS au bureau). La crête reste brute (saturation).
    const float xf = (float) x / PLEINE_ECHELLE;
    const float y = this->hp_b0_ * xf + this->hp_z1_;
    this->hp_z1_ = -2.0f * this->hp_b0_ * xf - this->hp_a1_ * y + this->hp_z2_;
    this->hp_z2_ = this->hp_b0_ * xf - this->hp_a2_ * y;
    somme2 += y * y;
    int32_t a = x < 0 ? -x : x;
    if (a > max_abs)
      max_abs = a;

    this->index_++;
    if (++this->remplis_ == this->par_paquet_)
      this->envoyer_();
  }

  portENTER_CRITICAL(&this->verrou_);
  this->somme2_ += somme2;
  this->n_stats_ += n;
  if (max_abs > this->max_abs_)
    this->max_abs_ = max_abs;
  portEXIT_CRITICAL(&this->verrou_);
}

void AudioUdp::envoyer_() {
  const int64_t maintenant = esp_timer_get_time();
  this->seq_++;  // même pour un paquet abandonné ou en échec : le PC voit le trou
  const uint16_t remplis = this->remplis_;
  this->remplis_ = 0;

  // pause en cours : on n'insiste pas, la liaison (et l'API, les mesures de HA) respire
  if (maintenant < this->pause_jusqu_a_) {
    portENTER_CRITICAL(&this->verrou_);
    this->abandons_++;
    portEXIT_CRITICAL(&this->verrou_);
    return;
  }

  const size_t taille = TAILLE_ENTETE + this->octets_ * remplis;
  int r = ::sendto(this->sock_, this->paquet_.data(), taille, MSG_DONTWAIT, (struct sockaddr *) &this->dest_,
                   sizeof(this->dest_));
  bool pause = false;
  if (r < 0) {
    // après une pause, un seul échec suffit à repartir en pause (plus longue)
    if (this->en_difficulte_ || ++this->echecs_suite_ >= ECHECS_AVANT_PAUSE) {
      this->pause_jusqu_a_ = maintenant + this->pause_us_;
      // fin de la pause : le témoin reste allumé pendant toute la pause puis encore la fenêtre choisie
      this->derniere_pause_ms_ = std::max<uint32_t>(1, (uint32_t) (this->pause_jusqu_a_ / 1000));
      this->pause_us_ = std::min(2 * this->pause_us_, PAUSE_MAX_US);
      this->en_difficulte_ = true;
      this->echecs_suite_ = 0;
      pause = true;
    }
  } else {
    this->echecs_suite_ = 0;
    this->en_difficulte_ = false;
    this->pause_us_ = PAUSE_MIN_US;
  }
  portENTER_CRITICAL(&this->verrou_);
  if (r < 0)
    this->echecs_++;
  else
    this->envoyes_++;
  if (pause)
    this->pauses_++;
  portEXIT_CRITICAL(&this->verrou_);
}

void AudioUdp::update() {
  portENTER_CRITICAL(&this->verrou_);
  double somme2 = this->somme2_;
  uint32_t n = this->n_stats_;
  int32_t max_abs = this->max_abs_;
  uint32_t envoyes = this->envoyes_, echecs = this->echecs_, abandons = this->abandons_, pauses = this->pauses_;
  this->somme2_ = 0.0;
  this->n_stats_ = 0;
  this->max_abs_ = 0;
  this->pauses_ = 0;
  portEXIT_CRITICAL(&this->verrou_);

  if (pauses > 0 || abandons != this->abandons_publies_)
    ESP_LOGW(TAG, "Liaison saturée : %" PRIu32 " pause(s) de l'envoi, %" PRIu32 " paquet(s) abandonné(s) depuis la dernière mise à jour",
             pauses, abandons - this->abandons_publies_);
  this->abandons_publies_ = abandons;
  if (this->paquets_ != nullptr)
    this->paquets_->publish_state(envoyes);
  if (this->erreurs_ != nullptr)
    this->erreurs_->publish_state(echecs);
  if (this->abandons_capteur_ != nullptr)
    this->abandons_capteur_->publish_state(abandons);
  if (n == 0)
    return;  // micro arrêté : on ne publie pas de niveau
  // niveau efficace au-dessus de 100 Hz, en dBFS (le signal filtré est déjà rapporté à la pleine échelle)
  double rms = std::sqrt(somme2 / n);
  if (this->niveau_ != nullptr)
    this->niveau_->publish_state(rms > 0 ? 20.0f * log10f((float) rms) : -144.0f);
  if (this->crete_ != nullptr)
    this->crete_->publish_state(max_abs > 0 ? 20.0f * log10f((float) max_abs / PLEINE_ECHELLE) : -144.0f);
}

void AudioUdp::dump_config() {
  ESP_LOGCONFIG(TAG,
                "Audio UDP :\n"
                "  Destination : %s:%u\n"
                "  Échantillons par paquet : %u (%u bits)\n"
                "  Session : %08" PRIx32,
                this->hote_.c_str(), this->port_, this->par_paquet_, 8 * this->octets_, this->session_);
  LOG_UPDATE_INTERVAL(this);
}

}  // namespace esphome::audio_udp

"""Composant ESPHome audio_udp : le son d'un micro I2S (INMP441) envoyé brut au PC en UDP.

Sert à construire le dataset audio de SpectraSense (pas destiné à la production) :
l'ESP lit le micro en 32 bits mono, garde les 24 bits utiles et les envoie par paquets
(format décrit dans audio_udp.h). Le PC horodate les paquets à la réception, écrit des WAV
et fait tout le traitement. Capteurs optionnels : niveau efficace et crête (dBFS) sur
l'intervalle de mise à jour, paquets envoyés et échecs d'envoi (totaux depuis le démarrage).
Le composant démarre le micro lui-même (après sa configuration).
"""
import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import microphone, sensor
from esphome.const import (
    CONF_ID,
    CONF_PORT,
    ENTITY_CATEGORY_DIAGNOSTIC,
    STATE_CLASS_MEASUREMENT,
    STATE_CLASS_TOTAL_INCREASING,
)

DEPENDENCIES = ["microphone"]
AUTO_LOAD = ["sensor"]

audio_udp_ns = cg.esphome_ns.namespace("audio_udp")
AudioUdp = audio_udp_ns.class_("AudioUdp", cg.PollingComponent)

CONF_MICROPHONE = "microphone"
CONF_HOST = "host"
CONF_SAMPLES_PER_PACKET = "samples_per_packet"
CONF_LEVEL = "level"
CONF_PEAK = "peak"
CONF_PACKETS = "packets"
CONF_ERRORS = "errors"

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(AudioUdp),
        cv.Required(CONF_MICROPHONE): cv.use_id(microphone.Microphone),
        cv.Required(CONF_HOST): cv.ipv4address,
        cv.Optional(CONF_PORT, default=50005): cv.port,
        # 28 + 3 × n octets doivent tenir dans un datagramme sans fragmentation (≤ 1 472)
        cv.Optional(CONF_SAMPLES_PER_PACKET, default=480): cv.int_range(min=16, max=481),
        cv.Optional(CONF_LEVEL): sensor.sensor_schema(
            unit_of_measurement="dBFS", accuracy_decimals=1, icon="mdi:microphone",
            state_class=STATE_CLASS_MEASUREMENT),
        cv.Optional(CONF_PEAK): sensor.sensor_schema(
            unit_of_measurement="dBFS", accuracy_decimals=1, icon="mdi:waveform",
            state_class=STATE_CLASS_MEASUREMENT),
        cv.Optional(CONF_PACKETS): sensor.sensor_schema(
            accuracy_decimals=0, icon="mdi:upload-network-outline",
            state_class=STATE_CLASS_TOTAL_INCREASING, entity_category=ENTITY_CATEGORY_DIAGNOSTIC),
        cv.Optional(CONF_ERRORS): sensor.sensor_schema(
            accuracy_decimals=0, icon="mdi:alert-circle-outline",
            state_class=STATE_CLASS_TOTAL_INCREASING, entity_category=ENTITY_CATEGORY_DIAGNOSTIC),
    }
).extend(cv.polling_component_schema("5s"))


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    micro = await cg.get_variable(config[CONF_MICROPHONE])
    cg.add(var.set_microphone(micro))
    cg.add(var.set_destination(str(config[CONF_HOST]), config[CONF_PORT]))
    cg.add(var.set_samples_per_packet(config[CONF_SAMPLES_PER_PACKET]))
    for cle, setter in ((CONF_LEVEL, "set_level_sensor"), (CONF_PEAK, "set_peak_sensor"),
                        (CONF_PACKETS, "set_packets_sensor"), (CONF_ERRORS, "set_errors_sensor")):
        if cle in config:
            s = await sensor.new_sensor(config[cle])
            cg.add(getattr(var, setter)(s))

"""Composant ESPHome nn_spectrasense (essais) : le réseau convolutif SpectraSense sur la carte.

Étape 1 : autotest (fenêtres de validation embarquées, comparées couche par couche aux
références numpy) et profilage (cycles par couche). Pas encore de PZEM ni d'automate.
Le moteur d'inférence est celui de nn_mnist (même code, une seule copie dans le firmware) :
nn_mnist doit donc aussi être chargé (components: [nn_mnist, nn_spectrasense]).
Headers : esp32/python/export_conv.py <id> --sortie <ce dossier> --espace nn_ss
          --inclure-moteur esphome/components/nn_mnist/nn_engine.h --par-classe 1
"""
import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import sensor, text_sensor
from esphome.const import CONF_ID, ENTITY_CATEGORY_DIAGNOSTIC, STATE_CLASS_MEASUREMENT

DEPENDENCIES = ["nn_mnist"]
AUTO_LOAD = ["sensor", "text_sensor"]

nn_spectrasense_ns = cg.esphome_ns.namespace("nn_spectrasense")
NnSpectrasense = nn_spectrasense_ns.class_("NnSpectrasense", cg.Component)

CONF_WEIGHTS_IN_RAM = "weights_in_ram"
CAPTEURS = {
    "inference_time": ("µs", 0, "mdi:timer-outline"),
    "max_error": (None, 9, "mdi:delta"),
}
TEXTES = {
    "model_id": "mdi:brain",
    "autotest": "mdi:clipboard-check-outline",
}

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(NnSpectrasense),
        cv.Optional(CONF_WEIGHTS_IN_RAM, default=True): cv.boolean,
        **{cv.Optional(k): sensor.sensor_schema(unit_of_measurement=u, accuracy_decimals=d, icon=i,
                                                state_class=STATE_CLASS_MEASUREMENT,
                                                entity_category=ENTITY_CATEGORY_DIAGNOSTIC)
           if u else sensor.sensor_schema(accuracy_decimals=d, icon=i, state_class=STATE_CLASS_MEASUREMENT,
                                          entity_category=ENTITY_CATEGORY_DIAGNOSTIC)
           for k, (u, d, i) in CAPTEURS.items()},
        **{cv.Optional(k): text_sensor.text_sensor_schema(icon=i, entity_category=ENTITY_CATEGORY_DIAGNOSTIC)
           for k, i in TEXTES.items()},
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    cg.add(var.set_weights_in_ram(config[CONF_WEIGHTS_IN_RAM]))
    for k in CAPTEURS:
        if k in config:
            cg.add(getattr(var, "set_%s_sensor" % k)(await sensor.new_sensor(config[k])))
    for k in TEXTES:
        if k in config:
            cg.add(getattr(var, "set_%s_text_sensor" % k)(await text_sensor.new_text_sensor(config[k])))

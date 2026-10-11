"""Composant ESPHome nn_mnist : liaison entre le moteur d'inférence et Home Assistant.

Toutes les entités sont facultatives. test_vectors (vrai par défaut) embarque
les images de test de test_vectors.h, nécessaires au bouton « image suivante »
et à l'autotest. activity_output (facultatif) : une sortie, par exemple la LED
de la carte, allumée pendant au moins activity_duration à chaque inférence.
"""
import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import output, sensor, text_sensor
from esphome.const import CONF_ID, ENTITY_CATEGORY_DIAGNOSTIC, STATE_CLASS_MEASUREMENT

AUTO_LOAD = ["output", "sensor", "text_sensor"]

nn_mnist_ns = cg.esphome_ns.namespace("nn_mnist")
NnMnist = nn_mnist_ns.class_("NnMnist", cg.Component)

CONF_TEST_VECTORS = "test_vectors"
CONF_ACTIVITY_OUTPUT = "activity_output"
CONF_ACTIVITY_DURATION = "activity_duration"
CAPTEURS = {
    # clé YAML : (unité, décimales, icône, catégorie)
    "prediction": (None, 0, "mdi:numeric", None),
    "confidence": ("%", 1, "mdi:percent", None),
    "margin": ("%", 1, "mdi:arrow-expand-vertical", None),
    "inference_time": ("µs", 0, "mdi:timer-outline", ENTITY_CATEGORY_DIAGNOSTIC),
    "expected_label": (None, 0, "mdi:tag-outline", None),
    "max_error": (None, 9, "mdi:delta", ENTITY_CATEGORY_DIAGNOSTIC),
}
TEXTES = {
    "model_id": ("mdi:brain", ENTITY_CATEGORY_DIAGNOSTIC),
    "autotest": ("mdi:clipboard-check-outline", ENTITY_CATEGORY_DIAGNOSTIC),
}


def _capteur(unite, decimales, icone, categorie):
    options = {"accuracy_decimals": decimales, "icon": icone,
               "state_class": STATE_CLASS_MEASUREMENT}
    if unite:
        options["unit_of_measurement"] = unite
    if categorie:
        options["entity_category"] = categorie
    return sensor.sensor_schema(**options)


def _texte(icone, categorie):
    return text_sensor.text_sensor_schema(icon=icone, entity_category=categorie)


CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(NnMnist),
        cv.Optional(CONF_TEST_VECTORS, default=True): cv.boolean,
        cv.Optional(CONF_ACTIVITY_OUTPUT): cv.use_id(output.BinaryOutput),
        cv.Optional(CONF_ACTIVITY_DURATION, default="150ms"): cv.positive_time_period_milliseconds,
        **{cv.Optional(cle): _capteur(*v) for cle, v in CAPTEURS.items()},
        **{cv.Optional(cle): _texte(*v) for cle, v in TEXTES.items()},
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    if config[CONF_TEST_VECTORS]:
        cg.add_define("USE_NN_MNIST_TEST_VECTORS")
    if CONF_ACTIVITY_OUTPUT in config:
        sortie = await cg.get_variable(config[CONF_ACTIVITY_OUTPUT])
        cg.add(var.set_activity_output(sortie))
        cg.add(var.set_activity_duration(config[CONF_ACTIVITY_DURATION].total_milliseconds))
    for cle in CAPTEURS:
        if cle in config:
            sens = await sensor.new_sensor(config[cle])
            cg.add(getattr(var, "set_%s_sensor" % cle)(sens))
    for cle in TEXTES:
        if cle in config:
            txt = await text_sensor.new_text_sensor(config[cle])
            cg.add(getattr(var, "set_%s_text_sensor" % cle)(txt))

"""Composant ESPHome nn_spectrasense (essais) : le réseau convolutif SpectraSense sur la carte.

- autotest (fenêtres de validation embarquées, comparées couche par couche aux références numpy)
  et profilage (cycles par couche) ;
- chaîne en direct si `power` est donné : lectures de puissance -> rafales re-datées -> grille à
  1 s -> porte -> réseau -> automate (chaine.h, identique à la chaîne Python, cf. parite_cpp.py),
  avec publication de l'état de la machine à café et des appareils simples (marche / arrêt).
Le moteur d'inférence est celui de nn_mnist (même code, une seule copie dans le firmware) :
nn_mnist doit donc aussi être chargé (components: [nn_mnist, nn_spectrasense]).
Headers : esp32/python/export_conv.py <id> --sortie <ce dossier> --espace nn_ss
          --inclure-moteur esphome/components/nn_mnist/nn_engine.h --par-classe 1
"""
import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import binary_sensor, sensor, text_sensor, time
from esphome.const import CONF_ID, ENTITY_CATEGORY_DIAGNOSTIC, STATE_CLASS_MEASUREMENT, STATE_CLASS_TOTAL_INCREASING

DEPENDENCIES = ["nn_mnist"]
AUTO_LOAD = ["sensor", "text_sensor", "binary_sensor"]

nn_spectrasense_ns = cg.esphome_ns.namespace("nn_spectrasense")
NnSpectrasense = nn_spectrasense_ns.class_("NnSpectrasense", cg.Component)

CONF_WEIGHTS_IN_RAM = "weights_in_ram"
CONF_POWER = "power"                  # capteur de puissance de la ligne (W)
CONF_CLOCK_DELAY = "clock_delay"      # retard de l'horloge qui ferme les cases dans les silences
CONF_THRESHOLD = "threshold"          # seuil de confiance pour publier (automate : seuil)
CONF_PEAK_THRESHOLD = "peak_threshold"
CONF_TIME_ID = "time_id"              # horloge réelle : état long de la machine gardé en flash
CONF_HOLD = "hold"                    # secondes consécutives au-dessus du seuil (automate : tenue)
CONF_OFF_HOLD = "off_hold"            # repos continu avant « arrêt » d'un appareil simple (automate : tenue_arret)
CAPTEURS = {
    "inference_time": ("µs", 0, "mdi:timer-outline"),
    "max_error": (None, 9, "mdi:delta"),
}
TEXTES = {
    "model_id": "mdi:brain",
    "autotest": "mdi:clipboard-check-outline",
    "network_class": "mdi:label-outline",       # classe dominante du réseau (diagnostic)
}
# probabilités brutes du réseau, dans l'ordre des sorties (fenetres.py : CLASSES_LIGNE)
CLASSES = ["repos", "allumage_chauffe", "allumage_regulation", "allumage_fin", "cafe_broyage",
           "cafe_extraction", "cafe_fin", "pic_maintien", "extinction", "grille_pain", "bouilloire",
           "consommateur_inconnu"]
# états publiés : machine (8 états) et appareils simples (marche / arrêt)
APPAREILS = {
    "toaster": "mdi:toaster",
    "kettle": "mdi:kettle",
    "unknown_load": "mdi:power-plug-outline",
}

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(NnSpectrasense),
        cv.Optional(CONF_WEIGHTS_IN_RAM, default=True): cv.boolean,
        cv.Optional(CONF_POWER): cv.use_id(sensor.Sensor),
        cv.Optional(CONF_TIME_ID): cv.use_id(time.RealTimeClock),
        cv.Optional(CONF_CLOCK_DELAY, default="2s"): cv.positive_time_period_milliseconds,
        cv.Optional(CONF_THRESHOLD, default=0.95): cv.float_range(min=0.5, max=1.0),
        cv.Optional(CONF_PEAK_THRESHOLD, default=0.5): cv.float_range(min=0.0, max=1.0),
        cv.Optional(CONF_HOLD, default=1): cv.int_range(min=1, max=30),
        cv.Optional(CONF_OFF_HOLD, default="8s"): cv.All(cv.positive_time_period_seconds,
                                                          cv.Range(min=cv.TimePeriod(seconds=1), max=cv.TimePeriod(seconds=120))),
        cv.Optional("machine"): text_sensor.text_sensor_schema(icon="mdi:coffee-maker-outline"),
        **{cv.Optional(k): binary_sensor.binary_sensor_schema(icon=i) for k, i in APPAREILS.items()},
        cv.Optional("gate"): binary_sensor.binary_sensor_schema(icon="mdi:gate-open",
                                                                entity_category=ENTITY_CATEGORY_DIAGNOSTIC),
        cv.Optional("coffee_count"): sensor.sensor_schema(accuracy_decimals=0, icon="mdi:coffee",
                                                          state_class=STATE_CLASS_TOTAL_INCREASING),
        cv.Optional("network_confidence"): sensor.sensor_schema(unit_of_measurement="%", accuracy_decimals=0,
                                                                icon="mdi:gauge", state_class=STATE_CLASS_MEASUREMENT,
                                                                entity_category=ENTITY_CATEGORY_DIAGNOSTIC),
        cv.Optional("class_probabilities"): cv.Schema(
            {cv.Optional(k): sensor.sensor_schema(unit_of_measurement="%", accuracy_decimals=0, icon="mdi:percent",
                                                  state_class=STATE_CLASS_MEASUREMENT,
                                                  entity_category=ENTITY_CATEGORY_DIAGNOSTIC) for k in CLASSES}),
        cv.Optional("invalid_seconds"): sensor.sensor_schema(unit_of_measurement="s", accuracy_decimals=0,
                                                             icon="mdi:timer-alert-outline",
                                                             state_class=STATE_CLASS_TOTAL_INCREASING,
                                                             entity_category=ENTITY_CATEGORY_DIAGNOSTIC),
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
    if CONF_POWER in config:
        cg.add(var.set_power_sensor(await cg.get_variable(config[CONF_POWER])))
    if CONF_TIME_ID in config:
        cg.add(var.set_time(await cg.get_variable(config[CONF_TIME_ID])))
    cg.add(var.set_clock_delay(config[CONF_CLOCK_DELAY].total_milliseconds / 1000.0))
    cg.add(var.set_reglages(config[CONF_THRESHOLD], config[CONF_PEAK_THRESHOLD], config[CONF_HOLD],
                            config[CONF_OFF_HOLD].total_seconds))
    if "machine" in config:
        cg.add(var.set_machine_text_sensor(await text_sensor.new_text_sensor(config["machine"])))
    for k in [*APPAREILS, "gate"]:
        if k in config:
            cg.add(getattr(var, "set_%s_binary_sensor" % k)(await binary_sensor.new_binary_sensor(config[k])))
    if "coffee_count" in config:
        cg.add(var.set_coffee_count_sensor(await sensor.new_sensor(config["coffee_count"])))
    if "network_confidence" in config:
        cg.add(var.set_network_confidence_sensor(await sensor.new_sensor(config["network_confidence"])))
    for k, nom in enumerate(CLASSES):
        if nom in config.get("class_probabilities", {}):
            cg.add(var.set_class_probability_sensor(k, await sensor.new_sensor(config["class_probabilities"][nom])))
    if "invalid_seconds" in config:
        cg.add(var.set_invalid_seconds_sensor(await sensor.new_sensor(config["invalid_seconds"])))
    for k in CAPTEURS:
        if k in config:
            cg.add(getattr(var, "set_%s_sensor" % k)(await sensor.new_sensor(config[k])))
    for k in TEXTES:
        if k in config:
            cg.add(getattr(var, "set_%s_text_sensor" % k)(await text_sensor.new_text_sensor(config[k])))

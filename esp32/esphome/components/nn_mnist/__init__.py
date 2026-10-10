import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.const import CONF_ID

nn_mnist_ns = cg.esphome_ns.namespace("nn_mnist")
NnMnist = nn_mnist_ns.class_("NnMnist", cg.Component)

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(NnMnist),
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

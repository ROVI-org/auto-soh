from pydantic import Field

from moirae.models.base import InputQuantities, OutputQuantities, ScalarParameter


class ECMInput(InputQuantities):
    """
    Control of a battery based on the feed current, temperature
    """
    temperature: ScalarParameter = Field(25., description='Temperature reading(s). Units: °C')


# TODO (vventuri): Remember we need to implement ways to denoise SOC, Qt, R0,
#                   which require more outputs
class ECMMeasurement(OutputQuantities):
    """
    Controls the outputs of the ECM.
    """
    pass

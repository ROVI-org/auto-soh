from pydantic import Field
from asoh.models.base import InputQuantities, OutputMeasurements


class ECMInput(InputQuantities):
    """
    Control of a battery based on the feed current, temperature
    """
    temperature: float = Field(default=None,
                               description='Temperature reading(s). Units: °C')


# TODO (vventuri): Remember we need to implement ways to denoise SOC, Qt, R0,
#                   which require more outputs
class ECMMeasurement(OutputMeasurements):
    """
    Controls the outputs of the ECM.
    """
    pass

from moirae.models.base import InputQuantities, OutputQuantities


class ECMInput(InputQuantities):
    """
    Control of a battery based on the feed current, temperature
    """
    pass


# TODO (vventuri): Remember we need to implement ways to denoise SOC, Qt, R0,
#                   which require more outputs
class ECMMeasurement(OutputQuantities):
    """
    Controls the outputs of the ECM.
    """
    pass

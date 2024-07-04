# General imports
from typing import Union, Optional, List
from pydantic import Field

# ASOH imports
from asoh.models.base import HiddenVector


################################################################################
#                               HIDDEN VECTOR                                  #
################################################################################
class ECMTransientVector(HiddenVector):
    soc: float = Field(default=0.0, description='State of charge (SOC)')
    q0: Optional[float] = \
        Field(default=[],
              description='Charge in the series capacitor. Units: Coulomb')
    i_rc: Optional[Union[float, List]] = \
        Field(default=[],
              description='Currents through RC components. Units: Amp')
    hyst: float = Field(default=0, description='Hysteresis voltage. Units: V')

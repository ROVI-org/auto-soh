# General imports
from pydantic import Field
from typing import Tuple, Optional

# ASOH imports
from asoh.models.base import HealthVariable

# Internal imports
from .components import (MaxTheoreticalCapacity,
                         CoulombicEfficiency,
                         Resistance,
                         Capacitance,
                         RCComponent,
                         OpenCircuitVoltage,
                         HysteresisParameters)


################################################################################
#                                    A-SOH                                     #
################################################################################
class ECMASOH(HealthVariable):
    Qt: MaxTheoreticalCapacity = \
        Field(description='Maximum theoretical discharge capacity (Qt).')
    CE: CoulombicEfficiency = \
        Field(default=CoulombicEfficiency(),
              description='Coulombic effiency (CE)')
    OCV: OpenCircuitVoltage = \
        Field(description='Open Circuit Voltage (OCV)')
    R0: Resistance = \
        Field(description='Series Resistance (R0)')
    C0: Optional[Capacitance] = \
        Field(default_factory=None,
              description='Series Capacitance (C0)',
              max_length=1)
    RCelements: Tuple[RCComponent, ...] = \
        Field(default=tuple,
              description='Tuple of RC components')
    H0: HysteresisParameters = \
        Field(default=HysteresisParameters(base_values=0.0, updatable=False),
              description='Hysteresis component')

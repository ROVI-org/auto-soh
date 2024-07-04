# General imports
from pydantic import Field
from typing import Tuple, Optional, Union, List
import numpy as np

# ASOH imports
from asoh.models.base import HealthVariable

# Internal imports
from .components import (MaxTheoreticalCapacity,
                         CoulombicEfficiency,
                         Resistance,
                         Capacitance,
                         RCComponent,
                         ReferenceOCV,
                         EntropicOCV,
                         OpenCircuitVoltage,
                         HysteresisParameters)
from .utils import realistic_fake_ocv


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


################################################################################
#                               PROVIDE TEMPLATE                               #
################################################################################
def provide_asoh_template(
        has_C0: bool,
        num_RC: float,
        Qt: float = 10.0,
        CE: float = 1.0,
        OCV: Union[float, np.ndarray, None] = None,
        R0: Union[float, np.ndarray] = 0.05,
        C0: Union[float, np.ndarray, None] = None,
        H0: Union[float, np.ndarray] = 0.05,
        RC: Union[List[Tuple[np.ndarray, ...]], None] = None,
        num_copies: int = 1
        ) -> Union[ECMASOH, List[ECMASOH]]:
    """
    Function to provide a basic ECM ASOH template. It cannot be used for full
    initialization, as it will "hardcode" a few parameters. It only provides a
    "default" ASOH based on the ECM description.
    """
    # Start preparing the requirements
    Qt = MaxTheoreticalCapacity(base_values=Qt)
    CE = CoulombicEfficiency(base_values=CE)
    # R0 prep
    R0 = Resistance(base_values=R0, temperature_dependence_factor=0.0025)
    # OCV prep
    OCVent = EntropicOCV(base_values=0.005)
    if OCV is None:
        socs = np.linspace(0, 1, 20)
        OCVref = ReferenceOCV(base_values=realistic_fake_ocv(socs))
    else:
        OCVref = ReferenceOCV(base_values=OCV)
    OCV = OpenCircuitVoltage(OCVref=OCVref, OCVentropic=OCVent)
    # H0 prep
    H0 = HysteresisParameters(base_values=H0, gamma=0.9)
    # Assemble minimal ASOH
    asoh = ECMASOH(Qt=Qt, CE=CE, OCV=OCV, R0=R0, H0=H0)
    # C0 prep
    if has_C0:
        if C0 is None:
            # Make it so that it's impact is at most 10 mV
            C0 = Qt.value / 0.01  # Recall it's stored in Amp-hour
        C0 = Capacitance(base_values=C0)
        asoh.C0 = C0
    # RC prep
    if num_RC:
        if RC is None:
            RC_R = Resistance(base_values=0.01,
                              temperature_dependence_factor=0.0025)
            RC_C = Capacitance(base_values=2500)
            RCcomps = tuple(RCComponent(R=RC_R, C=RC_C).model_copy()
                            for _ in range(num_RC))
        else:
            if len(RC) != num_RC:
                raise ValueError('Amount of RC information provided does not '
                                 'match number of RC elements specified!')
            RCcomps = tuple
            for RC_info in RC:
                R_info = RC_info[0]
                C_info = RC_info[1]
                RC_R = Resistance(base_values=R_info,
                                  temperature_dependence_factor=0.0025)
                RC_C = Capacitance(base_values=C_info)
                RCcomps += (RCComponent(R=RC_R, C=RC_C).model_copy(),)
        asoh.RCelements = RCcomps

    # Deal with copies
    if num_copies == 1:
        return asoh
    else:
        return [asoh.model_copy() for _ in range(num_copies)]

""" Testing components of ECM """
# General imports
import numpy as np
from pytest import fixture

# Test imports
from conftest import (coarse_soc,
                      fine_soc,
                      const_ocv,
                      linear_ocv,
                      realistic_ocv)
from asoh.models.ecm import ECMASOH
from asoh.models.ecm.components import (MaxTheoreticalCapacity,
                                        CoulombicEfficiency,
                                        Resistance,
                                        Capacitance,
                                        RCComponent,
                                        ReferenceOCV,
                                        EntropicOCV,
                                        OpenCircuitVoltage,
                                        HysteresisParameters)


@fixture
def basic_rint() -> ECMASOH:
    Qt = MaxTheoreticalCapacity(base_values=10)
    R0 = Resistance(base_values=0.05)
    OCVref = ReferenceOCV(base_values=const_ocv(coarse_soc))
    OCVent = EntropicOCV(base_values=0.)
    OCV = OpenCircuitVoltage(OCVref=OCVref, OCVentropic=OCVent)
    return ECMASOH(Qt=Qt, OCV=OCV, R0=R0)


def test_basic_rint(basic_rint):
    print(ECMASOH.get_parameters())

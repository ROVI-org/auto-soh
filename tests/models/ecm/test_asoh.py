""" Testing components of ECM """
# General imports
import numpy as np
from pytest import fixture

# Test imports
from conftest import (coarse_soc,
                      fine_soc,
                      const_ocv,
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


@fixture
def basic_thenevin() -> ECMASOH:
    Qt = MaxTheoreticalCapacity(base_values=10)
    R0 = Resistance(base_values=0.05)
    OCVref = ReferenceOCV(base_values=const_ocv(coarse_soc))
    OCVent = EntropicOCV(base_values=0.)
    OCV = OpenCircuitVoltage(OCVref=OCVref, OCVentropic=OCVent)
    RC = RCComponent(R=Resistance(base_values=0.01),
                     C=Capacitance(base_values=2500))
    return ECMASOH(Qt=Qt, OCV=OCV, R0=R0, RCelements=[RC])


@fixture
def basic_pngv() -> ECMASOH:
    Qt = MaxTheoreticalCapacity(base_values=10)
    CE = CoulombicEfficiency(base_values=0.999)
    R0 = Resistance(base_values=0.05)
    C0 = Capacitance(base_values=1.)
    OCVref = ReferenceOCV(base_values=const_ocv(coarse_soc))
    OCVent = EntropicOCV(base_values=0.)
    OCV = OpenCircuitVoltage(OCVref=OCVref, OCVentropic=OCVent)
    RC1 = RCComponent(R=Resistance(base_values=0.01),
                      C=Capacitance(base_values=2500))
    RC2 = RCComponent(R=Resistance(base_values=0.02),
                      C=Capacitance(base_values=1500))
    hyst = HysteresisParameters(base_values=0.01,
                                gamma=2.)
    return ECMASOH(Qt=Qt,
                   CE=CE,
                   OCV=OCV,
                   R0=R0,
                   C0=C0,
                   RCelements=[RC1, RC2],
                   H0=hyst)


@fixture
def full_pngv() -> ECMASOH:
    Qt = MaxTheoreticalCapacity(base_values=10)
    CE = CoulombicEfficiency(base_values=0.999)
    R0 = Resistance(base_values=np.array([0.04, 0.05, 0.06, 0.07]),
                    temperature_dependence_factor=0.1,
                    updatable=('base_values', 'temperature_dependence_factor'))
    C0 = Capacitance(base_values=np.array([1., 1.1]))
    OCVref = ReferenceOCV(base_values=realistic_ocv(fine_soc))
    OCVent = EntropicOCV(base_values=np.linspace(0.1, 0.15, 6))
    OCV = OpenCircuitVoltage(OCVref=OCVref, OCVentropic=OCVent)
    RC1 = RCComponent(R=Resistance(base_values=0.01),
                      C=Capacitance(base_values=2500))
    RC2 = RCComponent(R=Resistance(base_values=0.02),
                      C=Capacitance(base_values=1500))
    hyst = HysteresisParameters(base_values=0.01,
                                gamma=2.)
    return ECMASOH(Qt=Qt,
                   CE=CE,
                   OCV=OCV,
                   R0=R0,
                   C0=C0,
                   RCelements=[RC1, RC2],
                   H0=hyst)


def test_basic_rint(basic_rint):
    assert basic_rint.updatable == tuple
    assert np.allclose([10, 0.05, 0],  # [Qt, R0, OCVent]
                       basic_rint.get_parameters(),
                       atol=1e-12)

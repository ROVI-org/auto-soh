""" Testing components of ECM """
import numpy as np
from pytest import fixture
import pytest

from asoh.models.ecm import ECMASOH
from asoh.models.ecm.components import (MaxTheoreticalCapacity,
                                        Resistance,
                                        Capacitance,
                                        RCComponent,
                                        ReferenceOCV,
                                        EntropicOCV,
                                        OpenCircuitVoltage,
                                        HysteresisParameters)
from asoh.models.ecm.utils import realistic_fake_ocv


@fixture
def basic_rint(const_ocv, coarse_soc) -> ECMASOH:
    q_t = MaxTheoreticalCapacity(base_values=10)
    r0 = Resistance(base_values=0.05)
    ocvref = ReferenceOCV(base_values=const_ocv(coarse_soc))
    ocvent = EntropicOCV(base_values=0.)
    ocv = OpenCircuitVoltage(ocv_ref=ocvref, ocv_ent=ocvent)
    return ECMASOH(q_t=q_t, ocv=ocv, r0=r0)


@fixture
def basic_thenevin(const_ocv, coarse_soc) -> ECMASOH:
    q_t = MaxTheoreticalCapacity(base_values=10)
    r0 = Resistance(base_values=0.05)
    ocvref = ReferenceOCV(base_values=const_ocv(coarse_soc))
    ocvent = EntropicOCV(base_values=0.)
    ocv = OpenCircuitVoltage(ocv_ref=ocvref, ocv_ent=ocvent)
    RC = RCComponent(R=Resistance(base_values=0.01),
                     C=Capacitance(base_values=2500))
    return ECMASOH(
        q_t=q_t,
        ocv=ocv,
        r0=r0,
        rc_elements=[RC]
    )


@fixture
def basic_pngv(const_ocv, coarse_soc) -> ECMASOH:
    q_t = MaxTheoreticalCapacity(base_values=10)
    r0 = Resistance(base_values=0.05)
    c0 = Capacitance(base_values=1.)
    ocvref = ReferenceOCV(base_values=const_ocv(coarse_soc))
    ocvent = EntropicOCV(base_values=0.)
    ocv = OpenCircuitVoltage(ocv_ref=ocvref, ocv_ent=ocvent)
    RC1 = RCComponent(R=Resistance(base_values=0.01),
                      C=Capacitance(base_values=2500))
    RC2 = RCComponent(R=Resistance(base_values=0.02),
                      C=Capacitance(base_values=1500))
    hyst = HysteresisParameters(base_values=0.01,
                                gamma=2.)
    return ECMASOH(q_t=q_t,
                   ce=0.999,
                   ocv=ocv,
                   r0=r0,
                   c0=c0,
                   rc_elements=[RC1, RC2],
                   h0=hyst)


@fixture
def full_pngv(fine_soc) -> ECMASOH:
    q_t = MaxTheoreticalCapacity(base_values=10)
    r0 = Resistance(base_values=np.array([0.04, 0.05, 0.06, 0.07]),
                    temperature_dependence_factor=0.1,
                    updatable=('base_values', 'temperature_dependence_factor'))
    c0 = Capacitance(base_values=np.array([1., 1.1]))
    ocvref = ReferenceOCV(base_values=realistic_fake_ocv(fine_soc))
    ocvent = EntropicOCV(base_values=np.linspace(0.1, 0.15, 6))
    ocv = OpenCircuitVoltage(ocv_ref=ocvref, ocv_ent=ocvent)
    RC1 = RCComponent(R=Resistance(base_values=0.01),
                      C=Capacitance(base_values=2500))
    RC2 = RCComponent(R=Resistance(base_values=0.02),
                      C=Capacitance(base_values=1500))
    hyst = HysteresisParameters(base_values=0.01,
                                gamma=2.)
    return ECMASOH(q_t=q_t,
                   ce=0.999,
                   ocv=ocv,
                   r0=r0,
                   c0=c0,
                   rc_elements=[RC1, RC2],
                   h0=hyst)


@fixture
def full_asoh() -> ECMASOH:
    r_rc1 = np.array([1, 2])
    r_rc2 = np.array([3, 4])
    c_rc1 = np.array([500, 600])
    c_rc2 = np.array([600, 700])
    ocv = realistic_fake_ocv(np.linspace(0, 1, 10))
    asoh = ECMASOH.provide_template(has_C0=True, num_RC=2, OCV=ocv, RC=[(r_rc1, c_rc1), (r_rc2, c_rc2)])
    return asoh


@pytest.mark.xfail()
def test_basic_rint(basic_rint):
    assert basic_rint.updatable == set()
    basic_rint.mark_all_updatable()
    assert np.allclose([10, 0.05, 0],  # [q_t, r0, ocvent]
                       basic_rint.get_parameters(),
                       atol=1e-12)


def test_full_asoh_template(full_asoh):
    assert np.allclose(full_asoh.rc_elements[0].r.get_value(soc=0.5), 1.5), 'Wrong R_RC_1 value!'
    assert np.allclose(full_asoh.rc_elements[1].r.get_value(soc=0.25), 3.25), 'Wrong R_RC_2 value!'
    assert np.allclose(full_asoh.rc_elements[0].c.get_value(soc=0.75), 575), 'Wrong C_RC_1 value!'
    assert np.allclose(full_asoh.rc_elements[1].c.get_value(soc=0.37), 637), 'Wrong R_RC_2 value!'

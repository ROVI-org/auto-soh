""" Testing components of ECM """
import numpy as np
from pytest import fixture

from moirae.models.ecm import ECMASOH
from moirae.models.ecm.components import (MaxTheoreticalCapacity,
                                          Resistance,
                                          ReferenceOCV,
                                          EntropicOCV,
                                          OpenCircuitVoltage)
from moirae.models.ecm.utils import realistic_fake_ocv


@fixture
def full_asoh() -> ECMASOH:
    r_rc1 = np.array([1, 2])
    r_rc2 = np.array([3, 4])
    c_rc1 = np.array([500, 600])
    c_rc2 = np.array([600, 700])
    ocv = realistic_fake_ocv(np.linspace(0, 1, 10))
    asoh = ECMASOH.provide_template(has_C0=True, num_RC=2, OCV=ocv, RC=[(r_rc1, c_rc1), (r_rc2, c_rc2)])
    return asoh


def test_full_asoh_template(full_asoh):
    assert np.allclose(full_asoh.rc_elements[0].r.get_value(soc=0.5), 1.5), 'Wrong R_RC_1 value!'
    assert np.allclose(full_asoh.rc_elements[1].r.get_value(soc=0.25), 3.25), 'Wrong R_RC_2 value!'
    assert np.allclose(full_asoh.rc_elements[0].c.get_value(soc=0.75), 575), 'Wrong C_RC_1 value!'
    assert np.allclose(full_asoh.rc_elements[1].c.get_value(soc=0.37), 637), 'Wrong R_RC_2 value!'


def test_json(full_asoh):
    as_json = full_asoh.model_dump_json()
    from_json = ECMASOH.model_validate_json(as_json)
    for (name_a, value_a), (name_b, value_b) in zip(
            from_json.iter_parameters(updatable_only=False),
            full_asoh.iter_parameters(updatable_only=False)
    ):
        assert name_a == name_b
        assert np.allclose(value_a, value_b)


def test_energy():
    # Prepare fields for A-SOH
    qt = MaxTheoreticalCapacity(base_values=10.)
    r0 = Resistance(base_values=1.)
    ocv_ref = ReferenceOCV(base_values=np.ones(11))
    ocv_ent = EntropicOCV(base_values=np.zeros(11))
    ocv = OpenCircuitVoltage(ocv_ref=ocv_ref, ocv_ent=ocv_ent)
    asoh = ECMASOH(q_t=qt, ce=1, ocv=ocv, r0=r0)

    # With this simple example, we expect the energy to simply be the total discharge capacity multiplied by 1 V
    expected = asoh.q_t.amp_hour
    computed = asoh.get_theoretical_energy()
    assert np.allclose(expected, computed), \
        f'Mismatch in constant OCV ECMASOH energy calculation; expected {expected} but got {computed} Wh!'

    # Now, let's alter the OCV such that it is equal to the SOC, and, thus, forms a triangle
    asoh.mark_updatable(name='ocv.ocv_ref.base_values')
    asoh.update_parameters(values=np.linspace(0., 1., 11)[None, :])

    # In this case, we expect the energy to be Qt/2 (in Wh)
    expected = asoh.q_t.amp_hour / 2.
    computed = asoh.get_theoretical_energy()
    assert np.allclose(expected, computed), \
        f'Mismatch in linear OCV ECMASOH energy calculation; expected {expected} but got {computed} Wh!'

    # Now, let's check the case in which we want to evaluate the temperature effects.
    asoh.mark_updatable(name='ocv.ocv_ent.base_values')
    asoh.update_parameters(names=('ocv.ocv_ent.base_values',), values=np.ones(11)[None, :])
    # For a temperature 1 degree above the reference temperature, this should add an energy of Qt Wh
    temp = asoh.ocv.ocv_ref.reference_temperature + 1
    expected += asoh.q_t.amp_hour
    computed = asoh.get_theoretical_energy(temperature=temp)
    assert np.allclose(expected, computed), \
        f'Mismatch in temperature dependent ECMASOH energy calculation; expected {expected} but got {computed} Wh!'
    # Let's see if it still works if the entropic OCV is the same as the reference
    asoh.update_parameters(names=('ocv.ocv_ent.base_values',), values=np.linspace(0., 1., 11))
    expected = asoh.q_t.amp_hour
    computed = asoh.get_theoretical_energy(temperature=temp)
    assert np.allclose(expected, computed), \
        f'Mismatch in temp-dependent linear OCV ECMASOH energy calculation; expected {expected} but got {computed} Wh!'

    # Finally, let's check if this works with batching!
    asoh.mark_updatable(name='q_t.base_values')
    qt_vals = np.array([[10], [100]])
    ocv_ref_vals = np.array([np.ones(11), np.linspace(0., 1., 11)])
    ocv_ent_vals = np.array([np.linspace(0., 1., 11), np.linspace(0., 1., 11)])
    asoh.update_parameters(names=('q_t.base_values', 'ocv.ocv_ref.base_values', 'ocv.ocv_ent.base_values'),
                           values=np.hstack((qt_vals, ocv_ref_vals, ocv_ent_vals)))
    test_asoh = ECMASOH.provide_template(has_C0=False, num_RC=0, R0=[[6, 5, 4], [1, 2, 3]])
    assert False, f'{test_asoh.r0.get_value(soc=[0., 0.25])}'
    assert False, f'{asoh.ocv(soc=[0., 1.]).shape}'
    # First, without temperature, in which case the first value should just be 10 Wh, and the second one, 50 Wh
    expected = asoh.q_t.amp_hour
    expected[-1, :] /= 2.
    computed = asoh.get_theoretical_energy()
    assert np.allclose(expected, computed), \
        f'Wrong batched energy calculation, expected {expected} but got {computed} Wh!'

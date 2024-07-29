""" Testing components of ECM """
import numpy as np
from pytest import fixture

from moirae.models.ecm import ECMASOH
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

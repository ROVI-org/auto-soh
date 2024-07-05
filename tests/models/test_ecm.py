"""Tests for simple ECM and the base class"""

from pytest import fixture

from asoh.models.ecm import ECMASOH

from asoh.models.ecm.components import ConstantResistor, ConstantCapacitor, RCElement, OpenCircuitVoltage


@fixture()
def single_rc() -> ECMASOH:
    return ECMASOH(
        r0=ConstantResistor(base_value=0.01),
        rc_elements=[
            RCElement(
                r=ConstantResistor(base_value=0.001),
                c=ConstantCapacitor(base_value=2000.)
            ),
        ],
        ocv=OpenCircuitVoltage()
    )


def test_update_fields(single_rc):
    single_rc.mark_all_updatable()
    assert [k for k, _ in single_rc.iter_parameters()] == [
        'r0.base_value', 'rc_elements.0.r.base_value', 'rc_elements.0.c.base_value', 'ocv.slope', 'ocv.intercept'
    ]

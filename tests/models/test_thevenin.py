"""Test the interface to the Thevenin package"""

import numpy as np

from moirae.models.thevenin import TheveninInput, TheveninTransient, TheveninModel
from moirae.models.thevenin.components import SOCPolynomialVariable, SOCTempPolynomialVariable
from moirae.models.thevenin.state import TheveninASOH

rint = TheveninASOH(
    capacity=1.,
    ocv=SOCPolynomialVariable(coeffs=[1.5, 1.]),
    r=[SOCTempPolynomialVariable(soc_coeffs=[0.01, 0.01], t_coeffs=[0, 0.001])]
)


def test_rint():
    """Ensure the rint model's subcomponents work"""

    # Ensuring we can do SOC dependence
    assert rint.ocv(0.5) == 2.

    # Ensuring we can do SOC and temperature dependence
    assert rint.r[0](0.5, 298) == 0.015
    assert rint.r[0](0.5, 308) == 0.025

    # Test a single step at constant current
    state = TheveninTransient(soc=0., temp=298.)
    pre_inputs = TheveninInput(current=1., time=0., t_inf=298.)
    new_inputs = TheveninInput(current=1., time=30., t_inf=298.)

    model = TheveninModel()
    new_state = model.update_transient_state(pre_inputs, new_inputs, state, rint)
    assert np.allclose(new_state.soc.item(), 30 / 3600.)
    assert new_state.temp > state.temp  # Solving for the actual answer is annoying

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
    assert rint.ocv(0.5, batch_id=0) == 2.

    # Ensuring we can do SOC and temperature dependence
    assert rint.r[0](0.5, 298, batch_id=0) == 0.015
    assert rint.r[0](0.5, 308, batch_id=0) == 0.025

    # Test a single step at constant current
    state = TheveninTransient(soc=0., temp=298.)
    pre_inputs = TheveninInput(current=1., time=0., t_inf=298.)
    new_inputs = TheveninInput(current=1., time=30., t_inf=298.)

    model = TheveninModel()
    new_state = model.update_transient_state(pre_inputs, new_inputs, state, rint)
    assert new_state.batch_size == 1
    assert np.allclose(new_state.soc.item(), 30 / 3600.)
    assert new_state.temp > state.temp  # Solving for the actual answer is annoying

    # Get the terminal voltage
    voltage = model.calculate_terminal_voltage(new_inputs, new_state, rint)
    assert voltage.batch_size == 1
    assert np.allclose(voltage.terminal_voltage, (1.5 + 30 / 3600) + rint.r[0](30 / 3600, 298))


def test_soc_dependence():
    comp = SOCPolynomialVariable(coeffs=[[1, 0.5], [0.5, 0.25]])

    # Make sure it works with scalars for a single batch, as needed by Thevenin to update
    y = comp(0.5, batch_id=0)
    assert y.shape == ()
    assert np.isclose(y, 1.25)

    y = comp(0.5, batch_id=1)
    assert y.shape == ()
    assert np.isclose(y, 0.625)

    # Make sure it works with 2D inputs and all batches, as when computing terminal voltage
    y = comp(np.array([0.5]))
    assert y.shape == (2,)
    assert np.allclose(y, [1.25, 0.625])

    y = comp(np.array([0.5, 0.]))
    assert y.shape == (2,)
    assert np.allclose(y, [1.25, 0.5])


def test_soc_temp_dependence():
    comp = SOCTempPolynomialVariable(soc_coeffs=[[1, 0.5], [0.5, 0.25]], t_coeffs=[[0, -0.1]])

    # Make sure it works with scalars for a single batch, as needed by Thevenin to update
    y = comp(0.5, 299, batch_id=0)
    assert y.shape == ()
    assert np.isclose(y, 1.15)

    y = comp(0.5, 299, batch_id=1)
    assert y.shape == ()
    assert np.isclose(y, 0.525)

    # Make sure it works with 2D inputs and all batches, as when computing terminal voltage
    y = comp(np.array([0.5]), np.array([299]))
    assert y.shape == (2,)
    assert np.allclose(y, [1.15, 0.525])

    y = comp(np.array([0.5, 0.]), np.array([299]))
    assert y.shape == (2,)
    assert np.allclose(y, [1.15, 0.4])

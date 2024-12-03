"""Test the classes used in the documentation"""
from pydantic import Field
from typing import Union

import numpy as np
from numpy.polynomial.polynomial import polyval

from moirae.models.base import (
    GeneralContainer, ListParameter, ScalarParameter, HealthVariable, CellModel, InputQuantities, OutputQuantities
)


class ExampleContainer(GeneralContainer):
    """A container that holds each type of variable"""

    a: ScalarParameter = 1.
    """A variable which is always one value"""
    b: ListParameter = (2., 3.)
    """A variable which is a vector of any length"""


def test_example_container():
    ex = ExampleContainer()
    assert ex.a.shape == (1, 1)
    assert np.allclose(ex.a, [[1.]])

    assert ex.b.shape == (1, 2)
    assert np.allclose(ex.b, [[2., 3.]])


class OpenCircuitVoltage(HealthVariable):
    coeffs: ListParameter = [1, 0.5]
    """Parameters of a power-series polynomial"""

    def get_ocv(self, soc: Union[float, np.ndarray]) -> np.ndarray:
        """Compute the OCV as a function of SOC"""
        return polyval(soc, self.coeffs.T, tensor=False)


class TransientState(GeneralContainer):
    """A container that holds each type of variable"""

    soc: ScalarParameter = 0.
    """How much the battery has been charged. 1 is fully charged, 0 is fully discharged"""


class BatteryHealth(HealthVariable):
    q: ScalarParameter = 1
    """Battery capacity. Units: A-hr"""
    ocv: OpenCircuitVoltage = Field(default_factory=OpenCircuitVoltage)
    r: ScalarParameter = 0.01


class RintModel(CellModel):

    def update_transient_state(
            self,
            previous_inputs: InputQuantities,
            new_inputs: InputQuantities,
            transient_state: TransientState,
            asoh: BatteryHealth
    ) -> TransientState:
        new_output = transient_state.model_copy(deep=True)
        dt = new_inputs.time - previous_inputs.time
        new_output.soc = transient_state.soc + new_inputs.current * dt / 3600.
        return new_output

    def calculate_terminal_voltage(
            self,
            new_inputs: InputQuantities,
            transient_state: TransientState,
            asoh: BatteryHealth) -> OutputQuantities:
        v = new_inputs.current * asoh.r + asoh.ocv.get_ocv(transient_state.soc)
        return OutputQuantities(terminal_voltage=v)


def test_ocv():
    asoh = BatteryHealth()
    assert asoh.ocv.get_ocv(1.) == 1.5


def test_step():
    asoh = BatteryHealth()
    model = RintModel()
    state = TransientState()

    start_inputs = InputQuantities(current=1., time=0.)
    next_inputs = InputQuantities(current=1., time=10.)
    new_state = model.update_transient_state(start_inputs, next_inputs, state, asoh)
    assert np.isclose(new_state.soc, [10. / 3600])

    outputs = model.calculate_terminal_voltage(next_inputs, new_state, asoh)
    assert np.isclose(outputs.terminal_voltage, [1. + 0.5 * 10 / 3600 + 1 * 0.01])

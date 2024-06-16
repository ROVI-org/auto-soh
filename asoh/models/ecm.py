import numpy as np
from pydantic import Field

from .base import HealthModel, ControlState, InstanceState, Outputs


class ECMControl(ControlState):
    """Control of a battery based on the feed current"""

    pass


class ECMState(InstanceState):
    """State of a battery defined by an Equivalent circuit model"""

    charge: float = Field(0, description='State of charge of the battery element. Units: A-hr')
    r_serial: float = Field(description='Resistance of resistor in series with the battery element', gt=0)
    ocv_params: tuple[float, float] = Field(description='Parameters which define the open-circuit voltage of the battery element. '
                                                        'Constant component (units: V), component which varies linearly with charge (units: V/A-hr)')

    state_params: tuple[str, ...] = ('charge',)

    def compute_ocv(self) -> float:
        """Compute the open circuit voltage (OCV) given at the current state of charge

        Returns:
            OCV in Volts
        """
        return self.ocv_params[0] + self.charge * self.ocv_params[1]


class ECMOutputs(Outputs):
    """The only observable from an ECM model is the terminal voltage"""

    terminal_voltage: float = Field(description='Voltage at the terminal')


class SingleResistorModel(HealthModel):
    """A battery system modeled by a single resistor and open-circuit voltage which depends only on state of charge."""

    def dx(self, state: ECMState, control: ECMControl) -> np.ndarray:
        # The only change in the system is the state of charge increasing by the current
        return np.array([control.current / 3600.])

    def output(self, state: ECMState, control: ControlState) -> ECMOutputs:
        return ECMOutputs(
            terminal_voltage=state.compute_ocv() + state.r_serial * control.current
        )

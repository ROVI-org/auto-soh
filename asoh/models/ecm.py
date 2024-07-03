from pydantic import Field

from .base import InputState, HealthVariable, Measurements


class ECMInput(InputState):
    """Control of a battery based on the feed current"""

    pass


class Resistor(HealthVariable):
    """Represents a resistor that is affected by temperature and state-of-charge"""

    def get_resistance(self, soc: float, temp: float) -> float:
        """Get the effective resistance of this resistor given the battery state"""
        raise NotImplementedError()


class ConstantResistor(Resistor):
    """Resistor that always yields the same value"""

    r: float = 1.

    def get_resistance(self, soc: float, temp: float) -> float:
        return self.r


class OpenCircuitVoltage(HealthVariable):
    """Represents the open-circuit voltage of a battery that is dependent on SOC"""

    slope: float = 0.1
    intercept: float = 0.1

    def compute_ocv(self, soc: float) -> float:
        """Compute the open circuit voltage (OCV) given at the current state of charge

        Args:
            soc: State of charge for the battery
        Returns:
            OCV in Volts
        """
        return self.intercept + soc * self.slope


class ECMHealthState(HealthVariable):
    """State of a health for battery defined by an equivalent circuit model"""

    r_serial: ConstantResistor = Field(description='Resistance of resistor in series with the battery element', gt=0)
    ocv: OpenCircuitVoltage = Field(description='Model for the open circuit voltage')


class ECMMeasurements(Measurements):
    """The only observable from an ECM model is the terminal voltage"""

    terminal_voltage: float = Field(description='Voltage at the terminal')

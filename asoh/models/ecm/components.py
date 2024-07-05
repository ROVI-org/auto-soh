"""Models for the components of circuits"""

from asoh.models.base import HealthVariable


class ConstantElement:
    """Base class for elements which are constant regardless of SOC or temperature"""

    base_value: float

    def _compute(self, soc, temp):
        return self.base_value


class Resistor(HealthVariable):
    """Represents a resistor that is affected by temperature and state-of-charge"""

    def get_resistance(self, soc: float, temp: float) -> float:
        """Get the effective resistance of this resistor given the battery state

        Args:
            soc: State of charge of the battery, unitless
            temp: Temperature of the battery. Units: C
        Returns:
             The resistance of this element. Units: Ohm
        """
        raise NotImplementedError()


class ConstantResistor(Resistor, ConstantElement):
    """Resistor that always yields the same value"""

    def get_resistance(self, soc: float, temp: float) -> float:
        return self._compute(soc, temp)


class Capacitor(HealthVariable):
    """Base model for a capacitor"""

    def get_capacitance(self, soc: float, temp: float) -> float:
        """Get the expected capacitance for this element

        Args:
            soc: State of charge of the battery, unitless
            temp: Temperature of the battery. Units: C
        Returns:
             The resistance of this element. Units: Farad
        """
        raise NotImplementedError()


class ConstantCapacitor(Capacitor, ConstantElement):
    """Capacitor that always yields the same value"""

    def get_capacitance(self, soc: float, temp: float) -> float:
        return self._compute(soc, temp)


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


class RCElement(HealthVariable):
    """A single RC element within a circuit"""

    r: Resistor
    c: Capacitor

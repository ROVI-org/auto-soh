"""Models for the components of circuits"""
from typing import List, Optional, Union

from pydantic import Field, validate_call, ConfigDict
import numpy as np

from asoh.models.base import HealthVariable
from .utils import SOCInterpolatedHealth


class MaxTheoreticalCapacity(HealthVariable):
    """Defines maximum theoretical discharge capacity of a cell"""
    base_values: float = \
        Field(description='Maximum theoretical discharge capacity of a cell. Units: Amp-hour')
    updatable: set[str] = Field(default_factory=lambda: {'base_values'})

    @property
    def value(self) -> float:
        """
        Returns capacity in Amp-second
        """
        return 3600 * self.base_values

    @property
    def amp_hour(self) -> float:
        """
        Returns capacity in Amp-hour, as it was initialized.
        """
        return self.base_values


class Resistance(SOCInterpolatedHealth):
    """
    Defines the series resistance component of an ECM.
    """
    base_values: Union[float, np.ndarray] = \
        Field(
            description='Values of series resistance at specified SOCs. Units: Ohm')
    reference_temperature: Optional[float] = \
        Field(default=25,
              description='Reference temperature for internal parameters. Units: °C')
    temperature_dependence_factor: Optional[float] = \
        Field(default=0,
              description='Factor determining dependence of R0 with temperature. Units: 1/°C')

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_value(self,
                  soc: Union[float, List, np.ndarray],
                  temp: Union[float, List, np.ndarray, None] = None
                  ) -> Union[float, np.ndarray]:
        """
        Computes value of series resistance at a given SOC and temperature.
        """
        if isinstance(self.base_values, float):
            reference_value = self.base_values
        else:
            reference_value = self._interp_func(soc)
        if temp is None or self.temperature_dependence_factor == 0:
            return reference_value
        gamma = self.temperature_dependence_factor
        deltaT = np.array(temp) - self.reference_temperature
        new_value = reference_value * np.exp(- gamma * deltaT)
        return new_value


class Capacitance(SOCInterpolatedHealth):
    """
    Defines the series capacitance component of the ECM
    """
    base_values: Union[float, np.ndarray] = \
        Field(
            description='Values of series capacitance at specified SOCs. Units: F')


class RCComponent(HealthVariable):
    """
    Defines a RC component of the ECM
    """
    r: Resistance = Field(description='Resistive element of RC component')
    c: Capacitance = Field(description='Capacitive element of RC component')
    updatable: set[str] = \
        Field(default_factory=lambda: {'r', 'c'},
              description='Define updatable parameters (if any)')

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_value(self,
                  soc: Union[float, List, np.ndarray],
                  temp: Union[float, List, np.ndarray, None] = None
                  ) -> List[Union[float, np.ndarray]]:
        """
        Returns values of resistance and capacitance at given SOC and temperature.
        """
        r_val = self.r.get_value(soc=soc, temp=temp)
        c_val = self.c.get_value(soc=soc)
        return [r_val, c_val]

    def time_constant(self,
                      soc: Union[float, List, np.ndarray],
                      temp: Union[float, List, np.ndarray, None] = None
                      ) -> Union[float, np.ndarray]:
        r, c = self.get_value(soc=soc, temp=temp)
        return r * c


class ReferenceOCV(SOCInterpolatedHealth):
    base_values: Union[float, np.ndarray] = \
        Field(
            description='Values of reference OCV at specified SOCs. Units: V')
    reference_temperature: float = \
        Field(default=25,
              description='Reference temperature for OCV0. Units: °C')


class EntropicOCV(SOCInterpolatedHealth):
    base_values: Union[float, np.ndarray] = \
        Field(
            default=0,
            description='Values of entropic OCV term at specified SOCs. Units: V/°C')


class OpenCircuitVoltage(HealthVariable):
    ocv_ref: ReferenceOCV = \
        Field(description='Reference OCV at specified temperature')
    ocv_ent: EntropicOCV = \
        Field(description='Entropic OCV to determine temperature dependence')

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_value(self,
                  soc: Union[float, List, np.ndarray],
                  temp: Union[float, List, np.ndarray, None] = None
                  ) -> Union[float, np.ndarray]:
        """
        Returns values of OCV at given SOC(s) and temperature(s).
        """
        ocv = self.ocv_ref.get_value(soc=soc)
        if temp is not None:
            T_ref = self.ocv_ref.reference_temperature
            delta_T = temp - T_ref
            ocv += delta_T * self.ocv_ent.get_value(soc=soc)
        return ocv

    def __call__(self,
                 soc: Union[float, List, np.ndarray],
                 temp: Union[float, List, np.ndarray, None] = None
                 ) -> Union[float, np.ndarray]:
        """
        Allows this to be called and used as a function
        """
        return self.get_value(soc=soc, temp=temp)


class HysteresisParameters(SOCInterpolatedHealth):
    base_values: Union[float, np.ndarray] = \
        Field(
            description='Values of maximum hysteresis at specified SOCs. Units: V')
    gamma: float = Field(default=0.,
                         description='Exponential approach rate. Units: 1/V',
                         ge=0.)
    updatable: set[str] = \
        Field(default_factory=lambda: {'base_values', 'gamma'},
              description='Define updatable parameters (if any)')

from typing import List, Optional, Union, Literal, Callable
import numpy as np
from pydantic import Field, computed_field, validate_call, ConfigDict
from scipy.interpolate import interp1d

from .base import (InputQuantities,
                   OutputMeasurements,
                   HealthVariable,
                   HealthVariableCollection,
                   AdvancedStateOfHealth,
                   HealthModel)


################################################################################
#                               INPUTS & OUTPUT                                #
################################################################################
class ECMInput(InputQuantities):
    """
    Control of a battery based on the feed current, temperature
    """
    pass


# TODO (vventuri): Remember we need to implement ways to denoise SOC, Qt, R0,
#                   which require more outputs
class ECMMeasurement(OutputMeasurements):
    """
    Controls the outputs of the ECM.
    """
    pass


################################################################################
#                                HEALTH METRICS                                #
################################################################################
class MaxTheoreticalCapacity(HealthVariable):
    """
    Defines maximum theoretical discharge capacity of a cell
    """
    base_values: float = \
        Field(
            description='Maximum theoretical discharge capacity of a cell. Units: Amp-hour')
    name: Literal['Qt'] = Field('Qt', description='Name', allow_mutation=False)

    @property
    def value(self) -> float:
        return self.base_values

    # We need to redefine this function to provide capacity in Amp-seconds
    def get_updatable_parameter_values(self) -> List:
        """
        Returns maximum theoretical discharge capacity in Amp-seconds.
        """
        if self.updatable:
            return [3600 * self.base_values]
        return []


class CoulombicEfficiency(HealthVariable):
    """
    Holds Coulombic efficiency of the cell
    """
    base_values: float = Field(default=1.0, description="Coulombic efficiency")
    name: Literal['CE'] = Field('CE', description='Name', allow_mutation=False)

    @property
    def value(self) -> float:
        return self.base_values


class InterpolatedHealth(HealthVariable):
    """
    Defines basic functionality for HealthVariables that need interpolation
    between SOC pinpoints
    """
    base_values: Union[float, List] = \
        Field(default=0,
              description='Values at specified SOCs')
    soc_pinpoints: Optional[List] = \
        Field(default=[], description='SOC pinpoints for interpolation.')
    interpolation_style: \
        Literal['linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                'quadratic', 'cubic', 'previous', 'next'] = \
        Field(default='linear', description='Type of interpolation to perform')

    @computed_field
    @property
    def _interp_func(self) -> Callable:
        """
        Interpolate values. If soc_pinpoints have not been set, assume
        internal_parameters are evenly spread on an SOC interval [0,1].
        """
        if not len(self.soc_pinpoints):
            self.soc_pinpoints = np.linspace(0, 1, len(self.base_values))
        func = interp1d(self.soc_pinpoints,
                        self.base_values,
                        kind=self.interpolation_style,
                        bounds_error=False,
                        fill_value='extrapolate')
        return func

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def value(self,
              soc: Union[float, List, np.ndarray]
              ) -> Union[float, np.ndarray]:
        """
        Computes value(s) at given SOC(s)
        """
        if isinstance(self.base_values, float):
            return self.base_values
        return self._interp_func(soc)


class Resistance(InterpolatedHealth):
    """
    Defines the series resistance component of an ECM.
    """
    base_values: Union[float, List] = \
        Field(
            description='Values of series resistance at specified SOCs. Units: Ohm')
    reference_temperature: Optional[float] = \
        Field(default=25,
              description='Reference temperature for internal parameters. Units: °C')
    temperature_dependence_factor: Optional[float] = \
        Field(default=0,
              description='Factor determining dependence of R0 with temperature. Units: 1/°C')

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def value(self,
              soc: Union[float, List, np.ndarray],
              temp: Union[float, List, np.ndarray, None] = None
              ) -> Union[float, np.ndarray]:
        """
        Computes value of series resistance at a given SOC and temperature.
        """
        if isinstance(self.base_values, float):
            return self.base_values
        reference_value = self._interp_func(soc)
        if temp is None or self.temperature_dependence_factor == 0:
            return reference_value
        gamma = self.temperature_dependence_factor
        deltaT = np.array(temp) - self.reference_temperature
        new_value = reference_value * np.exp(- gamma * deltaT)
        return new_value


class Capacitance(InterpolatedHealth):
    """
    Defines the series capacitance component of the ECM
    """
    base_values: Union[float, List] = \
        Field(
            description='Values of series capacitance at specified SOCs. Units: F')


class SeriesResistance(Resistance):
    name: Literal['R0'] = Field(default='R0',
                                description='Name',
                                allow_mutation=False)


class SeriesCapacitance(Capacitance):
    name: Literal['C0'] = Field(default='C0',
                                description='Name',
                                allow_mutation=False)


class RCComponent(HealthVariableCollection):
    """
    Defines a RC component of the ECM
    """
    R: Resistance = Field(description='Resistive element of RC component')
    C: Capacitance = Field(description='Capacitive element of RC component')
    updatable: Union[Literal[False], tuple[str, ...]] = \
        Field(default=('R', 'C',),
              description='Define updatable parameters (if any)')
    name: Optional[str] = Field(default='RC', description='Name')

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def value(self,
              soc: Union[float, List, np.ndarray],
              temp: Union[float, List, np.ndarray, None] = None
              ) -> List[Union[float, np.ndarray]]:
        """
        Returns values of resistance and capacitance at given SOC and temperature.
        """
        r_val = self.R.value(soc=soc, temp=temp)
        c_val = self.C.value(soc=soc)
        return [r_val, c_val]


class ReferenceOCV(InterpolatedHealth):
    base_values: Union[float, List] = \
        Field(
            description='Values of reference OCV at specified SOCs. Units: V')
    reference_temperature: float = \
        Field(default=25,
              description='Reference temperature for OCV0. Units: °C')
    name: Literal['OCV0'] = Field(default='OCV0', allow_mutation=False)


class EntropicOCV(InterpolatedHealth):
    base_values: Union[float, List] = \
        Field(
            default=0,
            description='Values of entropic OCV term at specified SOCs. Units: V/°C')
    name: Literal['OCVentropic'] = Field(default='OCVentropic',
                                         allow_mutation=False)


class OpenCircuitVoltage(HealthVariableCollection):
    OCV0: ReferenceOCV = \
        Field(description='Reference OCV at specified temperature')
    OCVentropic: EntropicOCV = \
        Field(description='Entropic OCV to determine temperature dependence')
    updatable: Union[Literal[False], tuple[str, ...]] = \
        Field(default=False,
              description='Define updatable parameters (if any)')
    name: Literal['OCV'] = Field(default='OCV',
                                 allow_mutation=False)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def value(self,
              soc: Union[float, List, np.ndarray],
              temp: Union[float, List, np.ndarray, None] = None
              ) -> Union[float, np.ndarray]:
        """
        Returns values of OCV at given SOC(s) and temperature(s).
        """
        T_ref = self.OCV0.reference_temperature
        delta_T = temp - T_ref
        ocv = self.OCV0.value(soc=soc)
        ocv += delta_T * self.OCVentropic.value(soc=soc)
        return ocv

    def __call__(self,
                 soc: Union[float, List, np.ndarray],
                 temp: Union[float, List, np.ndarray, None] = None
                 ) -> Union[float, np.ndarray]:
        """
        Allows this to be called and used as a function
        """
        return self.value(soc=soc, temp=temp)


class HysteresisParameters(InterpolatedHealth):
    gamma: float = Field(default=0.,
                         description='Exponential approach rate. Units: 1/V',
                         ge=0.)


class ECMASOH(AdvancedStateOfHealth):
    Qt: MaxTheoreticalCapacity = \
        Field(description='Maximum theoretical discharge capacity (Qt).')
    CE: CoulombicEfficiency = \
        Field(default=CoulombicEfficiency(),
              description='Coulombic effiency (CE)')
    OCV: OpenCircuitVoltage = \
        Field(description='Open Circuit Voltage (OCV)')
    R0: SeriesResistance = \
        Field(description='Series Resistance (R0)')


################################################################################
#                              MODEL DEFINITION                                #
################################################################################
class EquivalentCircuitModel(HealthModel):
    """
    Class to model a battery cell as an equivalent circuit
    """

    def __init__(self,
                 use_series_capacitor: bool = False,
                 number_RC_components: int = 0,
                 ASOH: ECMASOH = None,
                 current_behavior: Literal['constant', 'linear'] = 'constant'
                 ) -> None:
        """
        Initialization of ECM.

        Arguments
        ---------
        use_series_capacitor: bool = False
            Boolean to determine whether or not to employ a series capacitor.
            Defaults to False
        number_RC_components: int = 0
            Number of RC components of equivalent circuit. Must be non-negative.
            Defaults to 0.0
        ASOH: ECMASOH = None
            Advanced State of Health (A-SOH) of the system. Used to parametrize
            the dynamics of the system. It does not need to be provided on
            initialization, but, if that is the case, it must be set on
            subsequent function calls.
            Defaults to None
        current_behavior: Literal['constant', 'linear'] = 'constant'
            Determines how to the total current behaves in-between time steps.
            Can be either 'constant' or 'linear'.
            Defaults to 'constant'
        """
        self.num_C0 = int(use_series_capacitor)
        self.num_RC = number_RC_components
        self.current_behavior = current_behavior
        self.asoh = ASOH
        # Lenght of hidden vector: SOC + q0 + I_RC_j + hysteresis
        self.len_hidden = int(1 + self.num_C0 + self.num_RC + 1)

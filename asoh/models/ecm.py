from warnings import warn
from numbers import Number
from typing import List, Optional, Union, Literal, Callable, Sized
import numpy as np
from pydantic import Field, computed_field, validate_call, ConfigDict
from scipy.interpolate import interp1d

from .base import (InputQuantities,
                   OutputMeasurements,
                   HiddenVector,
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
              soc: Union[float, List, np.ndarray],
              *args, **kwargs) -> Union[float, np.ndarray]:
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
    updatable: Union[Literal[False], tuple[str, ...]] = \
        Field(default=('base_values', 'gamma'),
              description='Define updatable parameters (if any)')


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
    H0: HysteresisParameters = \
        Field(default=HysteresisParameters(base_values=0.0, updatable=False),
              description='Hysteresis component')


################################################################################
#                               HIDDEN VECTOR                                  #
################################################################################
class ECMHiddenVector(HiddenVector):
    soc: float = Field(default=0.0, description='State of charge (SOC)')
    q0: Optional[float] = \
        Field(default=[],
              description='Charge in the series capacitor. Units: Coulomb')
    i_rc: Optional[Union[float, List]] = \
        Field(default=[],
              description='Currents through RC components. Units: Amp')
    hyst: float = Field(default=0, description='Hysteresis voltage. Units: V')


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
        self.transient = self.provide_transient_template()

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def provide_transient_template(self,
                                   soc: float = 0.0,
                                   temp: Union[float, None] = None,
                                   capacitive_overpotential:
                                   Union[float, None] = None,
                                   RC_overpotentials:
                                   Union[float, List[float], None] = None,
                                   hysteresis: float = 0.0,
                                   num_copies: int = 1
                                   ) -> Union[ECMHiddenVector,
                                              List[ECMHiddenVector]]:
        """
        Function to provide copies ECMHiddenVector specific to this ECM

        Arguments
        ---------
        soc: float = 0.0
            Value of SOC to be used.
            Defaults to 0.0
        temp: Union[float, None] = None
            Value of temperature in °C to be used for computing values of
            electrical properties.
            Defaults to None
        capacitive_overpotential: Union[float, None] = None
            Potential across series capacitor.
            Defaults to None
        RC_overpotentials: Union[float, List[float], None] = None
            Value(s) of potential difference accross RC components.
            Defaults to None
        hysteresis: float = 0.0
            Value of hysteresis voltage in Volts.
            Defaults to 0.0
        num_copies: int = 1
            Number of copies of a hidden state to provide. Copies can be helpful
            in case we want to evolve the system differently.
            Defaults to 1

        Outputs
        -------
        hidden: Union[ECMHiddenVector, List[ECMHiddenVector]]
            Either a single instance of ECMHiddenVector, in case num_copies==1,
            or a list of num_copies copies of the ECMHiddenVector
        """
        # Dealing with q0 first
        if capacitive_overpotential is not None and self.num_C0 == 0:
            msg = 'No series capacitors found, so capacitive overpotential '
            msg += 'will be ignored!'
            warn(msg)
            q0 = []
        else:
            try:
                capacitance = self.asoh.C0.value(soc=soc)
            except AttributeError as AttrErr:
                raise ValueError('Remember to include C0 field in A-SOH!') \
                    from AttrErr
            q0 = capacitance * capacitive_overpotential

        # Now, dealing with i_rc
        i_rc = [0 for _ in range(self.num_RC)]
        if RC_overpotentials is not None and self.num_RC == 0:
            msg = 'No RC elements found, so RC overpotential(s) '
            msg += 'will be ignored!'
            warn(msg)
        else:
            if isinstance(RC_overpotentials, Sized):
                if len(RC_overpotentials) != self.num_RC:
                    raise ValueError('Mismatch between number of RC '
                                     'overpotentials provided (' +
                                     str(len(RC_overpotentials)) + ') and '
                                     'number of RC elements set (' +
                                     str(self.num_RC) + ')!')
                for i, RC_dV in enumerate(RC_overpotentials):
                    RC_name = 'RC_%d' % (i + 1)
                    try:
                        RC_resistor = getattr(self.asoh, RC_name).R
                    except AttributeError as AttrErr:
                        raise ValueError('Missing \'' + RC_name + '\' from '
                                         'ASOH!') from AttrErr
                    resistance = RC_resistor.value(soc=soc, temp=temp)
                    i_rc[i] = RC_dV / resistance
            elif isinstance(RC_overpotentials, Number):
                for i in range(self.num_RC):
                    RC_name = 'RC_%d' % (i + 1)
                    try:
                        RC_resistor = getattr(self.asoh, RC_name).R
                    except AttributeError as AttrErr:
                        raise ValueError('Missing \'' + RC_name + '\' from '
                                         'ASOH!') from AttrErr
                    resistance = RC_resistor.value(soc=soc, temp=temp)
                    i_rc[i] = RC_overpotentials / resistance

        # Now, we can build the hidden vector
        hidden = ECMHiddenVector(soc=soc, q0=q0, i_rc=i_rc, hyst=hysteresis)
        # Sanity check TODO (vventuri): remove after implementing unit tests
        if len(hidden) != self.len_hidden:
            raise ValueError('Mismatch between expected and real length of '
                             'hidden vector for this ECM!')

        if num_copies == 1:
            return hidden.model_copy()
        return [hidden.model_copy() for _ in range(num_copies)]

    def update_transient_state(self,
                               input: Union[ECMInput, List[ECMInput]],
                               transient_state: Union[ECMHiddenVector,
                                                      List[ECMHiddenVector],
                                                      None] = None,
                               asoh: Union[ECMASOH, List[ECMASOH], None] = None,
                               *args, **kwargs
                               ) -> Union[ECMHiddenVector, List[ECMHiddenVector]]:
        """
        Update transient state.
        Remember how the hidden state is setup and meant to be updated assuming
        constant current behavior:
        soc_(k+1) = soc_k + (coulombic_eff * delta_t * current / Q_total)
        q0_(k+1) = q0_(k) - delta_t * current
        I_(j, k+1) = [exp(-delta_t/(R_(j,k)*C_(j,k))) * I_(j,k)] +
                        + [1 - exp(-delta_t/(R_(j,k)*C_(j,k))) * current]
        hyst_(k+1) = TODO (vventuri)
        """
        pass

    def predict_output(
            self,
            input: Union[ECMInput, List[ECMInput]],
            transient_state: Union[HiddenVector, List[HiddenVector]],
            asoh: Union[AdvancedStateOfHealth, List[AdvancedStateOfHealth]],
            *args, **kwargs) -> Union[ECMMeasurement, List[ECMMeasurement]]:
        """
        Predicts model output, including terminal voltage.
        """
        pass

    def calculate_terminal_voltage(
            self,
            input: ECMInput,
            transient_state: Union[HiddenVector, None] = None,
            asoh: Union[AdvancedStateOfHealth, None] = None,
            *args, **kwargs) -> ECMMeasurement:
        """
        Compute expected output (terminal voltage, etc.) of a the model.
        Recall the calculation of terminal voltage:
        V_T = OCV(SOC,T) +
                + [current * R0(SOC,T)] +
                + [q_i / C0(SOC)] +
                + Sum[I_j * R_j(SOC,T)] +
                + hyst(SOC,T)
        """
        if transient_state is None:
            transient_state = self.transient.model_copy()
        if asoh is None:
            asoh = self.asoh.model_copy()
        # Start with OCV
        Vt = asoh.OCV(soc=transient_state.soc, temp=input.temperature)

        # Add I*R drop ('DCIR')
        Vt += input.current * asoh.R0.value(soc=transient_state.soc,
                                            temp=input.temperature)

        # Check series capacitance
        if transient_state.q0:
            Vt += transient_state.q0 / asoh.C0.value(soc=transient_state.soc)

        # Check RC elements
        if transient_state.i_rc:
            if isinstance(transient_state.i_rc, Number):
                RC_resistor = asoh.RC_1.R
                RC_resistance = RC_resistor.value(soc=transient_state.soc,
                                                  temp=input.temperature)
                Vt += transient_state.i_rc * RC_resistance
            elif isinstance(transient_state.i_rc, Sized):
                for i, i_rc in enumerate(transient_state.i_rc):
                    RC_name = 'RC_%d' % (i + 1)
                    RC_resistor = getattr(asoh, RC_name).R
                    RC_resistance = RC_resistor.value(soc=transient_state.soc,
                                                      temp=input.temperature)
                    Vt += transient_state.i_rc * RC_resistance

        # Include hysteresis
        Vt += transient_state.hyst

        return ECMMeasurement(terminal_voltage=Vt)

# General imports
from typing import Union, Sized, Literal
from numbers import Number

# ASOH imports
from asoh.models.base import CellModel

# Internal imports
from asoh.models.ecm.ins_outs import ECMInput, ECMMeasurement
from asoh.models.ecm.transient import (ECMTransientVector,
                                       provide_transient_template)
from .advancedSOH import ECMASOH, provide_asoh_template


################################################################################
#                                    MODEL                                     #
################################################################################
class EquivalentCircuitModel(CellModel):
    def __init__(self,
                 use_series_capacitor: bool = False,
                 number_RC_components: int = 0,
                 ASOH: ECMASOH = None,
                 transient: ECMTransientVector = None,
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
        if ASOH is None:
            ASOH = provide_asoh_template(has_C0=use_series_capacitor,
                                         num_RC=number_RC_components)
        self.asoh = ASOH
        # Lenght of hidden vector: SOC + q0 + I_RC_j + hysteresis
        self.len_hidden = int(1 + self.num_C0 + self.num_RC + 1)
        if not transient:
            transient = provide_transient_template(has_C0=use_series_capacitor,
                                                   num_RC=number_RC_components)
        else:
            if len(transient) != self.len_hidden:
                raise ValueError('Mismatch between expected length of physical '
                                 'transient hidden state and the transient '
                                 'state provided!')
        self.transient = transient

    def calculate_terminal_voltage(
            self,
            ecm_input: ECMInput,
            transient_state: Union[ECMTransientVector, None] = None,
            asoh: Union[ECMASOH, None] = None,
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
        Vt = asoh.OCV(soc=transient_state.soc, temp=ecm_input.temperature)

        # Add I*R drop ('DCIR')
        Vt += ecm_input.current * asoh.R0.value(soc=transient_state.soc,
                                                temp=ecm_input.temperature)

        # Check series capacitance
        if transient_state.q0:
            Vt += transient_state.q0 / asoh.C0.value(soc=transient_state.soc)

        # Check RC elements
        if transient_state.i_rc:
            if isinstance(transient_state.i_rc, Number):
                RC_resistor = asoh.RCelements[0]
                RC_resistance = RC_resistor.value(soc=transient_state.soc,
                                                  temp=ecm_input.temperature)
                Vt += transient_state.i_rc * RC_resistance
            elif isinstance(transient_state.i_rc, Sized):
                for i_rc, RCel in zip(transient_state.i_rc, asoh.RCelements):
                    RC_resistor = RCel.R
                    RC_resistance = RC_resistor.value(soc=transient_state.soc,
                                                      temp=ecm_input.temperature)
                    Vt += i_rc * RC_resistance

        # Include hysteresis
        Vt += transient_state.hyst

        return ECMMeasurement(terminal_voltage=Vt)

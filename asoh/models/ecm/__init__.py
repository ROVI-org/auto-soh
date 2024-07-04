# General imports
from pydantic import Field
from typing import Tuple, Union, Optional, Sized
from numbers import Number

# ASOH imports
from asoh.models.base import HealthVariable, CellModel

# Internal imports
from .components import (MaxTheoreticalCapacity,
                         CoulombicEfficiency,
                         Resistance,
                         Capacitance,
                         RCComponent,
                         OpenCircuitVoltage,
                         HysteresisParameters)
from asoh.models.ecm.ins_outs import ECMInput, ECMMeasurement
from asoh.models.ecm.transient import ECMTransientVector


################################################################################
#                                    A-SOH                                     #
################################################################################
class ECMASOH(HealthVariable):
    Qt: MaxTheoreticalCapacity = \
        Field(description='Maximum theoretical discharge capacity (Qt).')
    CE: CoulombicEfficiency = \
        Field(default=CoulombicEfficiency(),
              description='Coulombic effiency (CE)')
    OCV: OpenCircuitVoltage = \
        Field(description='Open Circuit Voltage (OCV)')
    R0: Resistance = \
        Field(description='Series Resistance (R0)')
    C0: Optional[Capacitance] = \
        Field(default_factory=None,
              description='Series Capacitance (C0)',
              max_length=1)
    RCelements: Tuple[RCComponent, ...] = \
        Field(default=tuple,
              description='Tuple of RC components')
    H0: HysteresisParameters = \
        Field(default=HysteresisParameters(base_values=0.0, updatable=False),
              description='Hysteresis component')


################################################################################
#                                    MODEL                                     #
################################################################################
class EquivalentCircuitModel(CellModel):
    def __init__(self) -> None:
        pass

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

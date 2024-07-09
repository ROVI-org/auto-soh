"""Models that describe the state of health and transient state of an Equivalent Circuit Model"""
from typing import Literal

import numpy as np

from asoh.models.base import CellModel
from asoh.models.ecm.ins_outs import ECMInput, ECMMeasurement
from asoh.models.ecm.transient import ECMTransientVector
from .advancedSOH import ECMASOH
from .utils import hysteresis_solver_const_sign


class EquivalentCircuitModel(CellModel):
    """
    Equivalent Circuit Model (ECM) dynamics of a battery
    """

    @staticmethod
    def update_transient_state(new_input: ECMInput,
                               transient_state: ECMTransientVector,
                               asoh: ECMASOH,
                               previous_input: ECMInput,
                               current_behavior: Literal['constant', 'linear'] = 'constant'
                               ) -> ECMTransientVector:
        """
        Update transient state.
        Remember how the hidden state is setup and meant to be updated assuming
        constant current behavior:
        soc_(k+1) = soc_k +
            [coulombic_eff * delta_t * (I_k + (0.5 * I_slope * delta_t))/ Q_total]
        q0_(k+1) = q0_k - delta_t * I_k
        i_(j, k+1) = [exp(-delta_t/Tau_j,k) * i_j,k] +
            + [(1 - exp(-delta_t/Tau_j,k) * (I_k - I_slope * Tau_j,k)]
        hyst_(k+1) = [see code, it's messy]
        """
        # Get basic info
        delta_t = new_input.time - previous_input.time
        current_k = previous_input.current
        temp_k = previous_input.temperature
        current_kp1 = new_input.current
        current_slope = 0.0 if current_behavior == 'constant' else (current_kp1 - current_k) / delta_t
        # We will assume that all health parameters remain constant between time
        # steps, independent of temperature or SOC variations. The value used
        # will be the one at the previous SOC and temperature values.

        # Set Coulombic efficiency to 1. if discharging
        coul_eff = 1 if current_k < 0 else asoh.ce

        # Update SOC
        soc_k = transient_state.soc
        Qt = asoh.q_t.value
        charge_cycled = delta_t * (current_k + ((current_slope * delta_t) / 2))
        soc_kp1 = soc_k + (coul_eff * (charge_cycled / Qt))

        # Update q0
        q0_kp1 = transient_state.q0
        if q0_kp1 is not None:
            q0_kp1 += charge_cycled

        # Update i_RCs
        iRC_kp1 = transient_state.i_rc
        if iRC_kp1 is not None:
            tau = np.array([RC.time_constant(soc=soc_k, temp=temp_k)
                            for RC in asoh.rc_elements])
            exp_factor = np.exp(-delta_t / tau)
            iRC_kp1 *= exp_factor
            iRC_kp1 += (1 - exp_factor) * \
                       (new_input.current - (current_slope * tau))
            iRC_kp1 += current_slope * delta_t

        # Update hysteresis
        hyst_kp1 = transient_state.hyst
        # Needed parameters
        M = asoh.h0.get_value(soc=soc_k)
        # Recall that, if charging, than M has to be >0, but, if dischargin, it
        # has to be <0. The easiest way to check for that is to multiply by the
        # current and divide by its absolute value
        M *= current_k
        if current_k != 0:
            M *= current_k / abs(current_k)

        gamma = asoh.h0.gamma
        kappa = (coul_eff * gamma) / Qt
        # We need to figure out if the current changes sign during this process
        if current_k * current_kp1 >= 0:  # easier case
            hyst_kp1 = hysteresis_solver_const_sign(h0=transient_state.hyst,
                                                    M=M,
                                                    kappa=kappa,
                                                    dt=delta_t,
                                                    i0=current_k,
                                                    alpha=current_slope)
        # If the curret flips sign, we need to deal with two intervals
        else:
            # solving for time until current == 0
            phi = -current_k / current_slope
            h_mid = hysteresis_solver_const_sign(h0=transient_state.hyst,
                                                 M=M,
                                                 kappa=kappa,
                                                 dt=phi,
                                                 i0=current_k,
                                                 alpha=current_slope)
            # Use this new value to update for the next interval. Remember to:
            # 1. use the new hysteresis value
            # 2. flip the sign of maximum hysteresis to follow current
            # 3. use the starting current of 0.0
            hyst_kp1 = hysteresis_solver_const_sign(h0=h_mid,
                                                    M=-M,
                                                    kappa=kappa,
                                                    dt=phi,
                                                    i0=0.0,
                                                    alpha=current_slope)

        return ECMTransientVector(soc=soc_kp1,
                                  q0=q0_kp1,
                                  i_rc=iRC_kp1,
                                  hyst=hyst_kp1)

    @staticmethod
    def calculate_terminal_voltage(new_input: ECMInput,
                                   transient_state: ECMTransientVector,
                                   asoh: ECMASOH) -> ECMMeasurement:
        """
        Compute expected output (terminal voltage, etc.) of a the model.
        Recall the calculation of terminal voltage:
        V_T = OCV(SOC,T) +
                + [current * R0(SOC,T)] +
                + [q_i / C0(SOC)] +
                + Sum[I_j * R_j(SOC,T)] +
                + hyst(SOC,T)
        """
        # Start with OCV
        Vt = asoh.ocv(soc=transient_state.soc, temp=new_input.temperature)

        # Add I*R drop ('DCIR')
        Vt += new_input.current * asoh.r0.get_value(soc=transient_state.soc,
                                                    temp=new_input.temperature)

        # Check series capacitance
        if transient_state.q0 is not None:
            Vt += transient_state.q0 / asoh.c0.get_value(soc=transient_state.soc)

        # Check RC elements
        if transient_state.i_rc is not None:
            RC_Rs = np.array(
                [RC.r.get_value(soc=transient_state.soc,
                                temp=new_input.temperature)
                 for RC in asoh.rc_elements]
            )
            V_drops = transient_state.i_rc * RC_Rs
            Vt += sum(V_drops)

        # Include hysteresis
        Vt += transient_state.hyst

        return ECMMeasurement(terminal_voltage=Vt)

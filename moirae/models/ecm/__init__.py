"""Models that describe the state of health and transient state of an Equivalent Circuit Model"""
from typing import Literal

import numpy as np

from ..base import CellModel
from moirae.models.ecm.ins_outs import ECMInput, ECMMeasurement
from moirae.models.ecm.transient import ECMTransientVector
from .advancedSOH import ECMASOH
from .utils import hysteresis_solver_const_sign


# TODO (wardlt): Does "constant" mean the previous or current value is used in span between last timestep and current.
class EquivalentCircuitModel(CellModel):
    """
    Equivalent Circuit Model (ECM) dynamics of a battery

    The only option for how to implement the battery is whether we assume the current
    varies linearly between the previous and current step, or whether it is assumed to be constant.

    Args:
        current_behavior: How the current is assumed to vary between timesteps.
    """

    current_behavior: str
    """How it is assumed the current varies between timesteps"""

    def __init__(self, current_behavior: Literal['constant', 'linear'] = 'constant'):
        self.current_behavior = current_behavior

    def update_transient_state(
            self,
            previous_inputs: ECMInput,
            new_inputs: ECMInput,
            transient_state: ECMTransientVector,
            asoh: ECMASOH
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
        delta_t = new_inputs.time - previous_inputs.time
        current_k = previous_inputs.current
        temp_k = previous_inputs.temperature
        current_kp1 = new_inputs.current
        current_slope = 0.0 if self.current_behavior == 'constant' else (current_kp1 - current_k) / delta_t

        # We will assume that all health parameters remain constant between time
        # steps, independent of temperature or SOC variations. The value used
        # will be the one at the previous SOC and temperature values.

        # Set Coulombic efficiency to 1. if discharging
        coul_eff = 1 if current_k < 0 else asoh.ce

        # Update SOC
        soc_k = transient_state.soc.copy()
        Qt = asoh.q_t.value
        charge_cycled = delta_t * (current_k + ((current_slope * delta_t) / 2))
        soc_kp1 = soc_k + (coul_eff * (charge_cycled / Qt))

        # Update q0
        q0_kp1 = None
        if transient_state.q0 is not None:
            q0_kp1 = transient_state.q0 + charge_cycled

        # Update i_RCs
        iRC_kp1 = transient_state.i_rc.copy()  # Shape: batch_size, num_rc
        if iRC_kp1.shape[1] > 0:  # If there are RC elements
            tau = np.array([RC.time_constant(soc=soc_k, temp=temp_k)
                            for RC in asoh.rc_elements])[:, :, 0].T  # Shape: batch_size, num_rc
            exp_factor = np.exp(-delta_t / tau)
            iRC_kp1 *= exp_factor
            iRC_kp1 += (1 - exp_factor) * \
                       (current_kp1 - (current_slope * tau))
            iRC_kp1 += current_slope * delta_t

        # Update hysteresis
        hyst_kp1 = transient_state.hyst.copy()
        # Needed parameters
        M = asoh.h0.get_value(soc=soc_k)
        # Recall that, if charging, than M has to be >0, but, if dischargin, it
        # has to be <0. The easiest way to check for that is to multiply by the
        # current and divide by its absolute value
        M *= current_k
        if current_k != 0:
            M /= abs(current_k)

        gamma = asoh.h0.gamma
        kappa = (coul_eff * gamma) / Qt
        # We need to figure out if the current changes sign during this process

        if current_k * current_kp1 >= 0 or self.current_behavior == 'constant':
            hyst_kp1 = hysteresis_solver_const_sign(h0=transient_state.hyst.copy(),
                                                    M=M,
                                                    kappa=kappa,
                                                    dt=delta_t,
                                                    i0=current_k,
                                                    alpha=current_slope)
        # If the current flips sign, we need to deal with two intervals
        else:
            # solving for time until current == 0
            phi = -current_k / current_slope
            h_mid = hysteresis_solver_const_sign(h0=transient_state.hyst.copy(),
                                                 M=M,
                                                 kappa=kappa,
                                                 dt=phi,
                                                 i0=current_k,
                                                 alpha=current_slope)
            # Use this new value to update for the next interval. Remember to:
            # 1. use the new hysteresis value
            # 2. flip the sign of maximum hysteresis to follow current
            # 3. use the remaining time interval of delta_t - phi
            # 4. use the starting current of 0.0
            hyst_kp1 = hysteresis_solver_const_sign(h0=h_mid,
                                                    M=-M,
                                                    kappa=kappa,
                                                    dt=(delta_t - phi),
                                                    i0=0.0,
                                                    alpha=current_slope)

        return ECMTransientVector(soc=soc_kp1,
                                  q0=q0_kp1,
                                  i_rc=iRC_kp1,
                                  hyst=hyst_kp1)

    def calculate_terminal_voltage(
            self,
            new_inputs: ECMInput,
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
        Vt = asoh.ocv(soc=transient_state.soc, temp=new_inputs.temperature)

        # Add I*R drop ('DCIR')
        Vt += new_inputs.current * asoh.r0.get_value(soc=transient_state.soc,
                                                     temp=new_inputs.temperature)

        # Check series capacitance
        if transient_state.q0 is not None:
            Vt += transient_state.q0.copy() / asoh.c0.get_value(soc=transient_state.soc.copy())

        # Check RC elements
        if transient_state.i_rc.shape[-1] > 0:
            rc_rs = np.array(
                [rc.r.get_value(soc=transient_state.soc,
                                temp=new_inputs.temperature)
                 for rc in asoh.rc_elements]
            )  # Shape: (rc_rs, batch_dim, 1 resistance)
            V_drops = transient_state.i_rc * rc_rs[:, :, 0].T
            Vt += np.sum(V_drops, axis=1, keepdims=True)

        # Include hysteresis
        Vt += transient_state.hyst

        return ECMMeasurement(terminal_voltage=Vt)

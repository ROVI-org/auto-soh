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
        # Update transient state.
        # Remember how the hidden state is setup and meant to be updated assuming
        # constant current behavior:
        # soc_(k+1) = soc_k +
        #     [coulombic_eff * delta_t * (I_k + (0.5 * I_slope * delta_t))/ Q_total]
        # q0_(k+1) = q0_k - delta_t * I_k
        # i_(j, k+1) = [exp(-delta_t/Tau_j,k) * i_j,k] +
        #     + [(1 - exp(-delta_t/Tau_j,k) * (I_k - I_slope * Tau_j,k)]
        # hyst_(k+1) = [see code, it's messy]

        # Get basic info
        delta_t = new_inputs.time - previous_inputs.time  # shape (time_batch, 1)
        current_k = previous_inputs.current  # shape (current_batch, 1)
        temp_k = previous_inputs.temperature  # shape (temp_batch, 1)
        current_kp1 = new_inputs.current  # shape (current_batch, 1)
        current_slope = np.atleast_2d(0.0) if self.current_behavior == 'constant' \
            else (current_kp1 - current_k) / delta_t  # shape (trans_batch, 1)

        # We will assume that all health parameters remain constant between time
        # steps, independent of temperature or SOC variations. The value used
        # will be the one at the previous SOC and temperature values.

        # Set Coulombic efficiency to 1. if discharging
        coul_eff = np.atleast_2d(1) if current_k < 0 else asoh.ce  # shape (ce_batch, 1)

        # Update SOC
        soc_k = transient_state.soc.copy()  # shape (soc_batch, 1)
        Qt = asoh.q_t.value  # shape (qt_batch, 1)
        charge_cycled = delta_t * (current_k + ((current_slope * delta_t) / 2))  # shape (trans_batch, 1)
        # NOTE: the below operation only works if batches are broadcastable!!!
        soc_kp1 = soc_k + (coul_eff * (charge_cycled / Qt))  # shape (trans_batch, 1)

        # Update q0
        q0_kp1 = None
        if transient_state.q0 is not None:
            q0_kp1 = transient_state.q0 + charge_cycled  # shape (trans_batch, 1)

        # Update i_RCs
        iRC_kp1 = transient_state.i_rc.copy()  # Shape: (trans_batch_size, num_rc)
        if iRC_kp1.shape[1] > 0:  # If there are RC elements
            tau = np.array([RC.time_constant(soc=soc_k, temp=temp_k)
                            for RC in asoh.rc_elements])  # Shape: (num_rc, rc_batch, soc_batch, soc_dim=1)
            assert len(tau.shape) == 4, f'RC time constant has shape {tau.shape}D, instead of 4D!'
            # NOTE: we already assume that the A-SOH batch and transient batch are broadcastable!
            if tau.shape[1] == tau.shape[2]:
                tau = np.diagonal(tau, axis1=1, axis2=2)  # shape (num_rc, 1, batch_size)
                tau = np.swapaxes(tau, axis1=1, axis2=2)  # shape (num_rc, batch_size, 1)
            # Make sure this is 2D in a way that takes care of the case where one or both of the batche sizes is 1
            tau = tau.reshape((tau.shape[0], -1))  # shape (num_rc, batch_size)
            # Transpose to get shape of (batch_size, num_rc)
            tau = np.swapaxes(tau, axis1=0, axis2=1)
            # Now, tau should be two-dimensional
            assert len(tau.shape) == 2, f'RC time constant has shape {tau.shape}D, instead of 2D!'
            exp_factor = np.exp(-delta_t / tau)  # shape (batch_size, num_rc)
            iRC_kp1 = iRC_kp1 * exp_factor  # shape (batch_size, num_rc)
            iRC_kp1 = iRC_kp1 + ((1 - exp_factor) * (current_kp1 - (current_slope * tau)))  # shape (batch_size, num_rc)
            iRC_kp1 = iRC_kp1 + (current_slope * delta_t)  # shape (batch_size, num_rc)

        # Update hysteresis
        hyst_kp1 = transient_state.hyst.copy()  # shape (hyst_batch, 1)
        # Needed parameters
        M = asoh.h0.get_value(soc=soc_k)  # shape (h0_batch, soc_batch, 1)
        # Recall that, if charging, than M has to be >0, but, if discharging, it
        # has to be <0. The easiest way to check for that is to multiply by the
        # current and divide by its absolute value
        M = M * current_k  # shape (h0_batch, trans_batch, 1)
        if not np.allclose(current_k, 0):
            M = M / abs(current_k)  # shape (h0_batch, trans_batch, 1)

        gamma = asoh.h0.gamma  # shape (h0_batch, 1)
        kappa = (coul_eff * gamma) / Qt  # shape (asoh_batch, 1)
        # We need to figure out if the current changes sign during this process

        if current_k * current_kp1 >= 0 or self.current_behavior == 'constant':
            hyst_kp1 = hysteresis_solver_const_sign(h0=transient_state.hyst,
                                                    M=M,
                                                    kappa=kappa,
                                                    dt=delta_t,
                                                    i0=current_k,
                                                    alpha=current_slope)  # shape (h0_batch, trans_batch, 1)
        # If the current flips sign, we need to deal with two intervals
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
            # 3. use the remaining time interval of delta_t - phi
            # 4. use the starting current of 0.0
            hyst_kp1 = hysteresis_solver_const_sign(h0=h_mid,
                                                    M=-M,
                                                    kappa=kappa,
                                                    dt=(delta_t - phi),
                                                    i0=0.0,
                                                    alpha=current_slope)
        # Now, hyst_kp1 is a 3D array with shape shape (h0_batch, trans_batch, 1), we need to convert it to 2D
        assert len(hyst_kp1.shape) == 3, f'Hysteresis has shape {hyst_kp1.shape}, but we expected a 3D array!'
        if hyst_kp1.shape[0] == hyst_kp1.shape[1]:
            hyst_kp1 = np.diagonal(hyst_kp1, axis1=0, axis2=1)  # shape (1, batch_size)
            hyst_kp1 = np.swapaxes(hyst_kp1, axis1=0, axis2=1)  # shape (batch_size, 1)
        # Now, make sure array is 2D, even when the batches are not equal (which means at least one of them is 1 due to
        # the assumption that they must be broadcastable)
        hyst_kp1 = hyst_kp1.squeeze().reshape((-1, 1))

        return ECMTransientVector(soc=soc_kp1,
                                  q0=q0_kp1,
                                  i_rc=iRC_kp1,
                                  hyst=hyst_kp1)

    def calculate_terminal_voltage(
            self,
            new_inputs: ECMInput,
            transient_state: ECMTransientVector,
            asoh: ECMASOH) -> ECMMeasurement:
        # Recall the calculation of terminal voltage:
        # V_T = OCV(SOC,T) +
        #         + [current * R0(SOC,T)] +
        #         + [q_i / C0(SOC)] +
        #         + Sum[I_j * R_j(SOC,T)] +
        #         + hyst(SOC,T)
        # Start with OCV
        Vt = asoh.ocv(soc=transient_state.soc, temp=new_inputs.temperature)  # shape: (ocv_batch, soc_batch, soc_dim=1)

        # Add I*R drop ('DCIR')
        IR_drop = new_inputs.current  # shape: (current_batch, 1)
        r0 = asoh.r0.get_value(soc=transient_state.soc, temp=new_inputs.temperature)  # (r0_batch, soc_batch, soc_dim=1)
        IR_drop = IR_drop * r0  # broadcasting from right to left gives (r0_batch, trans_batch, soc_dim=1)
        Vt = Vt + IR_drop  # broadcasting from right to left gives (asoh_batch, trans_batch, soc_dim=1)

        # Check series capacitance
        if transient_state.q0 is not None:
            c0 = asoh.c0.get_value(soc=transient_state.soc.copy())  # shape (c0_batch, soc_batch, soc_dim=1)
            q0 = transient_state.q0  # shape (q0_batch, 1)
            deltaV = q0 / c0  # broadcasting from right to left gives (c0_batch, trans_batch, 1)
            Vt = Vt + deltaV  # broadcasting from right to left gives (asoh_batch, trans_batch, 1)

        # Check RC elements
        if transient_state.i_rc.shape[-1] > 0:
            rc_rs = np.array(
                [rc.r.get_value(soc=transient_state.soc,
                                temp=new_inputs.temperature)
                 for rc in asoh.rc_elements]
            )  # Shape: (num_rc, rc.r_batch, soc_batch, soc_dim=1)
            # Transpose to get shape of (rc.r_batch, num_rc, soc_batch, soc_dim=1)
            rc_rs = np.swapaxes(rc_rs, axis1=0, axis2=1)
            # Get currents through resistors in RC
            i_rc = transient_state.i_rc  # shape (i_rc_batch, num_rc)
            # For proper broadcasting, let's turn this into a shape of (num_rc, i_rc_batch, 1)
            i_rc = i_rc.T[:, :, None]
            V_drops = i_rc * rc_rs  # shape (rc.r_batch, num_rc, trans_batch, soc_dim=1)
            V_drop_sum = np.sum(V_drops, axis=1)  # shape (rc.r_batch, trans_batch, soc_dim=1)
            Vt = Vt + V_drop_sum  # shape (asoh_batch_size, trans_batch_size, 1)

        # Include hysteresis
        hyst = transient_state.hyst  # shape (hyst_batch, 1)
        Vt = Vt + hyst  # shape (asoh_batch_size, trans_batch_size, 1)

        # We operate under the assumption that the A-SOH and the transient batch sizes are broadcastable
        if Vt.shape[0] == Vt.shape[1]:
            Vt = np.diagonal(Vt, axis1=0, axis2=1)  # shape (1, batch_size) [idk why numpy puts diagonal dim at the end]
            Vt = Vt.T
        if Vt.shape[0] == 1 or Vt.shape[1] == 1:
            Vt = Vt.reshape((-1, 1))

        return ECMMeasurement(terminal_voltage=Vt)

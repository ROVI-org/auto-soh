"""
Auxiliary functions to be used in the extractors
"""
from typing import Union

import numpy as np


def compute_I_RCs(total_current: np.ndarray,
                  timestamps: np.ndarray,
                  tau_values: Union[float, np.ndarray],
                  qc0s: Union[float, np.ndarray] = 0.) -> np.ndarray:
    """
    Function that computes the current flowing through the resistive element of an RC component at the end of a segment

    Args:
        total_current: array of the total current flowing through the cell (in Amps)
        timestamps: timestamps corresponding to the total current values (in seconds)
        taus: values of relaxation time for the RC components, one value per component (in seconds)
        qc0s: total charge stored in the capacitor in the beginning of the segment, one per RC component. Assumed to be
            zero for all components (in Coulombs)

    Returs:
        values of current through RC pairs, one per pair
    """
    # We really only care about the differences in time
    delta_t = np.diff(timestamps)

    # Now, we will use an implicit "Euler-step", assumig constant values between timestamps.
    # The main equation we wish to solve is
    # dq_C / dt = I_T - (q_C / tau)
    # Organizing it in an implicit way: q_C^(t+1) = q_C^(t) + dt * (I_T^(t+1) - (q_C^(t+1)/tau))
    # gives q_C^(t+1) = (q_C^(t) + dt * I_T^(t+1)) / (1 + dt/tau)

    # Array to keep values of q_C and taus
    qcs = np.array(qc0s).flatten()
    taus = np.array(tau_values).flatten()
    # Step
    for dt, curr in zip(delta_t, total_current[1:]):
        # qcs_dot = curr - (qcs / taus)
        # qcs = qcs + (dt * qcs_dot)
        qcs = (qcs + (dt * curr)) / (1 + (dt / taus))

    return qcs / taus

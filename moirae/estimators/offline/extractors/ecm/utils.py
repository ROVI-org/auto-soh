"""
Auxiliary functions to be used in the extractors
"""
from typing import Tuple, Union

import numpy as np
from scipy.optimize import differential_evolution


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


def fit_exponential_decays(time: np.ndarray, measurements: np.ndarray, n_exp: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to fit exponential decays, useful for extracting parameters from RC components of and ECM

    Args:
        time: timestamps (in seconds) of the measurement time series
        measurements: reported measurements that supposedly follow a sum of exponential decays
        n_exp: number of exponential decays to fit; defaults to 1

    Returns:
        amplitude and relaxation time parameters for each exponential decay
    """
    # Let's first prepare the bounds for the parameters.
    bounds = np.zeros(((2 * n_exp) + 1, 2))
    # Recall the middle third of the parameters are the time constants, which are not observable if smaller than the
    # timestep present
    bounds[n_exp:-1, 0] = np.min(np.abs(np.diff(time)))
    # Similarly, the time constant cannot be larger than the duration of the period, otherwise, the fit is poor
    duration = time[-1] - time[0]
    bounds[n_exp:-1, 1] = duration
    # For the amplitude terms, what matters most is the "direction" of the decay: are we approaching a value larger than
    # the first, or smaller? We will use this difference for our amplitude bounds
    if measurements[-1] > measurements[0]:
        bounds[:n_exp, 0] = (measurements[0] - measurements[-1])
    else:
        bounds[:n_exp, 1] = (measurements[0] - measurements[-1])

    # Finally, the "y-offset", which is ultimately a measure of how for the "steady-state" stable value is from 0
    # Our best guess for it is the last value, but let's make it loose
    offset = abs(measurements[-1])
    bounds[-1, 0] = -10 * offset
    bounds[-1, 1] = 10 * offset

    # Now, we can fit our parameters
    fit_result = differential_evolution(func=sum_squared_error_exp_decay,
                                        bounds=bounds,
                                        popsize=120,
                                        updating='deferred',
                                        vectorized=True,
                                        args=(time, measurements, n_exp))

    # Get the best parameters
    params = fit_result.x.copy()
    # Split amplitudes and relaxation times
    amps = params[:n_exp]
    taus = params[n_exp:-1]
    # Sort by faster to slowest relaxation time
    idx = np.argsort(taus)

    return amps[idx], taus[idx]


def build_sum_exp_decays(params: np.ndarray, time: np.ndarray, n_exp: int) -> np.ndarray:
    """
    Auxliary function to build a sum of exponential decay functions from provided parameters

    Args:
        parameters: array of the parameters to be used in the exponential decay functions; the first axis must be of
            length 2*n_expp, since the first n_exp entries are amplitudes, and the rest are relaxation time constants,
            in the same units as the time array
        time: array of time to be used for building the sum of exponential decays
        n_exp: number of exponential decay functions to build

    Returns:
        values of the sum of the exponential decay for each entry in the time array
    """
    if params.shape[0] == (1 * n_exp) + 1:
        params_len = params.shape[0]
        raise ValueError(f'Number of parameters {params_len} not match number of exponential decays to build {n_exp}!')
    # Get amplitudes and relaxation times
    amps = params[:n_exp]
    taus = params[n_exp:(2 * n_exp)]
    offsets = params[-1:]

    # Build the sum of exponential decays
    t = time - time[0]

    # Now, let's make sure we can use this in vectorized form, that is:
    #   1) if params are passed as an array of shape (N,)
    #   2) if params are passed as an array of shape (N,S), where S is the different number of solution being attempted
    if len(params.shape) == 2:
        num_sols = params.shape[1]
        if num_sols > 1:
            amps = amps.reshape((n_exp, num_sols, 1))
            taus = taus.reshape((n_exp, num_sols, 1))
            offsets = offsets.reshape((1, num_sols, 1))
    else:
        amps = amps.reshape((n_exp, 1))
        taus = taus.reshape((n_exp, 1))
        offsets = offsets.reshape((1, 1))

    decays = offsets + (amps * np.exp(- t / taus))

    # sum over the functions
    decay_sum = decays.sum(axis=0)

    return decay_sum


def sum_squared_error_exp_decay(params: np.ndarray,
                                time: np.ndarray,
                                measurements: np.ndarray,
                                n_exp: int) -> Union[float, np.ndarray]:
    """
    Auxiliary function to compute the sum of squared errors between a sum of exponential decays and the observed
    measurements

    Args:
        params: array of parameters to be used when building the exponential decay functions
        time: timestamps for the measurements
        measurements: obsered measurements
        n_exp: number of exponential decay functions to be used

    Returns:
        sum of squared errors between the sum of exponential decays and the observed measurements
    """
    # Start by building the exponential decays
    decays = build_sum_exp_decays(params=params, time=time, n_exp=n_exp)

    # Compute errors
    sqr_errs = np.pow(measurements - decays, 2)

    return sqr_errs.sum(axis=-1)

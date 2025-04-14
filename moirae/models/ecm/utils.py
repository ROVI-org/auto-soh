from typing import Union

import numpy as np


def realistic_fake_ocv(
        soc_vals: Union[float, np.ndarray]) -> np.ndarray:
    """
    Returns somewhat realistic OCV relationship to SOC
    """
    x_scale = 0.9
    x_off = 0.05
    y_scale = 0.1
    y_off = 3.5
    mod_soc = x_scale * soc_vals
    mod_soc += x_off
    volts = np.log(mod_soc / (1 - mod_soc))
    volts *= y_scale
    volts += y_off
    volts = volts.astype(float)
    return volts


def unrealistic_fake_r0(
        soc_vals: Union[float, np.ndarray]) -> np.ndarray:
    """
    Returns not very realistic R0 relationship to SOC
    """
    ohms = 0.05 * np.ones(np.array(soc_vals).shape)
    return ohms


def unrealistic_fake_rc(
        soc_vals: Union[float, np.ndarray]) -> np.ndarray:
    """
    Returns not very realistic RC element relationships to SOC
    """
    ohms = 0.005*np.ones(np.array(soc_vals).shape)
    farads = 2500*np.ones(np.array(soc_vals).shape)
    return ((ohms, farads), (ohms, 4*farads))


def hysteresis_solver_const_sign(
        h0: Union[float, np.ndarray],
        M: Union[float, np.ndarray],
        kappa: Union[float, np.ndarray],
        dt: Union[float, np.ndarray],
        i0: Union[float, np.ndarray],
        alpha: Union[float, np.ndarray]
) -> float:
    """
    Helper function to solve for hysteresis at time dt given initial conditions,
    parameters, and current and current slope. Assumes current does not change
    sign during time interval

    Args:
        h0: Initial value of hysteresis, corresponding to h[0]
        M: Asymptotic value of hysteresis at present condition (the value h[t] should approach)
        kappa: Constant representing the product of gamma (SOC-based rate at which hysteresis approaches M),
            Coulombic efficienty, and 1/Qt
        dt: Length of time interval
        i0: Initial current
        alpha: Slope of current profile during time interval

    Returns:
        Hysteresis value at the end of the time interval
    """
    assert i0 * (i0 + (alpha * dt)) >= 0, 'Current flips sign in interval dt!!'
    exp_factor = kappa * dt  # shape (broadcasted_batch_size, 1)
    exp_factor = exp_factor * (i0 + (0.5 * alpha * dt))
    # Now, flip the sign depending if current is positive in the interval
    if i0 > -(alpha * dt):  # this indicates (i0 + alpha * t) > 0
        exp_factor = -exp_factor
    exp_factor = np.exp(exp_factor)
    h_dt = exp_factor * h0
    h_dt = h_dt + ((1 - exp_factor) * M)
    return h_dt

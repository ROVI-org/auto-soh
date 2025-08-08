"""
Tests utility functions from extractors
"""
from typing import Tuple

import numpy as np
from pytest import fixture

from moirae.estimators.offline.extractors.ecm.utils import compute_I_RCs


@fixture
def constant_current_proile() -> Tuple[np.ndarray, np.ndarray]:
    """
    Provides a simple constant current profile
    """
    current = 10. * np.ones(3000)
    time = np.arange(3000)
    return time, current


def test_single_RC(constant_current_proile) -> None:
    time, current = constant_current_proile
    current_mean = np.mean(current)

    # Set tau
    tau = int(time.shape[0] / 10)

    # After a single tau, we should be at 1/e on the way there
    i_rc_1tau = compute_I_RCs(total_current=current[:tau+1],
                              timestamps=time[:tau+1],
                              tau_values=tau)
    expected_1tau = current_mean * (1 - (1/np.e))
    assert np.allclose(i_rc_1tau, expected_1tau, rtol=0.001), f'{i_rc_1tau} != {expected_1tau}'

    # After two taus, we should be even closer
    i_rc_2tau = compute_I_RCs(total_current=current[:2*tau+1],
                              timestamps=time[:2*tau+1],
                              tau_values=tau)
    expected_2tau = current_mean * (1 - (1/(np.e**2)))
    assert np.allclose(i_rc_2tau, expected_2tau, rtol=0.001), f'{i_rc_2tau} != {expected_2tau}'

    # At the end of the experiment, we should be basically already there
    i_rc = compute_I_RCs(total_current=current,
                         timestamps=time,
                         tau_values=tau)
    assert np.allclose(i_rc, current_mean, rtol=0.001), f'{i_rc} != {current_mean}'


def test_multiple_RCs(constant_current_proile) -> None:
    time, current = constant_current_proile
    current_mean = np.mean(current)

    # Set taus corresponding to 5 and 25 minutes
    taus = [300, 1500]

    # Now, compute the I_RCs and the expected values
    i_rcs = compute_I_RCs(total_current=current,
                          timestamps=time,
                          tau_values=taus)
    powers = time.shape[0] / np.array(taus)
    expected_i_rcs = current_mean * (1 - (np.pow(np.e, -powers)))
    assert np.allclose(i_rcs, expected_i_rcs, rtol=0.001), f'{i_rcs} != {expected_i_rcs}'


def test_saturation(constant_current_proile) -> None:
    time, current = constant_current_proile
    current_mean = np.mean(current)

    # Set tau and qc0
    tau = int(time.shape[0] / 10)
    qc0 = current_mean * tau

    # After a single tau, we should already be there, as we started saturated
    i_rc_1tau = compute_I_RCs(total_current=current[:tau+1],
                              timestamps=time[:tau+1],
                              tau_values=tau,
                              qc0s=qc0)
    assert np.allclose(i_rc_1tau, current_mean, rtol=0.001), f'{i_rc_1tau} != {current_mean}'

    # At the end of the experiment, we should remain at I_total
    i_rc = compute_I_RCs(total_current=current,
                         timestamps=time,
                         tau_values=tau,
                         qc0s=qc0)
    assert np.allclose(i_rc, current_mean, rtol=0.001), f'{i_rc} != {current_mean}'

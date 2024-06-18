"""Test the unscented Kálmán filter"""

from pytest import fixture
import numpy as np

from asoh.observers.ukf import UnscentedKalmanFilter
from asoh.models.ecm import SingleResistorModel, ECMState, ECMControl, ECMOutputs


@fixture()
def initial_ecm_model() -> tuple[SingleResistorModel, ECMState]:
    """Create a model and estimated initial state"""
    return SingleResistorModel(), ECMState(r_serial=1., ocv_params=(1., 0.5), health_params=('r_serial',))


def test_ukf(initial_ecm_model):
    model, state = initial_ecm_model

    # Create the ukf
    ukf = UnscentedKalmanFilter(model, state, covariance_process_noise=np.eye(2,) * 1e-3, covariance_sensor_noise=np.array([[1e-3]]))
    assert ukf.cov_w.shape == (2, 2)
    assert ukf.cov_v.shape == (1, 1)
    assert ukf._aug_len == 2 * 2 + 1  # Two state variables, 1 output
    assert np.isclose(ukf.mean_weights.sum(), 1)

    # Test making sigma points using the current state estimation
    sigma = ukf.build_sigma_pts()
    assert sigma.shape == (2 * ukf._aug_len + 1, ukf._aug_len)
    mean_point = sigma.mean(axis=0)
    assert np.isclose(mean_point[:2], state.full_state).all()  # Mean of the sigma point should be the current state estimate
    assert np.isclose(mean_point[2:], 0).all()  # Noise parameters start at 0

    # Test computing the time update for the sigma points
    u = ECMControl(current=1.)
    cov_xy_k_minus, y_hat_k, cov_y_k = ukf.estimation_update(sigma, u, 1.)
    assert cov_xy_k_minus.shape == (2, 1)
    assert cov_y_k.shape == (1, 1)
    assert y_hat_k.shape == (1,)
    assert np.isclose(y_hat_k, [2.], atol=1e-3)  # Voltage should be about 1V from OCV and 1V from 1 Ohm resistor

    # Step, giving outputs which agree given the currently-supposed state
    #  After 1 second of charging at 1 A, the OCV should be 1 + 1/3600 V, and the voltage from the resistor should be 1 V
    ukf.step(u, ECMOutputs(terminal_voltage=2. + 1. / 3600), t_step=1)

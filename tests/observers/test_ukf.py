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
    ukf = UnscentedKalmanFilter(model, state)
    ukf.cov_Y.shape = (1, 1)

    # Step, giving outputs which agree given the currently-supposed state
    #  After 1 second of charging at 1 A, the OCV should be 1 + 1/3600 V, and the voltage from the resistor should be 1 V
    current = ECMControl(current=1.)
    ukf.step(current, ECMOutputs(terminal_voltage=2. + 1. / 3600), t_step=1)

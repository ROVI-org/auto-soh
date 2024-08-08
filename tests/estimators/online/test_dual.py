import numpy as np

from moirae.estimators.online.kalman.dual import DualUnscentedKalmanFilter
from moirae.models.ecm import ECMInput, ECMMeasurement


def test_step(simple_rint):
    """Test stepping the UKF"""

    # Make the serial resistor updatable
    rint_asoh, rint_transient, ecm_inputs, ecm_model = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    assert rint_asoh.num_updatable == 1

    # Make the filter without normalization
    ukf_dual = DualUnscentedKalmanFilter(
        model=ecm_model,
        initial_asoh=rint_asoh,
        initial_transients=rint_transient,
        initial_inputs=ecm_inputs,
        initial_transient_covariance=np.diag([0.1, 0.1]),
        initial_asoh_covariance=np.atleast_2d([[0.01]]),
    )

    end_soc = (1. / rint_asoh.q_t.value).item()
    end_voltage = (rint_asoh.ocv.get_value(soc=end_soc) + 1. * rint_asoh.r0.get_value(soc=end_soc)).item()
    ukf_dual.step(
        ECMInput(time=1, current=1),
        ECMMeasurement(terminal_voltage=end_voltage)
    )

import numpy as np

from moirae.estimators.online import DeltaDistribution
from moirae.estimators.online.kalman.dual import DualUnscentedKalmanFilter


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

    applied_control = DeltaDistribution(mean=np.array([1., 1., 25.]))  # Time=1s, I=1 Amp, 25C
    end_soc = (1. / rint_asoh.q_t.value).item()
    end_voltage = (rint_asoh.ocv.get_value(soc=end_soc) + 1. * rint_asoh.r0.get_value(soc=end_soc)).item()
    observed_voltage = DeltaDistribution(mean=np.array([end_voltage]))
    ukf_dual._step(
        applied_control,
        observed_voltage
    )

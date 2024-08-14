import numpy as np

from moirae.estimators.online.kalman.unscented import JointUnscentedKalmanFilter
from moirae.interface import run_online_estimate


def test_interface(simple_rint, timeseries_dataset):
    # Make a simple estimator
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    ukf = JointUnscentedKalmanFilter(
        model=ecm,
        initial_inputs=rint_inputs,
        initial_transients=rint_transient,
        initial_asoh=rint_asoh,
        initial_covariance=np.diag([0.1, 0.1, 0.01])
    )

    # Run then make sure it returns the proper data types
    state_mean, estimator = run_online_estimate(timeseries_dataset, ukf)
    assert state_mean.shape == (
        len(timeseries_dataset.raw_data),
        ukf.num_state_dimensions * 2 + ukf.num_output_dimensions * 2
    )
    assert estimator._u.get_mean()[0] == timeseries_dataset.raw_data['test_time'].max()

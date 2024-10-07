import h5py
from pytest import mark
import numpy as np

from moirae.estimators.online.joint import JointEstimator
from moirae.interface import run_online_estimate
from moirae.interface.hdf5 import HDF5Writer
from moirae.models.ecm import EquivalentCircuitModel

# Priors for the covariance matrix, taken from the JointUKF demo
cov_asoh_rint = np.diag([
    2.5e-05
])
cov_trans_rint = np.diag([
    1. / 12,  # R0
    4. / 12,  # Hysteresis
])
voltage_err = 1.0e-03  # mV voltage error
noise_sensor = ((voltage_err / 2) ** 2) * np.eye(1)
noise_asoh = np.diag([1e-10])  # Small covariance on the R0
noise_tran = 1.0e-08 * np.eye(2)


def make_joint_ukf(init_asoh, init_transients, init_inputs):
    return JointEstimator.initialize_unscented_kalman_filter(
        cell_model=EquivalentCircuitModel(),
        initial_asoh=init_asoh,
        initial_transients=init_transients,
        initial_inputs=init_inputs,
        covariance_transient=cov_trans_rint,
        covariance_asoh=cov_asoh_rint,
        transient_covariance_process_noise=noise_tran,
        asoh_covariance_process_noise=noise_asoh,
        covariance_sensor_noise=noise_sensor
    )


@mark.parametrize('estimator', [make_joint_ukf])
def test_interface(simple_rint, timeseries_dataset, estimator):
    # Make a simple estimator
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    estimator = estimator(rint_asoh, rint_transient, rint_inputs)

    # Run then make sure it returns the proper data types
    state_mean, estimator = run_online_estimate(timeseries_dataset, estimator)
    assert state_mean.shape == (
        len(timeseries_dataset.raw_data),
        estimator.num_state_dimensions * 2 + estimator.num_output_dimensions * 2
    )

    # TODO (wardlt): Would be nice to have a check that the SOC, at least, was determined well


def test_hdf5_writer(simple_rint, tmpdir):
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    estimator = make_joint_ukf(rint_asoh, rint_transient, rint_inputs)

    h5_path = tmpdir / 'example.h5'
    with HDF5Writer(h5_path) as writer:
        assert writer.is_ready
        writer.prepare(estimator)

    with h5py.File(h5_path) as f:
        assert 'state_estimates' in f

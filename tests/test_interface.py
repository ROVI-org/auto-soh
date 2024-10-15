from pathlib import Path

import h5py
from pytest import mark, raises
import numpy as np

from moirae.estimators.online.filters.distributions import MultivariateGaussian
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


def test_hdf5_writer_init(simple_rint, tmpdir):
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    estimator = make_joint_ukf(rint_asoh, rint_transient, rint_inputs)

    # Test with a resizable dataset
    h5_path = Path(tmpdir / 'example.h5')
    with HDF5Writer(hdf5_output=h5_path) as writer:
        assert writer.is_ready
        writer.prepare(estimator)

    assert not writer.is_ready
    with raises(ValueError):
        writer.prepare(estimator)

    with h5py.File(h5_path) as f:
        assert 'state_estimates' in f
        group = f.get('state_estimates')
        assert 'per_step' in group
        assert all(x in group.attrs for x in ['write_settings', 'estimator_name'])

    # Test with a fixed size
    h5_path.unlink()
    with HDF5Writer(hdf5_output=h5_path, resizable=False, per_cycle='full', per_timestep='mean') as writer:
        assert writer.is_ready
        with raises(ValueError):
            writer.prepare(estimator)
        writer.prepare(estimator, 128, 4)

    with h5py.File(h5_path) as f:
        assert 'state_estimates' in f
        group = f.get('state_estimates').get('per_step')
        assert group['state_mean'].shape == (128, 3)
        assert 'covariance' not in group
        assert group['time'].shape == (128,)

        group = f.get('state_estimates').get('per_cycle')
        assert group['state_mean'].shape == (4, 3)
        assert group['state_covariance'].shape == (4, 3, 3)
        assert group['time'].shape == (4,)


@mark.parametrize('what,expected_keys', [
    ('full', ('state_mean', 'state_covariance', 'output_mean', 'output_covariance')),
    ('mean_cov', ('state_mean', 'state_covariance', 'output_mean', 'output_covariance')),
    ('mean_var', ('state_mean', 'state_variance', 'output_mean', 'output_variance')),
    ('mean', ('state_mean', 'output_mean')),
    ('none', ())
])
def test_hdf5_write(simple_rint, tmpdir, what, expected_keys):
    # Prepare an HDF5 file for writing
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    estimator = make_joint_ukf(rint_asoh, rint_transient, rint_inputs)
    h5_path = Path(tmpdir / 'example.h5')
    example_output = MultivariateGaussian(mean=np.array([0.]), covariance=np.array([[1.]]))
    with HDF5Writer(hdf5_output=h5_path, per_timestep=what) as writer:
        assert writer.is_ready
        writer.prepare(estimator)

        # Write two states to the file
        writer.append_step(0., 0, estimator.state, example_output)

        new_state = estimator.state.copy(deep=True)
        new_state.mean = estimator.state.get_mean() + 0.1
        writer.append_step(1., 0, new_state, example_output)

    # Make sure it's got the desired values
    with h5py.File(h5_path) as f:
        # Test the per-step quantities
        group = f.get('state_estimates')
        if what == 'none':
            assert 'per_step' not in group
        else:
            my_group = group.get('per_step')
            # Mean should only be set in the first two rows
            assert np.allclose(my_group['state_mean'][0, :], estimator.state.get_mean())
            assert np.allclose(my_group['state_mean'][1, :], new_state.get_mean())
            assert np.isnan(my_group['state_mean'][2:, :]).all()

            # Check the other keys
            assert set(my_group.keys()) == set(expected_keys + ('time',))

            # Check the shapes of the variance
            if 'state_variance' in my_group:
                assert my_group['state_variance'].shape[1:] == (3,)
                assert my_group['output_variance'].shape[1:] == (1,)

        # Make sure per_cycle was unaffected, and it only recorded the first state
        my_group = group.get('per_cycle')

        assert np.allclose(my_group['state_mean'][0, :], estimator.state.get_mean())
        assert np.allclose(my_group['state_covariance'][0, :], estimator.state.get_covariance())
        assert np.allclose(my_group['time'][0], 0)

        assert np.isnan(my_group['state_mean'][1:, :]).all()


@mark.parametrize('mode', ('path', 'prefab'))
def test_interface_write(mode, simple_rint, tmpdir, timeseries_dataset):
    # Make a simple estimator
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    estimator = make_joint_ukf(rint_asoh, rint_transient, rint_inputs)

    # Prepare the input argument
    h5_path = Path(tmpdir) / 'states.hdf5'
    if mode == 'path':
        h5_output = h5_path
    else:
        h5_output = HDF5Writer(hdf5_output=h5_path, resizable=False, per_cycle='full')

    # Run the estimation
    _, estimator = run_online_estimate(timeseries_dataset, estimator, hdf5_output=h5_output)

    with h5py.File(h5_path) as f:
        assert 'state_estimates' in f
        group = f['state_estimates']

        # Test that steps only include the mean
        per_step = group['per_step']
        assert set(per_step.keys()) == {'time', 'state_mean', 'output_mean'}

        # Test that cycles includes the full version
        per_cycle = group['per_cycle']
        assert set(per_cycle.keys()) == {'time', 'state_mean', 'state_covariance',
                                         'output_mean', 'output_covariance'}

        # Ensure the shapes vary depending on prefab or path mode
        assert per_step['time'].shape == (len(timeseries_dataset.raw_data),)
        if mode == 'prefab':
            assert per_step['time'].maxshape == (len(timeseries_dataset.raw_data),)

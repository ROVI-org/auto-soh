from pathlib import Path

from pytest import mark, raises
import tables as tb
import numpy as np

from moirae.estimators.online.filters.distributions import MultivariateGaussian
from moirae.estimators.online.joint import JointEstimator
from moirae.interface import run_online_estimate, run_model
from moirae.interface.hdf5 import HDF5Writer, read_state_estimates
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


def test_interface_stream(simple_rint, timeseries_dataset, tmpdir):
    # Make a simple estimator
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    estimator = make_joint_ukf(rint_asoh, rint_transient, rint_inputs)

    # Run then make sure it returns the proper data types
    h5_path = str(Path(tmpdir) / 'example.h5')
    timeseries_dataset.to_hdf(h5_path)
    state_mean, estimator = run_online_estimate(h5_path, estimator, hdf5_output=Path(tmpdir) / 'estimates.h5')
    assert state_mean.shape[0] == len(timeseries_dataset.raw_data)


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

    with tb.open_file(h5_path) as f:
        assert 'state_estimates' in f.root
        group = f.root['/state_estimates']
        assert 'per_timestep' in group
        assert all(x in group._v_attrs for x in ['write_settings', 'estimator_name'])

        dtype = f.get_node('/state_estimates/per_timestep').dtype
        assert dtype['state_mean'].shape == (3,)
        assert 'covariance' not in dtype.fields
        assert dtype['time'].shape == ()

        dtype = f.get_node('/state_estimates/per_cycle').dtype
        assert dtype['state_mean'].shape == (3,)
        assert dtype['state_covariance'].shape == (3, 3)
        assert dtype['time'].shape == ()


def _make_simple_hf_estimates(simple_rint, what, tmpdir):
    """Write two states into a file

    Args:
        simple_rint: Package describing the model
        what: Write mode for the per-step quantities
        tmpdir: Directory in which to write
    Returns:
        - Path to the h5 file
        - State at timestep 0
        - State at timestep 1
    """
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
    return h5_path, estimator.state, new_state


@mark.parametrize('what,expected_keys', [
    ('full', ('state_mean', 'state_covariance', 'output_mean', 'output_covariance')),
    ('mean_cov', ('state_mean', 'state_covariance', 'output_mean', 'output_covariance')),
    ('mean_var', ('state_mean', 'state_variance', 'output_mean', 'output_variance')),
    ('mean', ('state_mean', 'output_mean')),
    ('none', ())
])
def test_hdf5_write(simple_rint, tmpdir, what, expected_keys):
    h5_path, state_0, state_1 = _make_simple_hf_estimates(simple_rint, what, tmpdir)

    # Make sure it's got the desired values
    with tb.open_file(h5_path) as f:
        # Test the per-step quantities
        group = f.root['/state_estimates']
        if what == 'none':
            assert 'per_timestep' not in group
        else:
            my_table = group['per_timestep']

            # Mean should only be set in the first two rows
            assert my_table.shape[0] == 2
            assert np.allclose(my_table[0]['state_mean'], state_0.get_mean())
            assert np.allclose(my_table[1]['state_mean'], state_1.get_mean())

            # Check the other keys
            dtype = my_table.dtype
            assert set(dtype.fields) == set(expected_keys + ('time',))

            # Check the shapes of the variance
            if 'state_variance' in dtype.fields:
                assert dtype['state_variance'].shape == (3,)
                assert dtype['output_variance'].shape == (1,)

        # Make sure per_cycle was unaffected, and it only recorded the first state
        my_table = group['per_cycle']

        assert my_table.shape[0] == 1
        assert np.allclose(my_table[0]['state_mean'], state_0.get_mean())
        assert np.allclose(my_table[0]['state_covariance'], state_0.get_covariance())
        assert np.allclose(my_table[0]['time'], 0)


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
        h5_output = HDF5Writer(hdf5_output=h5_path, per_cycle='full')

    # Run the estimation
    _, estimator = run_online_estimate(timeseries_dataset, estimator, hdf5_output=h5_output)

    with tb.open_file(h5_path) as f:
        assert 'state_estimates' in f.root
        group = f.root['state_estimates']

        # Test that steps only include the mean
        per_timestep = group['per_timestep']
        assert set(per_timestep.dtype.fields) == {'time', 'state_mean', 'output_mean'}

        # Test that cycles includes the full version
        per_cycle = group['per_cycle']
        assert set(per_cycle.dtype.fields) == {'time', 'cycle', 'state_mean', 'state_covariance',
                                               'output_mean', 'output_covariance'}

        # Ensure the shape is equal to the data size
        assert per_timestep.shape == (len(timeseries_dataset.tables['raw_data']) - 1,)
        assert per_cycle[:]['output_covariance'].shape == \
               (timeseries_dataset.tables['raw_data']['cycle_number'].max() + 1, 1, 1)


@mark.parametrize('what', ('full', 'mean_cov', 'mean_var', 'mean', 'none'))
def test_h5_read_what(simple_rint, tmpdir, what):
    """Test reading the state estimates from an HDF5 file"""
    h5_path, state_0, state_1 = _make_simple_hf_estimates(simple_rint, what, tmpdir)
    dist_iter = read_state_estimates(h5_path, per_timestep=True)

    # Special case: Nothing is written
    if what == 'none':
        with raises(ValueError, match='No data'):
            next(dist_iter)
        return

    # Other cases
    time, state_dist, _ = next(dist_iter)
    assert np.isclose(time, 0.)
    assert np.allclose(state_0.get_mean(), state_dist.get_mean())

    _, state_dist, _ = next(dist_iter)
    assert np.allclose(state_1.get_mean(), state_dist.get_mean())
    if what != 'mean':
        assert np.allclose(state_1.get_covariance(), state_dist.get_covariance())

    # Make sure it iterates no further data
    with raises(StopIteration):
        next(dist_iter)


def test_h5_open_from_group(simple_rint, tmpdir):
    """Make sure we can read from an already-open file"""
    h5_path, _, _ = _make_simple_hf_estimates(simple_rint, 'none', tmpdir)

    with tb.open_file(h5_path) as f:
        dist_iter = read_state_estimates(f.root['state_estimates'], per_timestep=False)
        time, _, _ = next(dist_iter)
        assert np.isclose(time, 0.)


def test_run_model(simple_rint, timeseries_dataset):
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint
    measurements = run_model(
        model=ecm,
        dataset=timeseries_dataset,
        asoh=rint_asoh,
        state_0=rint_transient
    )
    assert 'terminal_voltage' in measurements.columns
    assert len(measurements) == len(timeseries_dataset.tables['raw_data'])

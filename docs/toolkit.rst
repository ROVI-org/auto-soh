Interface to Data Sources
=========================

Moirae provides the `interface module <./source/interface.html>`_ to faciliate reading performance data from
data files produced using the `battery data toolkit <https://github.com/ROVI-org/battery-data-toolkit>`_
and writing state estimates to disk in HDF5 format.

Running Estimators on `BatteryDataset` Objects
----------------------------------------------

The :meth:`~moirae.interface.run_online_estimate` object runs an online estimator using the performance
data stored in the pandas-based, :class:`~batdata.data.BatteryDataset`.

.. code-block::

    # Start by making an estimator (see other parts of documentation)
    estimator = JointEstimator.initialize_unscented_kalman_filter(
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

    # Read data from disk, run using all data points
    dataset = BatteryDataset.from_batdata_hdf('cell.h5')
    states, estimator = run_online_estimate(dataset, estimator)

The estimation produces a summary of the states for each timestep
and the estimator after running across all data.

Access more state estimates by supplying a path at which to write an HDF5 file as the
``hdf5_output`` argument to ``run_online_estimate``.
The interface will save the mean of the estimate for the state and outputs at each timestep,
and the full probability distribution of the estimated state and outputs for the first timestep of each cycle.
Provide an :class:`~moirae.interface.hdf5.HDF5Writer` (described below) to ``hdf5_output``
to control which information is written to HDF5.

.. note:: We may change to storing the average over a cycle instead of only the first time step.

Writing Estimates to HDF5 Files
-------------------------------

HDF5 files store collections of multi-dimensional arrays in a high-performance binary format.
We store state estimates to disk using the :class:`~moirae.interface.hdf5.HDF5Writer` class.

Create an ``HDF5Writer`` by providing the details of where to write the file (e.g., path to a file),
how to write the data (e.g., with compression),
and which data to write (e.g., full state vs only mean).

Prepare to write states by opening the file using Python's ``with`` and then, if writing estimates for the first time,
calling the :meth:`~moirae.interface.hdf5.HDF5Writer.prepare` method providing an initial estimator.

.. code-block:: python

    writer = HDFWriter('states.hdf5', per_timestep='full')
    with writer:
        writer.prepare(estimator)

The ``prepare`` option records details about the estimator, such as its name and how the cell physics is being modeled,
then creates the `"Datasets" <https://docs.h5py.org/en/stable/high/dataset.html>`_ that will store the time and values of each state.

Write states to the file incrementally by calling :meth:`~moirae.interface.hdf5.HDF5Writer.append_step`.

.. code-block:: python

    with writer:
        writer.append_step(step=0, time=1., cycle=0, state=estimator.state, output=output)

The resultant data may not be available in the output HDF5 file until after the ``with`` block.

.. warning::

    Moirae currently supports writing to an HDF5 file only once.
    The states cannot be edited after exiting the ``with`` block.

Reading Estimates from HDF5
---------------------------

Moirae writes the states estimated by an online estimator to different `"Groups" <https://docs.h5py.org/en/stable/high/group.html>`_
within an HDF5 file.

Metadata about the online estimation process and the estimates, whether per timestep or cycle, are all stored in a group
named ``state_estimates``.
Metadata are listed as attributes.

.. code-block:: python

    with h5py.File('states.hdf5') as f:
        assert 'state_estimates' in f
        group = f['state_estimates']
        print(f'Estimates were performed by a {group.attrs["estimator_name"]}'
              f' with physics described by a {group.attrs["cell_model"]}')

The attributes stored by Moirae include:

- ``write_settings``: The settings used by the ``HDF5Writer``
- ``state_names``: Names of the states in the order provided in estimates
- ``output_names``: Names of the outputs in the order provided by the estimator
- ``estimator_name``: The name of the `estimator framework <estimators/index.html#online-estimators>`_ employed
- ``distribution_type``: The type of `probability distribution <source/online.html#module-moirae.estimators.online.filters.distributions>`_ used by the estimator
- ``cell_model``: Name of the `model used to describe cell behavior <system-models.html#defining-the-cell-physics>`_
- ``initial_asoh``: Initial estimate of the cell health parameters
- ``initial_transient_state``: Initial estimate of the cell transient state

The values of the estimates at each timestep and the first step in each cycle are stored in ``per_timestep`` and ``per_cycle`` subgroups, respectively.
The information in each varies depending on the choice of what to write.

.. code-block:: python

    with h5py.File('states.hdf5') as f:
        # Access the mean of all states for the first step of the first cycle
        per_cycle = group.get('per_cycle')
        per_cycle['mean'][0, :]

        # Access the standard deviation of the first state for all time steps
        per_cycle = group.get('per_timestep')
        per_cycle['covariance'][:, 0, 0]

Moirae provides a utility function, :meth:`~moirae.interface.hdf5.read_state_estimates`, to read the
distributions from the file sequentially as :class:`~moirae.estimators.online.filters.distributions.MultivariateRandomDistribution`.

.. code-block:: python

    for time, state_dist, output_dist in read_state_estimates('states.hdf5', per_timestep=True):
        continue  # Distributions are read into memory in batches as an iterator

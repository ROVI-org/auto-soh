"""Interfaces for running common workflows with Moirae,
with a particular emphasis on data built with
`battery-data-toolkit <https://github.com/ROVI-org/battery-data-toolkit>`_"""
from contextlib import nullcontext
from typing import Tuple, Union
from math import isfinite
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from battdat.data import BatteryDataset
from battdat.streaming import iterate_records_from_file

from moirae.estimators.online import OnlineEstimator
from moirae.interface.hdf5 import HDF5Writer
from moirae.models.base import InputQuantities, OutputQuantities
from moirae.models.ecm import ECMInput, ECMMeasurement

__all__ = ['row_to_inputs', 'run_online_estimate']


def row_to_inputs(row: pd.Series, default_temperature: float = 25) -> Tuple[InputQuantities, OutputQuantities]:
    """Convert a row from the time series data to a distribution object

    Args:
        row: Row from the `dataset.raw_data` dataframe
        default_temperature: Default temperature for the cells (units: C)
    Returns:
        - Distribution describing the inputs
        - Distribution describing the measurements (model outputs)
    """

    # First to an "inputs" class, which stores the proper order
    use_temp = 'temperature' in row and isfinite(row['temperature'])
    # TODO (wardlt): Remove hard code from ECM when we're ready (maybe a "from_batdata" to the model class?)
    inputs = ECMInput(
        time=row['test_time'],
        current=row['current'],
        temperature=row['temperature'] if use_temp else default_temperature
    )
    outputs = ECMMeasurement(
        terminal_voltage=row['voltage']
    )

    return inputs, outputs


# TODO (wardlt): Create generic "Writer" classes which can store data in other formats (e.g., streaming to DataHub)
def run_online_estimate(
        dataset: Union[BatteryDataset, str, Path],
        estimator: OnlineEstimator,
        pbar: bool = False,
        output_states: bool = True,
        hdf5_output: Union[Path, str, h5py.Group, HDF5Writer, None] = None,
) -> Tuple[pd.DataFrame, OnlineEstimator]:
    """Run an online estimation of battery parameters given a fixed dataset for the

    Args:
        dataset: Dataset containing the time series of a battery's performance.
            Provide either the path to a ``battdata`` HDF5 file or :class:`~batdata.data.BatteryDataset` object.
        estimator: Technique used to estimate the state of health, which is built using
            a physics model which describes the cell and initial guesses for the battery
            transient and health states.
        pbar: Whether to display a progress bar
        output_states: Whether to return summaries of the per-step states. Can require
            a large amount of memory for full datasets.
        hdf5_output: Path to an HDF5 file or group within an already-open file in which to
            write the estimated parameter values. Writes the mean for each timestep and
            the full state for the first timestep in each cycle by default. Modify what is written
            by providing a :class:`~moirae.interface.hdf5.HDF5Writer`.
    Returns:
        - Estimates of the parameters at all timesteps from the input dataset
        - Estimator after updating with the data in dataset
    """

    # Determine the number of rows in the dataset and an interator over the dataset
    if isinstance(dataset, BatteryDataset):
        # Ensure raw data are present in the data file
        if dataset.raw_data is None:
            raise ValueError('No time series data in the provided dataset')

        num_rows = dataset.raw_data.shape[0]
        num_cycles = dataset.raw_data['cycle_number'].max() + 1 if 'cycle_number' in dataset.raw_data else 0

        def _row_iter(d):
            for _, r in d.iterrows():
                yield r

        row_iter = _row_iter(dataset.raw_data.reset_index())  # .reset_index to iterate in sort order
    elif isinstance(dataset, (str, Path)):
        with pd.HDFStore(dataset, mode='r') as store:
            num_rows = store.get_storer('raw_data').nrows
        row_iter = iterate_records_from_file(dataset)
        num_cycles = None  # Cannot know this w/o reading
    else:
        raise ValueError(f'Unrecognized data type: {type(dataset)}')

    # Update the initial inputs for the
    initial_input, _ = row_to_inputs(next(row_iter))
    estimator._u = initial_input

    # Initialize the output arrays
    if output_states:
        state_mean = np.zeros((num_rows, estimator.num_state_dimensions))
        state_std = np.zeros((num_rows, estimator.num_state_dimensions))
        output_mean = np.zeros((num_rows, estimator.num_output_dimensions))
        output_std = np.zeros((num_rows, estimator.num_output_dimensions))

    # Open a H5 output if desired
    if isinstance(hdf5_output, (str, Path, h5py.Group)):
        h5_writer = HDF5Writer(hdf5_output=hdf5_output)
    elif hdf5_output is not None:
        h5_writer = hdf5_output
    else:
        h5_writer = nullcontext()

    # Iterate over all timesteps
    with h5_writer:
        # Prepare given the available data
        if hdf5_output is not None:
            h5_writer.prepare(estimator=estimator, expected_steps=num_rows, expected_cycles=num_cycles)

        for i, row in tqdm(
                enumerate(row_iter), total=num_rows, disable=not pbar,
        ):
            controls, measurements = row_to_inputs(row)
            new_state, new_outputs = estimator.step(controls, measurements)

            if output_states:
                state_mean[i, :] = new_state.get_mean()
                state_std[i, :] = np.diag(new_state.get_covariance())
                output_mean[i, :] = new_outputs.get_mean()
                output_std[i, :] = np.diag(new_outputs.get_covariance())

            # Store estimates
            if hdf5_output is not None:
                h5_writer.append_step(row['test_time'], row['cycle_number'], new_state, new_outputs)

    # Compile the outputs into a dataframe
    output = None
    if output_states:
        output = pd.DataFrame(
            np.concatenate([state_mean, state_std, output_mean, output_std], axis=1),
            columns=list(
                estimator.state_names + tuple(f'{s}_std' for s in estimator.state_names)
                + estimator.output_names + tuple(f'{s}_std' for s in estimator.output_names)
            )
        )
    return output, estimator

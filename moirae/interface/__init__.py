"""Interfaces for running common workflows with Moirae,
with a particular emphasis on data built with
`battery-data-toolkit <https://github.com/ROVI-org/battery-data-toolkit>`_"""
from typing import Tuple, Union, Literal
from math import isfinite
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from batdata.data import BatteryDataset

from moirae.estimators.online import OnlineEstimator
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
        current=-row['current'],  # Opposite sign convention from batdata
        temperature=row['temperature'] if use_temp else default_temperature
    )
    outputs = ECMMeasurement(
        terminal_voltage=row['voltage']
    )

    return inputs, outputs


def run_online_estimate(
        dataset: BatteryDataset,
        estimator: OnlineEstimator,
        pbar: bool = False,
        hdf5_output: Union[Path, str, h5py.Group, None] = None,
) -> Tuple[pd.DataFrame, OnlineEstimator]:
    """Run an online estimation of battery parameters given a fixed dataset for the

    Args:
        dataset: Dataset containing the time series of a battery's performance
        estimator: Technique used to estimate the state of health, which is built using
            a physics model which describes the cell and initial guesses for the battery
            transient and health states.
        pbar: Whether to display a progress bar
        hdf5_output: Path to an HDF5 file or group within an already-open file in which to
            write the estimated parameter values
    Returns:
        - Estimates of the parameters at all timesteps from the input dataset
        - Estimator after updating with the data in dataset
    """

    # Ensure raw data are present in the data file
    if dataset.raw_data is None:
        raise ValueError('No time series data in the provided dataset')

    # Update the initial inputs for the
    initial_input, _ = row_to_inputs(dataset.raw_data.iloc[0])
    estimator._u = initial_input

    # Initialize the output arrays
    state_mean = np.zeros((len(dataset.raw_data), estimator.num_state_dimensions))
    state_std = np.zeros((len(dataset.raw_data), estimator.num_state_dimensions))
    output_mean = np.zeros((len(dataset.raw_data), estimator.num_output_dimensions))
    output_std = np.zeros((len(dataset.raw_data), estimator.num_output_dimensions))

    # Open a H5 output if desired
    h5_handle = None
    if isinstance(hdf5_output, (str, Path)):
        h5_handle = h5py.File(hdf5_output)

    # Iterate over all timesteps
    try:
        for i, (_, row) in tqdm(
                enumerate(dataset.raw_data.reset_index().iterrows()), total=len(dataset.raw_data), disable=not pbar,
        ):  # .reset_index to iterate in sort order
            controls, measurements = row_to_inputs(row)
            new_state, new_outputs = estimator.step(controls, measurements)

            state_mean[i, :] = new_state.get_mean()
            state_std[i, :] = np.diag(new_state.get_covariance())
            output_mean[i, :] = new_outputs.get_mean()
            output_std[i, :] = np.diag(new_outputs.get_covariance())

            # Store estimates
    finally:
        # Close the HDF5 file if we opened one
        if h5_handle is not None:
            h5_handle.close()

    # Compile the outputs into a dataframe
    output = pd.DataFrame(
        np.concatenate([state_mean, state_std, output_mean, output_std], axis=1),
        columns=list(
            estimator.state_names + tuple(f'{s}_std' for s in estimator.state_names)
            + estimator.output_names + tuple(f'{s}_std' for s in estimator.output_names)
        )
    )
    return output, estimator

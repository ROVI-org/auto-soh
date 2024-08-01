"""Interfaces for running common workflows with Auto-SOH"""
from typing import Tuple
from math import isfinite

import numpy as np
import pandas as pd
from batdata.data import BatteryDataset

from moirae.estimators.online import OnlineEstimator, DeltaDistribution
from moirae.models.ecm import ECMInput, ECMMeasurement


def _row_to_inputs(row: pd.Series, default_temperature: float = 25) -> Tuple[DeltaDistribution, DeltaDistribution]:
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

    return (
        DeltaDistribution(mean=inputs.to_numpy()[0, :]),
        DeltaDistribution(mean=outputs.to_numpy()[0, :]),
    )


def run_online_estimate(
        dataset: BatteryDataset,
        estimator: OnlineEstimator
) -> Tuple[pd.DataFrame, OnlineEstimator]:
    """Run an online estimation of battery parameters given a fixed dataset for the

    Args:
        dataset: Dataset containing the time series of a battery's performance
        estimator: Technique used to estimate the state of health, which is built using
            a physics model which describes the cell and initial guesses for the battery
            transient and health states.
    Returns:
        - Estimates of the parameters at all timesteps from the input dataset
        - Estimator after updating with the data in dataset
    """

    # Ensure raw data are present in the data file
    if dataset.raw_data is None:
        raise ValueError('No time series data in the provided dataset')

    # Update the initial inputs for the
    initial_input, _ = _row_to_inputs(dataset.raw_data.iloc[0])
    estimator.u = initial_input

    # Initialize the output arrays
    state_mean = np.zeros((len(dataset.raw_data), estimator.num_hidden_dimensions))
    state_std = np.zeros((len(dataset.raw_data), estimator.num_hidden_dimensions))
    output_mean = np.zeros((len(dataset.raw_data), estimator.num_output_dimensions))
    output_std = np.zeros((len(dataset.raw_data), estimator.num_output_dimensions))

    # Iterate over all timesteps
    for i, (_, row) in enumerate(dataset.raw_data.reset_index().iterrows()):  # .reset_index to iterate in sort order
        controls, measurements = _row_to_inputs(row)
        new_outputs, new_state = estimator.step(controls, measurements)

        state_mean[i, :] = new_state.get_mean()
        state_std[i, :] = np.diag(new_state.get_covariance())
        output_mean[i, :] = new_outputs.get_mean()
        output_std[i, :] = np.diag(new_outputs.get_covariance())

    # Compile the outputs into a dataframe
    output = pd.DataFrame(
        np.concatenate([state_mean, state_std, output_mean, output_std], axis=1),
        columns=list(
            estimator.state_names + tuple(f'{s}_std' for s in estimator.state_names)
            + estimator.output_names + tuple(f'{s}_std' for s in estimator.output_names)
        )
    )
    return output, estimator

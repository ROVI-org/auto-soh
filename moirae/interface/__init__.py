"""Interfaces for running common workflows with Moirae,
with a particular emphasis on data built with
`battery-data-toolkit <https://github.com/ROVI-org/battery-data-toolkit>`_"""
from contextlib import nullcontext
from typing import Tuple, Union
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from tables import Group
from battdat.data import BatteryDataset
from battdat.streaming import iterate_records_from_file

from moirae.estimators.online import OnlineEstimator
from moirae.interface.hdf5 import HDF5Writer
from moirae.models.base import InputQuantities, OutputQuantities, CellModel, HealthVariable, GeneralContainer
from moirae.models.ecm import ECMInput, ECMMeasurement
from moirae.simulator import Simulator

__all__ = ['row_to_inputs', 'run_online_estimate', 'run_model']


def row_to_inputs(row: pd.Series,
                  default_temperature: float = 25,
                  input_class: type[InputQuantities] = ECMInput,
                  output_class: type[OutputQuantities] = ECMMeasurement) -> Tuple[InputQuantities, OutputQuantities]:
    """Convert a row from the time series data to a distribution object

    Args:
        row: Row from the `dataset.raw_data` dataframe
        default_temperature: Default temperature for the cells (units: C)
        input_class: Class to use to store inputs
        output_class: Class to use to store outputs
    Returns:
        - Distribution describing the inputs
        - Distribution describing the measurements (model outputs)
    """

    # First to an "inputs" class, which stores the proper order
    use_temp = 'temperature' in row and isfinite(row['temperature'])
    # TODO (wardlt): Allow ability to override mapping between battdat "raw_data" columns and variable names
    inputs = input_class(
        time=row['test_time'],
        current=row['current'],
        temperature=row['temperature'] if use_temp else default_temperature
    )
    outputs = output_class(
        terminal_voltage=row['voltage']
    )

    return inputs, outputs


# TODO (wardlt): Create generic "Writer" classes which can store data in other formats (e.g., streaming to DataHub)
def run_online_estimate(
        dataset: Union[BatteryDataset, str, Path],
        estimator: OnlineEstimator,
        pbar: bool = False,
        output_states: bool = True,
        hdf5_output: Union[Path, str, Group, HDF5Writer, None] = None,
        inout_types: tuple[type[InputQuantities], type[OutputQuantities]] = (ECMInput, ECMMeasurement)
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
        inout_types: Types used to represent input and measurement data.
            Uses those for the Moirae ECM by default
    Returns:
        - Estimates of the parameters at all timesteps from the input dataset
        - Estimator after updating with the data in dataset
    """

    # Determine the number of rows in the dataset and an interator over the dataset
    if isinstance(dataset, BatteryDataset):
        # Ensure raw data are present in the data file
        if 'raw_data' not in dataset.tables is None:
            raise ValueError('No time series data in the provided dataset')

        raw_data = dataset.tables['raw_data']
        num_rows = raw_data.shape[0]
        num_cycles = raw_data['cycle_number'].max() + 1 if 'cycle_number' in raw_data else 0

        def _row_iter(d):
            for _, r in d.iterrows():
                yield r

        row_iter = _row_iter(raw_data.reset_index())  # .reset_index to iterate in sort order
    elif isinstance(dataset, (str, Path)):
        with pd.HDFStore(dataset, mode='r') as store:
            num_rows = store.get_storer('raw_data').nrows
        row_iter = iterate_records_from_file(dataset)
        num_cycles = None  # Cannot know this w/o reading
    else:
        raise ValueError(f'Unrecognized data type: {type(dataset)}')

    # Initialize the output arrays
    if output_states:
        # num_rows - 1 because we do not record the first step
        state_mean = np.zeros((num_rows - 1, estimator.num_state_dimensions))
        state_std = np.zeros((num_rows - 1, estimator.num_state_dimensions))
        output_mean = np.zeros((num_rows - 1, estimator.num_output_dimensions))
        output_std = np.zeros((num_rows - 1, estimator.num_output_dimensions))

    # Open a H5 output if desired
    if isinstance(hdf5_output, (str, Path, Group)):
        h5_writer = HDF5Writer(hdf5_output=hdf5_output)
    elif hdf5_output is not None:
        h5_writer = hdf5_output
    else:
        h5_writer = nullcontext()

    # Iterate over all timesteps
    in_type, out_type = inout_types
    with h5_writer:
        # Prepare given the available data
        if hdf5_output is not None:
            h5_writer.prepare(estimator=estimator, expected_steps=num_rows, expected_cycles=num_cycles)

        # Update the inputs using the first step
        initial_input, _ = row_to_inputs(next(row_iter))
        estimator._u = initial_input

        for i, row in tqdm(
                enumerate(row_iter), total=num_rows, disable=not pbar,
        ):
            controls, measurements = row_to_inputs(row, input_class=in_type, output_class=out_type)
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


def run_model(
        model: CellModel,
        dataset: BatteryDataset,
        asoh: HealthVariable,
        state_0: GeneralContainer,
        inout_types: tuple[type[InputQuantities], type[OutputQuantities]] = (ECMInput, ECMMeasurement)
) -> pd.DataFrame:
    """Run a cell model following data provided by a :class:`~battdat.data.BatteryDataset`

    Args:
        model: Model describing the cell
        dataset: Observations of current and voltage used when running th emodel
        asoh: Parameters for the cell
        state_0: Starting state for the model
        inout_types: Types used to represent input and measurement data.
            Uses those for the Moirae ECM by default
    Returns:
        Outputs from the model as a function of time
    """

    # Init the simulation runner
    raw_data = dataset.tables['raw_data']
    in_class, out_class = inout_types
    simulator = Simulator(
        cell_model=model,
        asoh=asoh,
        transient_state=state_0,
        initial_input=row_to_inputs(raw_data.iloc[0], input_class=in_class, output_class=out_class)[0],
        keep_history=False
    )

    # Init the output array
    init_outputs = simulator.measurement
    output = np.zeros((len(raw_data), len(init_outputs)))
    output[0, :] = init_outputs.to_numpy()[0, :]

    for i, (_, row) in enumerate(raw_data.iloc[1:].iterrows()):
        inputs, _ = row_to_inputs(row, input_class=in_class, output_class=out_class)
        _, outputs = simulator.step(inputs)
        output[i + 1, :] = outputs.to_numpy()[0, :]
    return pd.DataFrame(output, columns=init_outputs.all_names)

"""Utility for running physics models for large numbers of steps"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from batdata.data import BatteryDataset
from moirae.models.base import (HealthVariable,
                                GeneralContainer,
                                InputQuantities,
                                OutputQuantities,
                                CellModel,
                                DegradationModel)
from moirae.models.utils import DummyDegradation


class Simulator:
    """
    Run a :class:`~moirae.models.base.CellModel` and track results

    The current states of a batch of systems are stored as attributes of the class,
    such as :attr:`transient` for the transient states.
    The history of the states and outputs are stored as lists,
    such as :attr:`transient_history`,
    if ``keep_history`` is True.

    Args:
        cell_model: Model used to simulate the battery system
        asoh: Advanced State of Health (A-SOH) of the system. Used to parametrize the dynamics of the system.
        transient_state: Initial transient state of the system
        initial_input: Initial input of the ECM
        degradation_model: Degradation model to be used for degrading the A-SOH; defaults to no degradation
        keep_history: Whether to keep history of the system.
    """

    transient_history: Optional[List[GeneralContainer]]
    """History of observed transient states"""
    input_history: Optional[List[InputQuantities]]
    """History of inputs into the system"""
    measurement_history: Optional[List[OutputQuantities]]
    """History of the outputs from the system"""

    measurement: OutputQuantities
    """Last measurement from the system"""
    asoh: HealthVariable
    """Health variables for each of the cells being simulated"""
    previous_input: InputQuantities
    """Last inputs to the system"""

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transient_state: GeneralContainer,
                 initial_input: InputQuantities,
                 degradation_model: DegradationModel = DummyDegradation(),
                 keep_history: bool = False):
        self.model = cell_model
        self.degradation_model = degradation_model

        # Store copies of the initial states
        self.asoh = asoh.model_copy(deep=True)
        self.transient = transient_state.model_copy(deep=True)
        self.previous_input = initial_input.model_copy(deep=True)

        # Get the initial measurement
        self.measurement = self.model.calculate_terminal_voltage(new_inputs=self.previous_input,
                                                                 transient_state=self.transient,
                                                                 asoh=self.asoh)

        # Initialize the storage arrays
        self.keep_history = keep_history
        if self.keep_history:
            self.input_history = [self.previous_input.model_copy(deep=True)]
            self.transient_history = [self.transient.model_copy(deep=True)]
            self.asoh_history = [self.asoh.model_copy(deep=True)]
            self.measurement_history = [self.measurement.model_copy(deep=True)]
        else:
            self.input_history = self.transient_history = self.measurement_history = None

    def step(self, new_inputs: InputQuantities) -> Tuple[GeneralContainer, OutputQuantities]:
        """
        Function to step the transient state of the system.

        Args:
            new_inputs: New ECM input to the system

        Returns:
            Tuple of the new transient state and corresponding measurement
        """
        # Get new transient
        new_transient = self.model.update_transient_state(new_inputs=new_inputs,
                                                          transient_state=self.transient,
                                                          asoh=self.asoh,
                                                          previous_inputs=self.previous_input)

        # Update internal
        self.transient = new_transient
        self.previous_input = new_inputs.model_copy(deep=True)

        # Get new measurement
        new_measurement = self.model.calculate_terminal_voltage(new_inputs=self.previous_input,
                                                                transient_state=self.transient,
                                                                asoh=self.asoh)

        # Update measurement
        self.measurement = new_measurement.model_copy(deep=True)

        # Degrade A-SOH
        new_asoh = self.degradation_model.update_asoh(previous_asoh=self.asoh,
                                                      new_inputs=new_inputs,
                                                      new_transients=new_transient,
                                                      new_measurements=new_measurement)
        self.asoh = new_asoh.model_copy(deep=True)

        if self.keep_history:
            self.input_history.append(self.previous_input)
            self.transient_history.append(new_transient)
            self.asoh_history.append(new_asoh)
            self.measurement_history.append(new_measurement)

        return new_transient, new_measurement

    def evolve(self, inputs: List[InputQuantities]) -> List[OutputQuantities]:
        """
        Evolves the simulator given a list of inputs.

        Args
            inputs: List of ECMInput objects

        Returns
            measurements: List of corresponding ECMMeasurements
        """

        measurements = []

        for new_input in inputs:
            _, measure = self.step(new_inputs=new_input)
            measurements.append(measure)

        return measurements

    def to_dataframe(self) -> pd.DataFrame:
        """
        Compile the history of the simulator as a Pandas dataframe

        Returns:
            Dataframe with the columns ordered by inputs, states, and outputs
        """

        if not self.keep_history:
            raise ValueError('History was not stored. Set keep_history=True')

        batch_size = self.measurement.batch_size

        def _squish(params: List[GeneralContainer]) -> pd.DataFrame:
            values = np.concatenate([
                x.to_numpy() if x.batch_size == batch_size else np.repeat(x.to_numpy(), batch_size, axis=0)
                for x in params], axis=0
            )
            return pd.DataFrame(values, columns=params[0].all_names)

        # Make a dataframe of each component of the history
        member = np.repeat(np.arange(batch_size), len(self.input_history))
        return pd.concat([
            pd.DataFrame({'batch': member}),
            _squish(self.input_history),
            _squish(self.transient_history),
            _squish(self.measurement_history)
        ], axis=1)

    def to_batdata(self, extra_columns: bool = False) -> List[BatteryDataset]:
        """
        Compile the cycling history as a Battery Data Toolkit dataset.

        Args:
            extra_columns: Whether to return columns whose names are not yet in the schema
        Returns:
            A battery dataset for each batch of the data
        """

        df = self.to_dataframe()

        # Rename key columns before storing as a battery dataset
        known_names = {
            'time': 'test_time',
            'terminal_voltage': 'voltage'
        }
        df.rename(columns=known_names, inplace=True)

        output = []
        for _, group in df.groupby('batch'):
            batch = BatteryDataset(raw_data=group.drop(columns=['batch']))
            batch.raw_data['current'] *= -1  # Moirae uses the opposite sign convention as batdata

            # Compile names for the other columns
            #  TODO (wardlt): I bet I can grab the description from the model fields.
            if extra_columns:
                batch.metadata.raw_data_columns.update(
                    (name, f'Input variable from {self.previous_input.__class__.__name__}')
                    for name in self.previous_input.all_names if name not in known_names
                )
                batch.metadata.raw_data_columns.update(
                    (name, f'Transient state variable from {self.transient.__class__.__name__}')
                    for name in self.transient.all_names if name not in known_names
                )
                batch.metadata.raw_data_columns.update(
                    (name, f'Measurement variable from {self.measurement.__class__.__name__}')
                    for name in self.measurement.all_names if name not in known_names
                )

            output.append(batch)
        return output

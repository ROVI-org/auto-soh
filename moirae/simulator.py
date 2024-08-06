from typing import Tuple, List, Optional

from moirae.models.base import HealthVariable, GeneralContainer, InputQuantities, CellModel, OutputQuantities


class Simulator:
    """
    Run a :class:`~moirae.models.base.CellModel` and track results

    The current states of a batch of systems are stored as attributes of the class,
    such as :attr:`transient` for the transient states.
    The history of the states and outputs are stored as a list of NumPy arrays,
    such as :class:`transient_history`.

    Args:
        asoh: Advanced State of Health (A-SOH) of the system. Used to parametrize the dynamics of the system.
        transient_state: Initial physical transient state of the ECM. If not provided, will be instantiated based on the
            A-SOH provided assuming with all values initialized to 0.0
        initial_input: Initial input of the ECM. If not provided, will be instantiated assuming the system starts at
            time = 0.0 seconds with a current of 0.0 Amps.
        current_behavior: Determines how to the total current behaves in-between time steps. Can be either 'constant' or
            'linear'.
        keep_history: Boolean to determine whether we wish to keep history of the system.
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
    measurement: OutputQuantities
    """Last measurement result"""

    def __init__(self,
                 model: CellModel,
                 asoh: HealthVariable,
                 transient_state: GeneralContainer,
                 initial_input: InputQuantities,
                 keep_history: bool = False):
        self.model = model

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

        if self.keep_history:
            self.input_history.append(self.previous_input)
            self.transient_history.append(new_transient)
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

from typing import Literal, Tuple, List

from moirae.models.ecm import EquivalentCircuitModel as ECM
from moirae.models.ecm import (ECMASOH,
                               ECMTransientVector,
                               ECMInput,
                               ECMMeasurement)


# TODO (wardlt): Make this capable of running with any CellModel
class ECMSimulator():
    """
    Class used to simulate and store information about an ECM

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

    def __init__(self,
                 asoh: ECMASOH,
                 transient_state: ECMTransientVector = None,
                 initial_input: ECMInput = None,
                 current_behavior: Literal['constant', 'linear'] = 'constant',
                 keep_history: bool = False,
                 ) -> None:
        has_C0 = asoh.c0 is not None
        num_RC = len(asoh.rc_elements)
        self.current_behavior = current_behavior
        self.asoh = asoh.model_copy(deep=True)
        self.keep_history = keep_history
        if transient_state is None:
            transient_state = ECMTransientVector.provide_template(has_C0=has_C0, num_RC=num_RC)
        self.transient = transient_state.model_copy(deep=True)
        if initial_input is None:
            initial_input = ECMInput(time=0., current=0.)
        self.previous_input = initial_input.model_copy(deep=True)
        self.measurement = ECM().calculate_terminal_voltage(new_inputs=self.previous_input,
                                                            transient_state=self.transient,
                                                            asoh=self.asoh)

        if self.keep_history:
            self.input_history = [self.previous_input.model_copy(deep=True)]
            self.transient_history = [self.transient.model_copy(deep=True)]
            self.measurement_history = [self.measurement.model_copy(deep=True)]

    def step(self, new_inputs: ECMInput) -> Tuple[ECMTransientVector, ECMMeasurement]:
        """
        Function to step the transient state of the system.

        Args:
            new_inputs: New ECM input to the system

        Returns:
            Tuple of the new transient state and corresponding measurement
        """
        # Get new transient
        new_transient = ECM(self.current_behavior).update_transient_state(new_inputs=new_inputs,
                                                                          transient_state=self.transient,
                                                                          asoh=self.asoh,
                                                                          previous_inputs=self.previous_input)

        # Update internal
        self.transient = new_transient.model_copy(deep=True)
        self.previous_input = new_inputs.model_copy(deep=True)

        # Get new measurement
        new_measurement = ECM(self.current_behavior).calculate_terminal_voltage(new_inputs=self.previous_input,
                                                                                transient_state=self.transient,
                                                                                asoh=self.asoh)

        # Update measurement
        self.measurement = new_measurement.model_copy(deep=True)

        if self.keep_history:
            self.input_history += [self.previous_input.model_copy(deep=True)]
            self.transient_history += [self.transient.model_copy(deep=True)]
            self.measurement_history += [self.measurement.model_copy(deep=True)]

        return new_transient, new_measurement

    def evolve(self, inputs: List[ECMInput]) -> List[ECMMeasurement]:
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

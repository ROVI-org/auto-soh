from typing import Literal, Tuple, List

from asoh.models.ecm import EquivalentCircuitModel as ECM
from asoh.models.ecm import (ECMASOH,
                             ECMTransientVector,
                             ECMInput,
                             ECMMeasurement)


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
        self.has_C0 = asoh.c0 is not None
        self.num_RC = len(asoh.rc_elements)
        self.current_behavior = current_behavior
        self.asoh = asoh.model_copy(deep=True)
        self.keep_history = keep_history
        # Lenght of hidden vector: SOC + q0 + I_RC_j + hysteresis
        self.len_hidden = int(1 + int(self.has_C0) + self.num_RC + 1)
        if transient_state is None:
            transient_state = ECMTransientVector.provide_template(has_C0=self.has_C0, num_RC=self.num_RC)
        self.transient = transient_state.model_copy(deep=True)
        if initial_input is None:
            initial_input = ECMInput(time=0., current=0.)
        self.previous_input = initial_input.model_copy(deep=True)
        self.measurement = ECM().calculate_terminal_voltage(new_input=self.previous_input,
                                                            transient_state=self.transient,
                                                            asoh=self.asoh)

        if self.keep_history:
            self.input_history = [self.previous_input]
            self.transient_history = [self.transient]
            self.measurement_history = [self.measurement]

    def step(self, new_input: ECMInput) -> Tuple[ECMTransientVector, ECMMeasurement]:
        """
        Function to step the transient state of the system.

        Args:
            new_input: New ECM input to the system

        Yields:
            Tuple of the new transient state and corresponding measurement
        """
        # Get new transient
        new_transient = ECM().update_transient_state(new_input=new_input,
                                                     transient_state=self.transient,
                                                     asoh=self.asoh,
                                                     previous_input=self.previous_input,
                                                     current_behavior=self.current_behavior)

        # Update internal
        self.transient = new_transient.model_copy(deep=True)
        self.previous_input = new_input.model_copy(deep=True)

        # Get new measurement
        new_measurement = ECM().calculate_terminal_voltage(new_input=self.previous_input,
                                                           transient_state=self.transient,
                                                           asoh=self.asoh)

        # Update measurement
        self.measurement = new_measurement.model_copy(deep=True)

        if self.keep_history:
            self.input_history += [self.previous_input]
            self.transient_history += [self.transient]
            self.measurement_history += [self.measurement]

        return new_transient, new_measurement

    def evolve(self, inputs: List[ECMInput]) -> List[ECMMeasurement]:
        """
        Evolves the simulator given a list of inputs.

        Args
            inputs: List of ECMInput objects

        Yields
            measurements: List of corresponding ECMMeasurements
        """

        measurements = []

        for new_input in inputs:
            _, measure = self.step(new_input=new_input)
            measurements.append(measure)

        return measurements

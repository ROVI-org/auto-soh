"""Tools to reduce operations on :class:`~moirae.models.baseCellModel` to functions which act only on
widely-used Python types, such as Numpy Arrays."""

import numpy as np
from typing import Tuple

from moirae.models.base import InputQuantities, GeneralContainer, HealthVariable, CellModel


class CellModelInterface:
    """Function to produce an updated estimate for hidden states based on a :class:`CellModel`

    Create the hidden update function by defining
        - Which portions of the transient state and ASOH are used as inputs to function
        - Values for the ASOH parameters that remain fixed
        - An example transient state and input to be passed to the function which will be used as a template

    The resultant function will take numpy arrays as inputs and produce numpy arrays as outputs

    Args:
        cell_model: Model which defines the physics of the system being modeled
        asoh: Values for all state of health parameters of the model
        transient_state: Current values of the transient state of the system
        input_template: Example input values for the model
        transient_inputs: Whether to include the transient state as inputs
        asoh_inputs: Names of the ASOH parameters to include as inputs
    """

    cell_model: CellModel
    """Cell model underpinning the update functions"""
    asoh: HealthVariable
    """ASOH values passed to each call of the cell model"""
    transients: GeneralContainer
    """Transient states used for the inputs of the model"""

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transient_state: GeneralContainer,
                 input_template: InputQuantities,
                 transient_inputs: bool,
                 asoh_inputs: Tuple[str]):
        self.model = cell_model

        # Store the ASOH and transient state, making sure they are not batched
        if asoh.batch_size > 1:
            raise ValueError(f'The batch size of the ASOH must be 1. Found: {asoh.batch_size}')
        self.asoh = asoh
        if transient_state.batch_size > 1:
            raise ValueError(f'The batch size of the transient state must be 1. Found: {transient_state.batch_size}')
        self.transient_state = transient_state

        # Store the information about the identity of variables in the transient state
        self.transient_inputs = transient_inputs
        self.asoh_inputs = asoh_inputs
        self.input_template = input_template
        self.num_transients = transient_state.to_numpy().shape[1] if transient_inputs else 0
        self.num_asoh = asoh.get_parameters(self.asoh_inputs).shape[1]

    def create_hidden_state(self, asoh: HealthVariable, transients: GeneralContainer) -> np.ndarray:
        """Transform the state of health and transients states (quantities used by CellModel)
        into the "hidden state" vector used by the actual filter

        Args:
            asoh: Values of the ASOH parameter
            transients: Values of the transient states
        Returns:
            A hidden state vector ready for use in :meth:`__call__`
        """

        if not self.transient_inputs:
            return asoh.get_parameters(self.asoh_inputs)
        else:
            return np.concatenate([
                transients.to_numpy(),
                asoh.get_parameters(self.asoh_inputs)
            ], axis=1)

    def create_cell_model_inputs(self, hidden_states: np.ndarray) -> Tuple[HealthVariable, GeneralContainer]:
        """Convert the hidden states into the forms used by CellModel

        Args:
            hidden_states: Hidden states as used by the estimator
        Returns:
            - ASOH with values from the hidden states
            - Transients state from the hidden states
        """

        # Update any parameters for the transient state
        my_transients = self.transient_state.model_copy(deep=True)
        batch_transients = np.repeat(self.transient_state.to_numpy(), axis=0, repeats=hidden_states.shape[0])
        if self.transient_inputs:
            batch_transients = hidden_states[:, :self.num_transients]
        my_transients.from_numpy(batch_transients)

        # Update the ASOH accordingly
        my_asoh = self.asoh.model_copy(deep=True)
        my_asoh.update_parameters(hidden_states[:, self.num_transients:], self.asoh_inputs)
        return my_asoh, my_transients

    def update_hidden_state(self,
                            hidden_states: np.ndarray,
                            new_control: np.ndarray,
                            previous_control: np.ndarray) -> np.ndarray:
        """Predict the update for hidden states under specified controls

        Args:
            hidden_states: A batch of hidden states as a 2D numpy array. The first dimension is the batch dimension
            new_control: Control at the present timestep
            previous_control: Control at the previous timestep
        Returns:
            Updated hidden states
        """

        # If transients are not included in the transients, the hidden state will not be updated
        if not self.transient_inputs:
            return hidden_states.copy()

        previous_inputs = self.input_template.model_copy(deep=True)
        previous_inputs.from_numpy(previous_control)
        new_inputs = self.input_template.model_copy(deep=True)
        new_inputs.from_numpy(new_control)

        # Undo any normalizing
        my_asoh, my_transients = self.create_cell_model_inputs(hidden_states)

        # Now, iterate through the hidden states to create ECMTransient states and update them
        output = hidden_states.copy()
        new_transients = self.model.update_transient_state(previous_inputs, new_inputs=new_inputs,
                                                           transient_state=my_transients,
                                                           asoh=my_asoh)
        output[:, :self.num_transients] = new_transients.to_numpy()
        return output

    def predict_outputs(self,
                        hidden_states: np.ndarray,
                        new_control: np.ndarray) -> np.ndarray:
        """Predict the observable outputs of the CellModel provided the current hidden states

        Args:
            hidden_states: A batch of hidden states as a 2D numpy array. The first dimension is the batch dimension
            new_control: Control at the present timestep
        Returns:
            Outputs for each hidden state in the batch
        """

        # First, transform the controls into ECM inputs
        inputs = self.input_template.model_copy(deep=True)
        inputs.from_numpy(new_control)

        # Now, iterate through hidden states to compute terminal voltage
        my_asoh, my_transients = self.create_cell_model_inputs(hidden_states)
        outputs = self.model.calculate_terminal_voltage(new_inputs=inputs, transient_state=my_transients, asoh=my_asoh)
        return outputs.to_numpy()

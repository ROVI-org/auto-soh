"""Tools to reduce operations on :class:`~moirae.models.baseCellModel` to functions which act only on
widely-used Python types, such as Numpy Arrays."""
from abc import abstractmethod

import numpy as np
from typing import Tuple

from moirae.models.base import InputQuantities, GeneralContainer, HealthVariable, CellModel


# TODO (wardlt): Implement the "ASOHOnly" interface needed by the Dual Estimator by making it such that
#  the `predict_output` function first estimates how the transients will update for each set of ASOH,
#  then uses the updated transient states and ASOH to determine the outptus

class CellModelInterface:
    """Link between the :class:`~moirae.model.base.CellModel` and the numpy-only interface of
    the filter implementations."""

    model: CellModel
    """Cell model underpinning the update functions"""
    asoh: HealthVariable
    """ASOH values passed to each call of the cell model"""
    transients: GeneralContainer
    """Transient states used for the inputs of the model"""

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer
                 ):

        # Store the ASOH and transient state, making sure they are not batched
        if asoh.batch_size > 1:
            raise ValueError(f'The batch size of the ASOH must be 1. Found: {asoh.batch_size}')
        if transients.batch_size > 1:
            raise ValueError(f'The batch size of the transient state must be 1. Found: {transients.batch_size}')

        self.transients = transients
        self.model = cell_model
        self.asoh = asoh

    @abstractmethod
    def update_hidden_state(self, hidden_states, new_control, previous_control):
        """Predict the update for hidden states under specified controls

        Args:
            hidden_states: A batch of hidden states as a 2D numpy array. The first dimension is the batch dimension
            new_control: Control at the present timestep
            previous_control: Control at the previous timestep
        Returns:
            Updated hidden states
        """
        pass

    @abstractmethod
    def predict_outputs(self, hidden_states, new_control):
        """Predict the observable outputs of the CellModel provided the current hidden states

        Args:
            hidden_states: A batch of hidden states as a 2D numpy array. The first dimension is the batch dimension
            new_control: Control at the present timestep
        Returns:
            Outputs for each hidden state in the batch
        """
        pass


class JointCellModelInterface(CellModelInterface):
    """Interface used when the hidden state used by a filter includes the transient states.

    Create the interface by defining
        - Which portions of the ASOH are used as inputs to function
        - Values for the ASOH parameters that remain fixed
        - An example transient state and input to be passed to the function which will be used as a template

    The resultant function will take numpy arrays as inputs and produce numpy arrays as outputs

    Args:
        cell_model: Model which defines the physics of the system being modeled
        asoh: Values for all state of health parameters of the model
        transients: Current values of the transient state of the system
        input_template: Example input values for the model
        asoh_inputs: Names of the ASOH parameters to include as inputs
    """

    def __init__(self,
                 model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 input_template: InputQuantities,
                 asoh_inputs: Tuple[str]):
        super().__init__(model, asoh, transients)

        # Store the information about the identity of variables in the transient state
        self.asoh_inputs = asoh_inputs
        self.input_template = input_template
        self.num_transients = transients.to_numpy().shape[1]
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
        my_transients = self.transients.model_copy(deep=True)
        batch_transients = np.repeat(self.transients.to_numpy(), axis=0, repeats=hidden_states.shape[0])
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

        # Transmute the controls and hidden state into the form required for the CellModel
        previous_inputs = self.input_template.model_copy(deep=True)
        previous_inputs.from_numpy(previous_control)
        new_inputs = self.input_template.model_copy(deep=True)
        new_inputs.from_numpy(new_control)

        my_asoh, my_transients = self.create_cell_model_inputs(hidden_states)

        # Produce an updated estimate for the transient states, hold the ASOH parameters constant
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

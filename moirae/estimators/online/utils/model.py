"""Tools to reduce operations on :class:`~moirae.models.base.CellModel` to functions which act only on
widely-used Python types, such as Numpy Arrays."""
import numpy as np
from typing import Tuple, Optional, Union

from moirae.estimators.online.filters.base import ModelWrapper
from moirae.estimators.online.filters.distributions import DeltaDistribution, MultivariateGaussian
from moirae.models.base import InputQuantities, OutputQuantities, GeneralContainer, HealthVariable, CellModel


def convert_vals_model_to_filter(
        model_quantities: Union[GeneralContainer, InputQuantities, OutputQuantities],
        uncertainty_matrix: Optional[np.ndarray] = None) -> Union[DeltaDistribution, MultivariateGaussian]:
    """
    Function that converts model-related quantities (but not HealthVariable!) to filter-related quantities.
    If uncertainty is provided, assumes a Multivariate Gaussian. Otherwise, assumes Delta Distribution

    Args:
        model_quantities: model-related object to be converted into filter-related object
        uncertainty_matrix: 2D array to be used as covariance matrix; if not provided, returns DeltaDistribution

    Returns:
        a corresponding MultivariateRandomDistribution (either Gaussian or Delta)
    """
    if uncertainty_matrix is None:
        return DeltaDistribution(mean=model_quantities.to_numpy())
    return MultivariateGaussian(mean=model_quantities.to_numpy(), covariance=uncertainty_matrix)


# TODO (wardlt): Implement the "ASOHOnly" interface needed by the Dual Estimator by making it such that
#  the `predict_output` function first estimates how the transients will update for each set of ASOH,
#  then uses the updated transient states and ASOH to determine the outptus

class CellModelInterface(ModelWrapper):
    """Link between the :class:`~moirae.model.base.CellModel` and the numpy-only interface of
    the filter implementations."""

    cell_model: CellModel
    """Cell model underpinning the update functions"""
    asoh: HealthVariable
    """ASOH values passed to each call of the cell model"""
    transients: GeneralContainer
    """Transient states used for the inputs of the model"""

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 input_template: InputQuantities):

        # Store the ASOH and transient state, making sure they are not batched
        if asoh.batch_size > 1:
            raise ValueError(f'The batch size of the ASOH must be 1. Found: {asoh.batch_size}')
        if transients.batch_size > 1:
            raise ValueError(f'The batch size of the transient state must be 1. Found: {transients.batch_size}')

        self.transients = transients
        self.cell_model = cell_model
        self.asoh = asoh
        self.input_template = input_template

        # Capture the shape of the outputs
        self._num_output_dimensions = self.cell_model.calculate_terminal_voltage(self.input_template,
                                                                                 self.transients,
                                                                                 self.asoh).to_numpy().shape[1]

    @property
    def num_output_dimensions(self) -> int:
        return self._num_output_dimensions


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
        asoh_inputs: Names of the ASOH parameters to include as part of the hidden state
    """

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 input_template: InputQuantities,
                 asoh_inputs: Optional[Tuple[str]] = None):
        super().__init__(cell_model=cell_model, asoh=asoh, transients=transients, input_template=input_template)

        # Store the information about the identity of variables in the transient state
        if asoh_inputs is None:
            asoh_inputs = asoh.updatable_names
        self.asoh_inputs = asoh_inputs
        self.num_transients = transients.to_numpy().shape[1]
        self.num_asoh = asoh.get_parameters(self.asoh_inputs).shape[1]

    @property
    def num_hidden_dimensions(self) -> int:
        return self.num_transients + self.num_asoh

    def create_hidden_state(self, asoh: HealthVariable, transients: GeneralContainer) -> np.ndarray:
        """Transform the state of health and transients states (quantities used by CellModel)
        into the "hidden state" vector used by the actual filter

        Args:
            asoh: Values of the ASOH parameter
            transients: Values of the transient states
        Returns:
            A hidden state vector ready for use in a filter
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

    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: np.ndarray,
                             new_controls: np.ndarray) -> np.ndarray:
        # Transmute the controls and hidden state into the form required for the CellModel
        previous_inputs = self.input_template.model_copy(deep=True)
        previous_inputs.from_numpy(previous_controls)
        new_inputs = self.input_template.model_copy(deep=True)
        new_inputs.from_numpy(new_controls)

        my_asoh, my_transients = self.create_cell_model_inputs(hidden_states)

        # Produce an updated estimate for the transient states, hold the ASOH parameters constant
        output = hidden_states.copy()
        new_transients = self.cell_model.update_transient_state(previous_inputs, new_inputs=new_inputs,
                                                                transient_state=my_transients,
                                                                asoh=my_asoh)
        output[:, :self.num_transients] = new_transients.to_numpy()
        return output

    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: np.ndarray) -> np.ndarray:
        # First, transform the controls into ECM inputs
        inputs = self.input_template.model_copy(deep=True)
        inputs.from_numpy(controls)

        # Now, iterate through hidden states to compute terminal voltage
        my_asoh, my_transients = self.create_cell_model_inputs(hidden_states)
        outputs = self.cell_model.calculate_terminal_voltage(new_inputs=inputs,
                                                             transient_state=my_transients,
                                                             asoh=my_asoh)
        return outputs.to_numpy()

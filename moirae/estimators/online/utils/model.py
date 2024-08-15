"""Tools to reduce operations on :class:`~moirae.models.base.CellModel` to functions which act only on
widely-used Python types, such as Numpy Arrays."""
import numpy as np
from typing import Tuple, Optional, Union

from moirae.estimators.online.filters.base import ModelWrapper
from moirae.estimators.online.filters.distributions import DeltaDistribution, MultivariateGaussian
from moirae.models.base import InputQuantities, GeneralContainer, HealthVariable, CellModel


def convert_vals_model_to_filter(
        model_quantities: GeneralContainer,
        uncertainty_matrix: Optional[np.ndarray] = None) -> Union[DeltaDistribution, MultivariateGaussian]:
    """
    Function that converts :class:`~moirae.model.base.GeneralContainer` object to filter-related quantities.
    If uncertainty is provided, assumes a :class:`~moirae.estimators.online.filters.distributions.MultivariateGaussian`.
    Otherwise, assumes :class:`~moirae.estimators.online.filters.distributions.DeltaDistribution`.

    Args:
        model_quantities: model-related object to be converted into filter-related object
        uncertainty_matrix: 2D array to be used as covariance matrix; if not provided, returns DeltaDistribution

    Returns:
        a corresponding MultivariateRandomDistribution (either Gaussian or Delta)
    """
    if uncertainty_matrix is None:
        return DeltaDistribution(mean=model_quantities.to_numpy().flatten())
    return MultivariateGaussian(mean=model_quantities.to_numpy().flatten(), covariance=uncertainty_matrix)


def convert_numpy_to_model(filter_array: np.ndarray,
                           template: Union[GeneralContainer, HealthVariable],
                           names: Optional[Tuple[str, ...]] = None) -> Union[GeneralContainer, HealthVariable]:
    """
    Function to convert numpy arrays to model-related object given a template of said object: either a
    :class:`~moirae.model.base.GeneralContainer` or a :class:`~moirae.model.base.HealthVariable`.

    Args:
        filter_array: array of numerical values to be converted
        template: template object to be used; original object will not be modified
        names: in case of a `HealthVariable`, can update specific names

    Returns:
        corresponding model-related object given by template with numerical values from the array
    """
    model_related_object = template.model_copy(deep=True)
    if isinstance(model_related_object, GeneralContainer):
        return model_related_object.from_numpy(filter_array)
    else:
        return model_related_object.update_parameters(values=filter_array, names=names)


class BaseCellWrapper(ModelWrapper):
    """
    Base link between the :class:`~moirae.model.base.CellModel` and the numpy-only interface of
    the filter implementations.
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
                 transients: GeneralContainer,
                 inputs: InputQuantities):

        # Store the ASOH and transient state, making sure they are not batched
        if asoh.batch_size > 1:
            raise ValueError(f'The batch size of the ASOH must be 1. Found: {asoh.batch_size}')
        if transients.batch_size > 1:
            raise ValueError(f'The batch size of the transient state must be 1. Found: {transients.batch_size}')

        self.transients = transients
        self.cell_model = cell_model
        self.asoh = asoh
        self.inputs = inputs

        # Capture the shape of the outputs
        self._num_output_dimensions = self.cell_model.calculate_terminal_voltage(new_inputs=self.inputs,
                                                                                 transient_state=self.transients,
                                                                                 asoh=self.asoh).to_numpy().shape[1]

    @property
    def num_output_dimensions(self) -> int:
        return self._num_output_dimensions


class CellModelWrapper(BaseCellWrapper):
    """
    Base link between the :class:`~moirae.model.base.CellModel` and the numpy-only interface of
    :class:`~moirae.estimators.online.filters.base.BaseFilter` filter implementations. This particular wrapper does not
    touch the A-SOH, but only uses it for predictions.
    """
    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 inputs: InputQuantities) -> None:
        super().__init__(cell_model=cell_model, asoh=asoh, transients=transients, inputs=inputs)

        self.num_transients = transients.to_numpy().shape[1]

    @property
    def num_hidden_dimensions(self) -> int:
        return self.num_transients

    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: np.ndarray,
                             new_controls: np.ndarray) -> np.ndarray:
        """
        Function that takes a numpy representation of the transient states and updates it based on the cell model, the
        previous and new controls. Recall the cell model needs the A-SOH as well!
        """
        # Convert objects
        transients = convert_numpy_to_model(filter_array=hidden_states, template=self.transients)
        previous_inputs = convert_numpy_to_model(filter_array=previous_controls, template=self.inputs)
        new_inputs = convert_numpy_to_model(filter_array=new_controls, template=self.inputs)

        # Update transients
        new_transients = self.cell_model.update_transient_state(previous_inputs=previous_inputs,
                                                                new_inputs=new_inputs,
                                                                transient_state=transients,
                                                                asoh=self.asoh)

        return new_transients.to_numpy()

    def predict_measurement(self, hidden_states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """
        Function that takes a numpy representation of the transient state and of the controls, and predicts measurements
        from it
        """
        # Convert objects
        transients = convert_numpy_to_model(filter_array=hidden_states, template=self.transients)
        inputs = convert_numpy_to_model(filter_array=controls, template=self.inputs)

        # Get output
        measurements = self.cell_model.calculate_terminal_voltage(new_inputs=inputs,
                                                                  transient_state=transients,
                                                                  asoh=self.asoh)

        return measurements.to_numpy()


class DegradationModelWrapper(BaseCellWrapper):
    """
    Link between A-SOH degradation models and the numpy-only interface of the
    :class:`~moirae.estimators.online.filters.base.BaseFilter`. If provides the model wrapper need for dual estimation
    frameworks
    """

    asoh_inputs: Tuple[str]
    """Names of the parameters from the ASOH which are used as inputs to the model"""

    def __init__(self,
                 cell_model: CellModel,
                 # TODO (vventuri): allow for passing an A-SOH degradation model!
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 inputs: InputQuantities,
                 asoh_inputs: Optional[Tuple[str]] = None) -> None:

        super().__init__(cell_model=cell_model, asoh=asoh, transients=transients, inputs=inputs)

        # Store the information about the identity of variables in the transient state
        if asoh_inputs is None:
            asoh_inputs = asoh.updatable_names
        self.asoh_inputs = asoh_inputs
        self.num_transients = transients.to_numpy().shape[1]
        self.num_asoh = asoh.get_parameters(self.asoh_inputs).shape[1]
        # Storing "previous inputs" just in case a call to `predict_measurement` is made prematurely
        self._previous_inputs = inputs.model_copy(deep=True)

    @property
    def num_hidden_dimensions(self) -> int:
        return self.num_asoh

    def _convert_hidden_to_asoh(self, hidden_states: np.ndarray) -> HealthVariable:
        """
        Helper function to take hidden states and convert them to A-SOH object to be given to degradation model
        """
        return convert_numpy_to_model(filter_array=hidden_states, template=self.asoh, names=self.asoh_inputs)

    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: np.ndarray,
                             new_controls: np.ndarray) -> np.ndarray:
        """
        Function that takes a numpy representation of the A-SOH and degrades it using the degradation model and the
        previous and new controls. If the degradation model needs information for further in the past, it is responsible
        for keeping track of that. The degration model will also be provided the current estimate of the transient
        vector.
        """
        # Remember that, during this step, we should also store the previous controls so that the transient vector can
        # be propagated through the hidden states in the predict measurement step
        previous_inputs = convert_numpy_to_model(filter_array=previous_controls, template=self.inputs)
        self._previous_inputs = previous_inputs

        return hidden_states.copy()

    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: np.ndarray) -> np.ndarray:
        """
        Function that takes the numpy-representation of the estimated of A-SOH and computes predictions of the
        measurement. Recall that, for that, we first need to propagate the transients through the A-SOH estimates
        """
        # First, transform the controls into ECM inputs
        inputs = self.inputs.model_copy(deep=True)
        inputs.from_numpy(controls)

        # Do the same for the A-SOH
        asoh = self._convert_hidden_to_asoh(hidden_states=hidden_states)

        # Now, propagate the transients through the A-SOH
        propagated_transients = self.cell_model.update_transient_state(previous_inputs=self._previous_inputs,
                                                                       new_inputs=inputs,
                                                                       transient_state=self.transients,
                                                                       asoh=asoh)

        # Finally, compute outputs
        outputs = self.cell_model.calculate_terminal_voltage(new_inputs=inputs,
                                                             transient_state=propagated_transients,
                                                             asoh=asoh)
        return outputs.to_numpy()


class JointCellModelWrapper(BaseCellWrapper):
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
        inputs: Example input values for the model
        asoh_inputs: Names of the ASOH parameters to include as part of the hidden state
    """

    asoh_inputs: Tuple[str]
    """Names of the parameters from the ASOH which are used as inputs to the model"""

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 inputs: InputQuantities,
                 asoh_inputs: Optional[Tuple[str]] = None):
        super().__init__(cell_model=cell_model, asoh=asoh, transients=transients, inputs=inputs)

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
        my_transients.from_numpy(hidden_states[:, :self.num_transients])

        # Update the ASOH accordingly
        my_asoh = self.asoh.model_copy(deep=True)
        my_asoh.update_parameters(hidden_states[:, self.num_transients:], self.asoh_inputs)
        return my_asoh, my_transients

    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: np.ndarray,
                             new_controls: np.ndarray) -> np.ndarray:
        # Transmute the controls and hidden state into the form required for the CellModel
        previous_inputs = self.inputs.model_copy(deep=True)
        previous_inputs.from_numpy(previous_controls)
        new_inputs = self.inputs.model_copy(deep=True)
        new_inputs.from_numpy(new_controls)

        my_asoh, my_transients = self.create_cell_model_inputs(hidden_states)

        # Produce an updated estimate for the transient states, hold the ASOH parameters constant
        output = hidden_states.copy()
        new_transients = self.cell_model.update_transient_state(previous_inputs=previous_inputs,
                                                                new_inputs=new_inputs,
                                                                transient_state=my_transients,
                                                                asoh=my_asoh)
        output[:, :self.num_transients] = new_transients.to_numpy()
        return output

    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: np.ndarray) -> np.ndarray:
        # First, transform the controls into ECM inputs
        inputs = self.inputs.model_copy(deep=True)
        inputs.from_numpy(controls)

        # Now, iterate through hidden states to compute terminal voltage
        my_asoh, my_transients = self.create_cell_model_inputs(hidden_states)
        outputs = self.cell_model.calculate_terminal_voltage(new_inputs=inputs,
                                                             transient_state=my_transients,
                                                             asoh=my_asoh)
        return outputs.to_numpy()

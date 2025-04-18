"""Tools to reduce operations on :class:`~moirae.models.base.CellModel` to functions which act only on
widely-used Python types, such as Numpy Arrays."""
from abc import ABC

import numpy as np
from typing import Tuple, Optional, Union

from moirae.estimators.online.filters.base import ModelWrapper, ModelWrapperConverters
from moirae.estimators.online.filters.distributions import DeltaDistribution, MultivariateGaussian
from moirae.models.base import InputQuantities, GeneralContainer, HealthVariable, CellModel, DegradationModel
from moirae.models.utils import NoDegradation


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


class BaseCellWrapper(ModelWrapper, ABC):
    """
    Base link between the :class:`~moirae.model.base.CellModel` and the numpy-only interface of
    the filter implementations.

    Attributes provide access to the ASOH and transient state employed in the last call of the previous model.

    Args:
        cell_model: Model which defines the physics of the system being modeled
        asoh: Values for all state of health parameters of the model
        transients: Current values of the transient state of the system
        inputs: Example input values for the model
        converters: set of converters to be used to translate between filter-coordinate system and model
            coordinate-system
    """

    cell_model: CellModel
    """Cell model underpinning the update functions"""
    asoh: HealthVariable
    """ASOH values passed to each call of the cell model"""
    transients: GeneralContainer
    """Transient states used for the inputs of the model"""
    previous_inputs: InputQuantities
    """Inputs at the last timestep"""
    inputs: InputQuantities
    """Inputs at the current timestep"""

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 inputs: InputQuantities,
                 converters: ModelWrapperConverters = ModelWrapperConverters.defaults()):

        # Give converters to parent class
        super().__init__(**converters)

        # Store the ASOH and transient state, making sure they are not batched
        if asoh.batch_size > 1:
            raise ValueError(f'The batch size of the ASOH must be 1. Found: {asoh.batch_size}')
        if transients.batch_size > 1:
            raise ValueError(f'The batch size of the transient state must be 1. Found: {transients.batch_size}')

        # Make copies because these class atributes will be altered during
        #  execution of the model
        self.transients = transients.model_copy(deep=True)
        self.cell_model = cell_model
        self.asoh = asoh.model_copy(deep=True)
        self.inputs = inputs.model_copy(deep=True)
        self.previous_inputs = inputs.model_copy(deep=True)

        # Capture the shape of the outputs
        self._num_output_dimensions = self.cell_model.calculate_terminal_voltage(new_inputs=self.inputs,
                                                                                 transient_state=self.transients,
                                                                                 asoh=self.asoh).to_numpy().shape[1]

    @property
    def num_output_dimensions(self) -> int:
        return self._num_output_dimensions


class CellModelWrapper(BaseCellWrapper):
    """
    This particular wrapper does not touch the A-SOH, but only uses it for predictions.
    """

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 inputs: InputQuantities,
                 converters: ModelWrapperConverters = ModelWrapperConverters.defaults()) -> None:
        super().__init__(cell_model=cell_model, asoh=asoh, transients=transients, inputs=inputs, converters=converters)
        self.num_transients = transients.to_numpy().shape[1]

    @property
    def num_hidden_dimensions(self) -> int:
        return self.num_transients

    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: np.ndarray,
                             new_controls: np.ndarray) -> np.ndarray:
        # Convert objects
        self.transients.from_numpy(self.hidden_conversion.transform_samples(samples=hidden_states))
        self.previous_inputs.from_numpy(
            self.control_conversion.transform_samples(samples=previous_controls)
        )
        self.inputs.from_numpy(
            values=self.control_conversion.transform_samples(samples=new_controls)
        )

        # Update transients
        new_transients = self.cell_model.update_transient_state(previous_inputs=self.previous_inputs,
                                                                new_inputs=self.inputs,
                                                                transient_state=self.transients,
                                                                asoh=self.asoh)

        # Convert back to filter language
        new_transients = self.hidden_conversion.inverse_transform_samples(transformed_samples=new_transients.to_numpy())

        return new_transients

    def predict_measurement(self, hidden_states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """
        Function that takes a numpy representation of the transient state and of the controls, and predicts measurements
        from it
        """
        # Convert objects
        self.transients.from_numpy(self.hidden_conversion.transform_samples(samples=hidden_states))
        self.inputs.from_numpy(self.control_conversion.transform_samples(samples=controls))

        # Get output
        measurements = self.cell_model.calculate_terminal_voltage(new_inputs=self.inputs,
                                                                  transient_state=self.transients,
                                                                  asoh=self.asoh)

        # Convert back to filter lingo
        measurements = self.output_conversion.inverse_transform_samples(transformed_samples=measurements.to_numpy())

        return measurements


class DegradationModelWrapper(BaseCellWrapper):
    """
    Link between A-SOH degradation models and the numpy-only interface of the
    :class:`~moirae.estimators.online.filters.base.BaseFilter`. It provides the model wrapper need for dual estimation
    frameworks

    It takes an additional argument compared to the :class:`moirae.estimators.online.utils.model.BaseCellWrapper`, as it
    needs to know the degrataion model being used. If none is passed, defaults to dummy degradation.

    Args:
        cell_model: Model which defines the physics of the system being modeled
        asoh: Values for all state of health parameters of the model
        transients: Current values of the transient state of the system
        inputs: Example input values for the model
        asoh_inputs: Names of the ASOH parameters to include as part of the hidden state
        degradation_model: degradation model to be used; if not passed, degradation will be fully derived from data
        converters: set of converters to be used to translate between filter-coordinate system and model
            coordinate-system
    """

    asoh_inputs: Tuple[str]
    """Names of the parameters from the ASOH which are used as inputs to the model"""

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 inputs: InputQuantities,
                 asoh_inputs: Optional[Tuple[str]] = None,
                 degradation_model: DegradationModel = NoDegradation(),
                 converters: ModelWrapperConverters = ModelWrapperConverters.defaults()) -> None:
        super().__init__(cell_model=cell_model, asoh=asoh, transients=transients, inputs=inputs, converters=converters)

        # Store the information about the identity of variables in the transient state
        if asoh_inputs is None:
            asoh_inputs = asoh.updatable_names
        self.asoh_inputs = asoh_inputs
        self.num_transients = transients.to_numpy().shape[1]
        self.num_asoh = asoh.get_parameters(self.asoh_inputs).shape[1]
        # Storing "previous inputs" just in case a call to `predict_measurement` is made prematurely
        self.previous_inputs = inputs.model_copy(deep=True)
        # Store degradation model
        self.deg_model = degradation_model

    @property
    def num_hidden_dimensions(self) -> int:
        return self.num_asoh

    def _update_hidden_asoh(self, hidden_states: np.ndarray):
        """
        Helper function to take hidden states and update the internal ASOH object to be given to degradation model
        """
        self.asoh.update_parameters(
            values=self.hidden_conversion.transform_samples(samples=hidden_states),
            names=self.asoh_inputs
        )

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
        self.previous_inputs.from_numpy(values=self.control_conversion.transform_samples(samples=previous_controls))

        # Convert to A-SOH object
        self._update_hidden_asoh(hidden_states=hidden_states)
        # Convert new controls
        self.inputs.from_numpy(self.control_conversion.transform_samples(samples=new_controls))
        # Compute measurements
        new_measurements = self.cell_model.calculate_terminal_voltage(new_inputs=self.inputs,
                                                                      transient_state=self.transients,
                                                                      asoh=self.asoh)
        # Degrade A-SOH
        deg_asoh = self.deg_model.update_asoh(previous_asoh=self.asoh,
                                              new_inputs=self.inputs,
                                              new_transients=self.transients,
                                              new_measurements=new_measurements)

        return self.hidden_conversion.inverse_transform_samples(
            transformed_samples=deg_asoh.get_parameters(names=self.asoh_inputs)
        )

    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: np.ndarray) -> np.ndarray:
        """
        Function that takes the numpy-representation of the estimated A-SOH and computes predictions of the
        measurement. Recall that, for that, we first need to propagate the transients through the A-SOH estimates
        """
        # First, transform the controls into ECM inputs
        self.inputs.from_numpy(values=self.control_conversion.transform_samples(samples=controls))

        # Do the same for the A-SOH
        self._update_hidden_asoh(hidden_states=hidden_states)

        # Now, propagate the transients through the A-SOH
        propagated_transients = self.cell_model.update_transient_state(previous_inputs=self.previous_inputs,
                                                                       new_inputs=self.inputs,
                                                                       transient_state=self.transients,
                                                                       asoh=self.asoh)

        # Finally, compute outputs
        outputs = self.cell_model.calculate_terminal_voltage(new_inputs=self.inputs,
                                                             transient_state=propagated_transients,
                                                             asoh=self.asoh)

        # Convert outputs to filter lingo
        outputs = self.output_conversion.inverse_transform_samples(transformed_samples=outputs.to_numpy())
        return outputs


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
        degradation_model: degradation model to be used; if not passed, degradation will be fully derived from data
        converters: set of converters to be used to translate between filter-coordinate system and model
            coordinate-system
    """

    asoh_inputs: Tuple[str]
    """Names of the parameters from the ASOH which are used as inputs to the model"""

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transients: GeneralContainer,
                 inputs: InputQuantities,
                 asoh_inputs: Optional[Tuple[str]] = None,
                 degradation_model: DegradationModel = NoDegradation(),
                 converters: ModelWrapperConverters = ModelWrapperConverters.defaults()) -> None:
        super().__init__(cell_model=cell_model, asoh=asoh, transients=transients, inputs=inputs, converters=converters)

        # Store the information about the identity of variables in the transient state
        if asoh_inputs is None:
            asoh_inputs = asoh.updatable_names
        self.asoh_inputs = asoh_inputs
        self.num_transients = transients.to_numpy().shape[1]
        self.num_asoh = asoh.get_parameters(self.asoh_inputs).shape[1]
        self.deg_model = degradation_model

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

        # Get raw values
        trans_raw = transients.to_numpy()
        asoh_raw = asoh.get_parameters(names=self.asoh_inputs)

        # Concatenate
        joint_raw = np.concatenate([trans_raw, asoh_raw], axis=1)

        # Convert to filter
        joint = self.hidden_conversion.inverse_transform_samples(transformed_samples=joint_raw)

        return joint

    def update_cell_inputs(self, hidden_states: np.ndarray):
        """Convert the hidden states into the forms used by CellModel

        Alters the copies of the ASOH (:attr:`self.asoh`) and
        the transients (:attr:`self.transients`) used by the model wrapper.

        Args:
            hidden_states: Hidden states as used by the estimator
        """

        # Get raw values
        joint_raw = self.hidden_conversion.transform_samples(samples=hidden_states)

        # Update any parameters for the transient state
        self.transients.from_numpy(values=joint_raw[:, :self.num_transients])

        # Update the ASOH accordingly
        self.asoh.update_parameters(joint_raw[:, self.num_transients:], names=self.asoh_inputs)

    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: np.ndarray,
                             new_controls: np.ndarray) -> np.ndarray:
        # Transmute the controls and hidden state into the form required for the CellModel
        self.previous_inputs.from_numpy(self.control_conversion.transform_samples(samples=previous_controls))
        self.inputs.from_numpy(self.control_conversion.transform_samples(samples=new_controls))
        self.update_cell_inputs(hidden_states)

        # Produce an updated estimate for the transient states, hold the ASOH parameters constant
        output = self.hidden_conversion.transform_samples(samples=hidden_states)
        new_transients = self.cell_model.update_transient_state(previous_inputs=self.previous_inputs,
                                                                new_inputs=self.inputs,
                                                                transient_state=self.transients,
                                                                asoh=self.asoh)

        # Let's also degrade the A-SOH
        # We need the measurement
        new_measurement = self.cell_model.calculate_terminal_voltage(new_inputs=self.inputs,
                                                                     transient_state=self.transients,
                                                                     asoh=self.asoh)
        deg_asoh = self.deg_model.update_asoh(previous_asoh=self.asoh,
                                              new_inputs=self.inputs,
                                              new_transients=self.transients,
                                              new_measurements=new_measurement)

        # Convert this back to filter lingo
        output[:, :self.num_transients] = new_transients.to_numpy()
        output[:, self.num_transients:] = deg_asoh.get_parameters(names=self.asoh_inputs)
        output = self.hidden_conversion.inverse_transform_samples(transformed_samples=output)
        return output

    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: np.ndarray) -> np.ndarray:
        # First, transform the controls into ECM inputs
        self.inputs.from_numpy(values=self.control_conversion.transform_samples(samples=controls))

        # Now, iterate through hidden states to compute terminal voltage
        self.update_cell_inputs(hidden_states)
        outputs = self.cell_model.calculate_terminal_voltage(new_inputs=self.inputs,
                                                             transient_state=self.transients,
                                                             asoh=self.asoh)
        # Convert to filter lingo
        outputs = self.output_conversion.inverse_transform_samples(transformed_samples=outputs.to_numpy())
        return outputs

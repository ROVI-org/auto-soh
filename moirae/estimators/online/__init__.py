"""
Collection of online estimators

Here, we include base classes to facilitate  definition of online estimators and hidden states, as well as to establish
interfaces between online estimators and models, transient states, and A-SOH parameters.
"""
from abc import abstractmethod
from functools import cached_property
from typing import Tuple, Union, Collection, Optional

import numpy as np
from pydantic import BaseModel, Field

from moirae.models.base import CellModel, GeneralContainer, InputQuantities, HealthVariable


class MultivariateRandomDistribution(BaseModel, arbitrary_types_allowed=True):
    """
    Base class to help represent a multivariate random variable
    """

    @abstractmethod
    def get_mean(self) -> np.ndarray:
        """
        Provides mean (first moment) of distribution
        """
        pass


class HiddenState(MultivariateRandomDistribution):
    """
    Defines the hidden state that is updated by the online estimator.
    """
    mean: np.ndarray = Field(default=None,
                             description='Mean of the random distribution that describes the hidden state')

    def get_mean(self) -> np.ndarray:
        return self.mean.copy()


class OutputMeasurements(MultivariateRandomDistribution):
    """
    Defines a container for the outputs
    """
    mean: np.ndarray = Field(default=None,
                             description='Mean of the random distribution that describes the output measurement')

    def get_mean(self) -> np.ndarray:
        return self.mean.copy()


class ControlVariables(MultivariateRandomDistribution):
    """
    Define the container for the controls. We are setting as a random variable, but, for most purposes, its probability
    distribution is to be considered a delta function centered on the mean.
    """
    mean: np.ndarray = Field(default=None,
                             description='Mean of the random distribution that describes the control variables')

    def get_mean(self) -> np.ndarray:
        return self.mean.copy()


# TODO (wardlt): Consider letting users pass a custom normalization function rather than implementing a subclass
class OnlineEstimator:
    """
    Defines the base structure of an online estimator.

    All estimators require...

    1. A :class:`~moirae.models.base.CellModel` which describes how the system state is expected to change and
        relate the current state to observable measurements.
    2. An initial estimate for the parameters of the system, which we refer to as the Advanced State of Health (ASOH).
    3. An initial estimate for the transient states of the system

    Different implementations may require other information, such as an initial guess for the
    probability distribution for the values of the states (transient or ASOH).

    Use the estimator by calling the :meth:`step` function to update the estimated state
    provided a new observation of the outputs of the system.

    """
    model: CellModel
    """Link to the model describing the known physics of the system"""

    def __init__(self,
                 model: CellModel,
                 initial_asoh: HealthVariable,
                 initial_transients: GeneralContainer,
                 initial_inputs: InputQuantities,
                 updatable_asoh: Union[bool, Collection[str]] = True):
        """
        Args:
            model: Model used to describe the underlying physics of the storage system
            initial_asoh: Initial estimates for the health parameters of the battery, those being estimated or not
            initial_transients: Initial estimates for the transient states of the battery
            initial_inputs: Initial inputs to the system
            updatable_asoh: Whether to estimate values for all updatable parameters (``True``),
                none of the updatable parameters (``False``),
                or only a select set of them (provide a list of names).
        """

        self.model = model
        self._asoh = initial_asoh.model_copy(deep=True)
        self._transients = initial_transients.model_copy(deep=True)
        self._inputs = initial_inputs.model_copy(deep=True)
        self._num_outputs = len(model.calculate_terminal_voltage(initial_inputs, self._transients, self._asoh))

        # Determine which parameters to treat as updatable in the ASOH
        self._updatable_names: Optional[list[str]]
        if isinstance(updatable_asoh, bool):
            if updatable_asoh:
                self._updatable_names = self._asoh.updatable_names
            else:
                self._updatable_names = []
        else:
            self._updatable_names = list(updatable_asoh)

    @cached_property
    def num_hidden_dimensions(self) -> int:
        """ Expected dimensionality of hidden state """
        return self.num_transients + self._asoh.get_parameters(self._updatable_names).shape[-1]

    @cached_property
    def num_transients(self):
        """Number of values from the hidden state which belong to the transients"""
        return len(self._transients)

    @property
    def num_output_dimensions(self) -> int:
        """ Expected dimensionality of output measurements """
        return self._num_outputs

    def _denormalize_hidden_array(self, hidden_array: np.ndarray) -> np.ndarray:
        """Apply transformations to the hidden array which transform it from the
        form used by the estimator to the form used by the model

        Args:
            hidden_array: Input array of points to be evaluated with the model. Should not be modified
        Returns:
            An array ready for use in the model
        """
        return hidden_array

    def _normalize_hidden_array(self, hidden_array: np.ndarray) -> np.ndarray:
        """Apply transformations to the hidden array which transform it to the
        form used by the estimator to the form used by the model

        Args:
            hidden_array: Input array of points to be used by the filter. Should not be modified
        Returns:
            An array ready produced by the model
        """
        return hidden_array

    # TODO (wardlt): Re-establish allowing controls to be a list when we need it
    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: ControlVariables,
                             new_controls: ControlVariables) -> np.ndarray:
        """
        Function that updates the hidden state based on the control variables provided.

        Args:
            hidden_states: current hidden states of the system as a numpy.ndarray object
            previous_controls: controls at the time the hidden states are being reported
            new_controls: new controls to be used in the hidden state update

        Returns:
            new_hidden: updated hidden states as a numpy.ndarray object
        """

        # First, transform the controls into the input class used by the model
        previous_inputs = self._inputs.model_copy(deep=True)
        previous_inputs.from_numpy(previous_controls.get_mean())
        current_inputs = self._inputs.model_copy(deep=True)
        current_inputs.from_numpy(new_controls.get_mean())

        # Undo any normalizing
        hidden_states = self._denormalize_hidden_array(hidden_states)

        # Now, iterate through the hidden states to create ECMTransient states and update them
        output = hidden_states.copy()
        for i, hidden_array in enumerate(hidden_states):
            # Run the update on the provided state
            self._transients.from_numpy(hidden_array[:self.num_transients])
            self._asoh.update_parameters(hidden_array[self.num_transients:], self._updatable_names)
            new_transient = self.model.update_transient_state(
                previous_input=previous_inputs,
                current_input=current_inputs,
                transient_state=self._transients,
                asoh=self._asoh,
            )

            # Only the new transients (the first part) are updated
            output[i, :self.num_transients] = new_transient.to_numpy()
        return self._normalize_hidden_array(output)

    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: ControlVariables) -> np.ndarray:
        """
        Function to predict measurement from the hidden state

        Args:
            hidden_states: current hidden states of the system as a numpy.ndarray object
            controls: controls to be used for predicting outputs

        Returns:
            pred_measurement: predicted measurements as a numpy.ndarray object
        """
        # First, transform the controls into ECM inputs
        inputs = self._inputs.model_copy(deep=True)
        inputs.from_numpy(controls.get_mean())

        # Now, iterate through hidden states to compute terminal voltage
        voltages = []
        for hidden_array in hidden_states:
            self._transients.from_numpy(hidden_array[:self.num_transients])
            self._asoh.update_parameters(hidden_array[self.num_transients:], self._updatable_names)
            ecm_out = self.model.calculate_terminal_voltage(inputs=inputs,
                                                            transient_state=self._transients,
                                                            asoh=self._asoh)
            voltages.append(ecm_out.to_numpy())

        return np.array(voltages)

    @abstractmethod
    def step(self, u: ControlVariables, y: OutputMeasurements) -> Tuple[OutputMeasurements, HiddenState]:
        """
        Function to step the estimator, provided new control variables and output measurements.

        Args:
            u: control variables
            y: output measurements

        Returns:
            Corrected estimate of the hidden state of the system
        """
        pass

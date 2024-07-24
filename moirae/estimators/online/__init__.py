"""
Collection of online estimators

Here, we include base classes to facilitate  definition of online estimators and hidden states, as well as to establish
interfaces between online estimators and models, transient states, and A-SOH parameters.
"""
from abc import abstractmethod
from functools import cached_property
from typing import Tuple

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


class OnlineEstimator():
    """
    Defines the base structure of an online estimator.

    Args:
        initial_state: initial hidden state of the system
        initial_control: initial control on the system
    """

    def __init__(self,
                 initial_state: HiddenState,
                 initial_control: ControlVariables):
        self.state = initial_state.model_copy(deep=True)
        self.u = initial_control.model_copy(deep=True)

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


class ModelFilterInterface:
    """
    Defines the interface between the cell model and the online estimators.

    Each instance of the model filter interface holds the transient and health parameters
    for a specific storage system, and a :class:`asoh.models.base.CellModel`
    used to update them under the influence of new inputs.

    The "hidden state" associated with a specific ModelFilterInterface includes
    all values in the :attr:`transients` and the updatable parameters from :attr:`asoh`.

    The :meth:`update_hidden_states` calls and :meth:`predict_measurement` accept batches
    of hidden states then either predict a new hidden state (updating only the transient states)
    or render estimates for the outputs.
    """

    model: CellModel
    """Model used to update the hidden state"""

    # TODO (wardlt): We might be able to detect Input/Output types if CellModel was a Generic
    def __init__(self,
                 model: CellModel,
                 initial_transients: GeneralContainer,
                 initial_asoh: HealthVariable,
                 initial_inputs: InputQuantities):
        self.model = model
        self._initial_inputs = initial_inputs.model_copy(deep=True)
        self.transients = initial_transients.model_copy(deep=True)
        self.asoh = initial_asoh.model_copy(deep=True)
        self._num_outputs = len(model.calculate_terminal_voltage(initial_inputs, self.transients, self.asoh))

    @property
    def num_hidden_dimensions(self) -> int:
        """ Expected dimensionality of hidden state """
        return len(self.transients) + self.asoh.num_updatable

    @cached_property
    def num_transients(self):
        """Number of values from the hidden state which belong to the transients"""
        return len(self.transients)

    @property
    def num_output_dimensions(self) -> int:
        """ Expected dimensionality of output measurements """
        return self._num_outputs

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
        previous_inputs = self._initial_inputs.model_copy(deep=True)
        previous_inputs.from_numpy(previous_controls.get_mean())
        current_inputs = self._initial_inputs.model_copy(deep=True)
        current_inputs.from_numpy(new_controls.get_mean())

        # Now, iterate through the hidden states to create ECMTransient states and update them
        output = hidden_states.copy()
        for i, hidden_array in enumerate(hidden_states):
            # Run the update on the provided state
            self.transients.from_numpy(hidden_array[:self.num_transients])
            self.asoh.update_parameters(hidden_array[self.num_transients:])
            new_transient = self.model.update_transient_state(
                previous_input=previous_inputs,
                current_input=current_inputs,
                transient_state=self.transients,
                asoh=self.asoh,
            )

            # Only the new transients (the first part) are updated
            output[i, :self.num_transients] = new_transient.to_numpy()
        return output

    @abstractmethod
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
        inputs = self._initial_inputs.model_copy(deep=True)
        inputs.from_numpy(controls.get_mean())

        # Now, iterate through hidden states to compute terminal voltage
        voltages = []
        for hidden_array in hidden_states:
            self.transients.from_numpy(hidden_array[:self.num_transients])
            self.asoh.update_parameters(hidden_array[self.num_transients:])
            ecm_out = self.model.calculate_terminal_voltage(inputs=inputs,
                                                            transient_state=self.transients,
                                                            asoh=self.asoh)
            voltages.append(ecm_out.to_numpy())

        return np.array(voltages)

"""
Collection of online estimators

Here, we include base classes to facilitate  definition of online estimators and hidden states, as well as to establish
interfaces between online estimators and models, transient states, and A-SOH parameters.
"""
from abc import abstractmethod
from typing import Union, Tuple, List, Type

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


class ModelFilterInterface():
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

    transients: GeneralContainer
    """Current estimate for the transient state of this system"""
    asoh: HealthVariable
    """Current estimate for the state of health of this system"""
    model: CellModel
    """Model used to update the hidden state"""

    # TODO (wardlt): We might be able to detect Input/Output types if CellModel was a Generic
    def __init__(self,
                 model: CellModel,
                 input_type: Type[InputQuantities],
                 output_type: Type[OutputMeasurements],
                 initial_transients: GeneralContainer,
                 initial_asoh: HealthVariable,
                 initial_inputs: InputQuantities):
        self.model = model
        self._output_type = output_type
        self.transients = initial_transients.model_copy(deep=True)
        self.asoh = initial_asoh.model_copy(deep=True)
        self._num_outputs = len(model.calculate_terminal_voltage(initial_inputs, self.transients, self.asoh))

    @property
    def num_hidden_dimensions(self) -> int:
        """ Outputs expected dimensionality of hidden state """
        return len(self.transients) + self.asoh.num_updatable

    @property
    def num_output_dimensions(self) -> int:
        """ Outputs expected dimensionality of output measurements """
        return self._num_outputs

    @abstractmethod
    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: Union[ControlVariables, List[ControlVariables]],
                             new_controls: Union[ControlVariables, List[ControlVariables]]) -> np.ndarray:
        """
        Function that updates the hidden state based on the control variables provided.

        Args:
            hidden_states: current hidden states of the system as a numpy.ndarray object
            previous_control: controls at the time the hidden states are being reported. If provided as a list, each
                entry must correspond to one of the hidden states provided. Otherwise, assumes same control for all
                hidden states.
            new_control: new controls to be used in the hidden state update. If provided as a list, each entry must
                correspond to one of the hidden states provided. Otherwise, assumes same control for all hidden states.

        Returns:
            new_hidden: updated hidden states as a numpy.ndarray object
        """
        pass

    @abstractmethod
    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: Union[ControlVariables, List[ControlVariables]]) -> np.ndarray:
        """
        Function to predict measurement from the hidden state

        Args:
            hidden_states: current hidden states of the system as a numpy.ndarray object
            controls: controls to be used for predicting outputs

        Returns:
            pred_measurement: predicted measurements as a numpy.ndarray object
        """
        pass

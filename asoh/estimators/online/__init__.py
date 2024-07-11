"""
Collection of online estimators

Here, we include base classes to facilitate  definition of online estimators and hidden states, as well as to establish
interfaces between online estimators and models, transient states, and A-SOH parameters.
"""
from abc import abstractmethod

import numpy as np
from pydantic import BaseModel, Field

from asoh.models.base import CellModel


class MultivariateRandomDistribution(BaseModel, arbitrary_types_allowed=True):
    """
    Base class to help represent a multivariate random variable
    """

    @property
    def expected_value(self) -> np.ndarray:
        """
        Provides expected value of distribution
        """
        if hasattr(self, 'mean'):
            return getattr(self, 'mean')
        raise ValueError('Mean not defined nor calculated! Expected value calculation not defined!')


class HiddenState(MultivariateRandomDistribution):
    """
    Defines the hidden state that is updated by the online estimator.
    """
    mean: np.ndarray = Field(default=None,
                             description='Mean of the random distribution that describes the hidden state')


class OutputMeasurements(MultivariateRandomDistribution):
    """
    Defines a container for the outputs
    """
    mean: np.ndarray = Field(default=None,
                             description='Mean of the random distribution that describes the output measurement')


class ControlVariables(MultivariateRandomDistribution):
    """
    Define the container for the controls. We are setting as a random variable, but, for most purposes, its probability
    distribution is to be considered a delta function centered on the mean.
    """
    mean: np.ndarray = Field(default=None,
                             description='Mean of the random distribution that describes the control variables')


class OnlineEstimator():
    """
    Defines the base structure of an online estimator
    """

    @abstractmethod
    def step(self, u: ControlVariables, y: OutputMeasurements) -> HiddenState:
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
    Defines the interface between the cell model and the online estimators. Communication between these is established
    through the use of numpy.ndarray objects.

    Args:
        cell_model: CellModel object that knows how to update transient states from A-SOH and inputs, and knows how to
            calculate outputs from A-SOH, inputs, and transient state
    """

    cell_model: CellModel

    def __init__(self, cell_model: CellModel):
        self.cell_model = cell_model

    @property
    @abstractmethod
    def num_hidden_dimensions(self) -> int:
        """ Outputs expected dimensionality of hidden state """
        pass

    @property
    @abstractmethod
    def num_output_dimensions(self) -> int:
        """ Outputs expected dimensionality of output measurements """
        pass

    @abstractmethod
    def update_hidden_states(self, hidden_state: HiddenState) -> HiddenState:
        """
        Function that updates the hidden state.

        Args:
            hidden_state: current hidden state of the system

        Returns:
            new_hidden: updated hidden state
        """
        pass

    @abstractmethod
    def predict_measurement(self, hidden_state: HiddenState) -> OutputMeasurements:
        """
        Function to predict measurement from the hidden state

        Args:
            hidden_state: current hidden state of the system

        Returns:
            pred_measurement: predicted measurement
        """
        pass

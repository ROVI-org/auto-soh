"""
Base classes to facilitate  definition of online estimators and hidden states, as well as to establish interfaces
between online estimators and models, transient states, and A-SOH parameters.
"""
from abc import abstractmethod

import numpy as np
from pydantic import BaseModel

from asoh.models.base import CellModel


class MultivariateRandomVariable(BaseModel, arbitrary_types_allowed=True):
    """
    Base class to help represent a multivariate random variable
    """

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        """
        Provides mean of distribution
        """
        pass


class HiddenState(MultivariateRandomVariable):
    """
    Defines the hidden state that is updated by the online estimator.
    """
    pass


class OutputMeasurements(MultivariateRandomVariable):
    """
    Defines a container for the outputs
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

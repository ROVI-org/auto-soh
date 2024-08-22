"""Interface for the models used by filters to describe the underlying physics"""
from abc import abstractmethod

import numpy as np


# TODO (wardlt): Move this whole module to the ./filters/ package?
class ModelWrapper():
    """
    Base class that dictates how a model has to be wrapped to interface with the filters

    All inputs, whether hidden state or controols, are 2D arrays where the first
    dimension is the batch dimension. The batch dimension must be 1 or, if not,
    the same value as any other non-unity batch sizes for the purpose
    of `NumPy broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.
    """

    @property
    @abstractmethod
    def num_hidden_dimensions(self) -> int:
        raise NotImplementedError('Please implement in child class!')

    @property
    @abstractmethod
    def num_output_dimensions(self) -> int:
        raise NotImplementedError('Please implement in child class!')

    @abstractmethod
    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: np.ndarray,
                             new_controls: np.ndarray) -> np.ndarray:
        """
        Method to update hidden states.

        Args:
            hidden_states: numpy array of hidden states, where each row is a hidden state array
            previous_controls: numpy array corresponding to the previous controls
            new_controls: numpy array corresponding to the new controls

        Returns:
            numpy array corresponding to updated hidden states
        """
        raise NotImplementedError('Please implement in child class!')

    @abstractmethod
    def predict_measurement(self, hidden_states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """
        Method to update hidden states.

        Args:
            hidden_states: numpy array of hidden states, where each row is a hidden state array
            controls: numpy array corresponding to the controls

        Returns:
            numpy array corresponding to predicted outputs
        """
        raise NotImplementedError('Please implement in child class!')

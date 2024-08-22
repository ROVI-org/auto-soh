"""Wrappers which alter the inputs before passing data to a wrapped function"""
import numpy as np

from . import ModelWrapper


class TransformedModel(ModelWrapper):
    """
    Base class for a model which transforms input data before passing
    it to a wrapped :class:`~moirae.estimators.online.model.ModelWrapper`

    The transformation moves data between two coordinate systems:
    (1) that used by the class which calls this transformed model,
    and (2) the one used by the wrapped model.
    A :class:``TransformedModel`` provides interfaces to transform
    to and from the coordinate system of the wrapped model for
    both individual points and coordinate systems.

    Args:
        wrapped: Underlying model
    """

    def __init__(self, wrapped: ModelWrapper):
        self.wrapped = wrapped

    def transform_hidden_to_wrapped(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Produce a hidden state transformed to meet the needs of the underlying model

        Args:
            hidden_states: Hidden states to be transformed

        Returns:
            Hidden states turned into form used by underlying model
        """
        raise NotImplementedError()

    def transform_hidden_from_wrapped(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Produce a hidden state transformed into the form used by the calling class

        Args:
            hidden_states: Hidden state in the form used by the underlying model

        Returns:
            Hidden states in the form used by the calling class
        """
        raise NotImplementedError()

    def transform_covariance_to_wrapped(self, covariance: np.ndarray) -> np.ndarray:
        """
        Transform a covariance matrix into coordinate system used by the underlying model

        Args:
            covariance: Covariance matrix in the form used by the calling class
        Returns:
            Covariance in the coordinate system used by the underlying model
        """
        raise NotImplementedError()

    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: np.ndarray,
                             new_controls: np.ndarray) -> np.ndarray:
        for_wrapped = self.transform_hidden_to_wrapped(hidden_states)
        from_wrapped = self.wrapped.update_hidden_states(for_wrapped, previous_controls, new_controls)
        return self.transform_hidden_from_wrapped(from_wrapped)

    def predict_measurement(self, hidden_states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        for_wrapped = self.transform_hidden_to_wrapped(hidden_states)
        return self.wrapped.predict_measurement(for_wrapped, controls)

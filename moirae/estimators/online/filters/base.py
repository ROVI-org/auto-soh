""" Base definitions for all filters """
from abc import abstractmethod
from typing import Tuple, Optional, TypedDict
from typing_extensions import NotRequired, Self

import numpy as np

from .distributions import MultivariateRandomDistribution
from .conversions import ConversionOperator, IdentityConversionOperator


class ModelWrapperConverters(TypedDict):
    hidden_conversion_operator: NotRequired[ConversionOperator]
    control_conversion_operator: NotRequired[ConversionOperator]
    output_conversion_operator: NotRequired[ConversionOperator]

    @classmethod
    def defaults(cls) -> Self:
        return {'hidden_conversion_operator': IdentityConversionOperator(),
                'control_conversion_operator': IdentityConversionOperator(),
                'output_conversion_operator': IdentityConversionOperator()}


class ModelWrapper():
    """
    Base class that dictates how a model has to be wrapped to interface with the filters

    All inputs, whether hidden state or controls, are 2D arrays where the first
    dimension is the batch dimension. The batch dimension must be 1 or, if not,
    the same value as any other non-unity batch sizes for the purpose
    of `NumPy broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

    Args:
        hidden_conversion_operator: operator that determines how the filter hidden states must be converted to be given
            to the model
        control_conversion_operator: operator that determines how filter controls must be converted to be given to the
            model
        output_conversion_operator: operator that determines how model outputs must be converted to be given to the
            filter
    """
    def __init__(self,
                 hidden_conversion_operator: Optional[ConversionOperator] = IdentityConversionOperator(),
                 control_conversion_operator: Optional[ConversionOperator] = IdentityConversionOperator(),
                 output_conversion_operator: Optional[ConversionOperator] = IdentityConversionOperator()) -> None:
        self._hidden_conversion = hidden_conversion_operator.model_copy(deep=True)
        self._control_conversion = control_conversion_operator.model_copy(deep=True)
        self._output_conversion = output_conversion_operator.model_copy(deep=True)

    def _convert_from_hidden_samples(self, filter_hidden_samples: np.ndarray) -> np.ndarray:
        """
        Auxiliary function to convert array of hidden samples from filter to model language.

        Args:
            hidden_samples: samples of hidden state to be converted
        """
        return self._hidden_conversion.transform_samples(samples=filter_hidden_samples)

    def _convert_from_control_samples(self, filter_control_samples: np.ndarray) -> np.ndarray:
        """
        Auxiliary function to convert array of control samples from filter to model language.

        Args:
            control_samples: samples of hidden state to be converted
        """
        return self._control_conversion.transform_samples(samples=filter_control_samples)

    def _convert_from_output_samples(self, filter_output_samples: np.ndarray) -> np.ndarray:
        """
        Auxiliary function to convert array of output samples from filter to model language.

        Args:
            output_samples: samples of hidden state to be converted
        """
        return self._output_conversion.transform_samples(samples=filter_output_samples)

    def _convert_to_hidden_samples(self, model_hidden_samples: np.ndarray) -> np.ndarray:
        """
        Auxiliary function to convert array of hidden samples from model to filter language.

        Args:
            hidden_samples: samples of hidden state to be converted
        """
        return self._hidden_conversion.inverse_transform_samples(samples=model_hidden_samples)

    def _convert_to_control_samples(self, model_control_samples: np.ndarray) -> np.ndarray:
        """
        Auxiliary function to convert array of control samples from model to filter language.

        Args:
            control_samples: samples of hidden state to be converted
        """
        return self._control_conversion.inverse_transform_samples(samples=model_control_samples)

    def _convert_to_output_samples(self, model_output_samples: np.ndarray) -> np.ndarray:
        """
        Auxiliary function to convert array  output samples from model to filter language.

        Args:
            output_samples: samples of hidden state to be converted
        """
        return self._output_conversion.inverse_transform_samples(samples=model_output_samples)

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


class BaseFilter:
    """
    Args:
        model: model that describes how to
            updated_hidden_state as a function of control
            calculate_output as a function of hidden state and control
        initial_hidden: initial hidden state of the system
        initial_controls: initial control state of the system
    """

    def __init__(self,
                 model: ModelWrapper,
                 initial_hidden: MultivariateRandomDistribution,
                 initial_controls: MultivariateRandomDistribution) -> None:
        self.model = model
        self.hidden = initial_hidden.model_copy(deep=True)
        self.controls = initial_controls.model_copy(deep=True)

    @abstractmethod
    def step(self,
             new_controls: MultivariateRandomDistribution,
             measurements: MultivariateRandomDistribution
             ) -> Tuple[MultivariateRandomDistribution, MultivariateRandomDistribution]:
        """
        Function to step the filter.

        Args:
            new_controls: new control state
            measurements: measurement obtained

        Returns:
            hidden_estimate: estimate of the hidden state
            output_prediction: predicted output, which is what the filter predicted the 'measurements' argument to be
        """
        raise NotImplementedError('Please implement in child class!')

""" Base definitions for all filters """
from abc import abstractmethod
from typing import Tuple

from .distributions import MultivariateRandomDistribution
from ..model import ModelWrapper


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

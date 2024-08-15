"""
Estimators which incrementally adjust estimates for system state.
"""
from abc import abstractmethod
from functools import cached_property
from typing import Tuple, Union, Collection, Optional

from moirae.estimators.online.filters.distributions import MultivariateRandomDistribution

from moirae.models.base import CellModel, GeneralContainer, InputQuantities, HealthVariable, OutputQuantities


class OnlineEstimator:
    """
    The interface for all online estimation frameworks

    Args:
        cell_model: Cell model used to describe the underlying physics of the storage system
        initial_asoh: Initial estimates for the health parameters of the battery, those being estimated or not
        initial_transients: Initial estimates for the transient states of the battery
        initial_inputs: Initial inputs to the system
        updatable_asoh: Whether to estimate values for all updatable parameters (``True``),
            none of the updatable parameters (``False``),
            or a list of specific parameters.
    """

    def __init__(self,
                 cell_model: CellModel,
                 initial_asoh: HealthVariable,
                 initial_transients: GeneralContainer,
                 initial_inputs: InputQuantities,
                 updatable_asoh: Union[bool, Collection[str]] = True):
        self.cell_model = cell_model
        self.asoh = initial_asoh.model_copy(deep=True)
        self.transients = initial_transients.model_copy(deep=True)
        self._inputs = initial_inputs.model_copy(deep=True)

        # Cache information about the outputs
        example_outputs = cell_model.calculate_terminal_voltage(initial_inputs, self.transients, self.asoh)
        self._num_outputs = len(example_outputs)
        self._output_names = example_outputs.all_names

        # The batch size of the two components must be 1
        assert self.transients.batch_size == 1
        assert self.asoh.batch_size == 1

        # Determine which parameters to treat as updatable in the ASOH
        self._updatable_names: Optional[list[str]]
        if isinstance(updatable_asoh, bool):
            if updatable_asoh:
                self._updatable_names = self.asoh.updatable_names
            else:
                self._updatable_names = []
        else:
            self._updatable_names = list(updatable_asoh)

    @cached_property
    def num_state_dimensions(self) -> int:
        """ Dimensionality of the state """
        return len(self.state_names)

    @property
    def num_output_dimensions(self) -> int:
        """ Expected dimensionality of output measurements """
        return self._num_outputs

    @cached_property
    def state_names(self) -> Tuple[str, ...]:
        """ Names of each state variable """
        # trans_names = tuple([self.transients.all_names[s] for s in self._updatable_transients])
        trans_names = self.transients.all_names
        return trans_names + self.asoh.expand_names(self._updatable_names)

    @cached_property
    def output_names(self) -> Tuple[str, ...]:
        """ Names of each output variable """
        return self._output_names

    @cached_property
    def control_names(self) -> Tuple[str, ...]:
        """ Names for each of the control variables """
        return self._inputs.all_names

    @property
    def state(self) -> MultivariateRandomDistribution:
        """Multivariate probability distribution for all state variables"""
        raise NotImplementedError()

    @abstractmethod
    def get_estimated_state(self) -> Tuple[GeneralContainer, HealthVariable]:
        """
        Compute current estimator for the transient states and ASOH

        Returns:
            - Estimate for the transient state
            - Estimator for the ASOH
        """
        raise NotImplementedError()

    @abstractmethod
    def step(self, inputs: InputQuantities, measurements: OutputQuantities) -> \
            Tuple[MultivariateRandomDistribution, MultivariateRandomDistribution]:
        """Function to step the estimator, provided new control variables and output measurements.

        Args:
            inputs: control variables
            measurements: output measurements

        Returns:
            - Updated estimate of the hidden state, which includes only the variables defined
              in :attr:`state_names`
            - Estimate of the measurements as predicted by the underlying model
        """
        raise NotImplementedError()

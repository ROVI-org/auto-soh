"""Base class for all loss function.

In separate file to make easily findable and as protection against circular imports"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from battdat.data import BatteryDataset
from moirae.models.base import CellModel, HealthVariable, GeneralContainer


# TODO (wardlt): Add degradation model after we decide how to handle its parameters
@dataclass
class BaseLoss:
    """
    Base class for objective functions which evaluate the ability of a set
    of battery health parameters to explain the observed performance data.

    All Loss classes should follow the convention that better sets of
    parameters yield values which are less than worse parameters.
    There are no constraints on whether the values need to be positive or negative.

    Args:
        cell_model: Model that describes battery physics
        asoh: Initial guesses for ASOH parameter values
        transient_state: Initial guesses for transient state
        observations: Observations of battery performance
    """

    cell_model: CellModel
    """Cell model used to compute """
    asoh: HealthVariable
    """Initial guess for battery health"""
    transient_state: GeneralContainer
    """Initial guess for battery transient state"""

    def __post_init__(self):
        self.asoh = self.asoh.model_copy(deep=True)
        self.transient_state = self.transient_state.model_copy(deep=True)

    @cached_property
    def num_states(self) -> int:
        """Number of output variables which correspond to transient states"""
        return len(self.transient_state)

    def x_to_state(self, x: np.ndarray, inplace: bool = True) -> tuple[GeneralContainer, HealthVariable]:
        """Copy batch of parameters into ASOH and state classes

        Args:
            x: Batch of parameters
            inplace: Whether to edit the copies of ASOH and state held by the loss function,
                or return a copy
        """
        states = x[:, :self.num_states]
        asoh = x[:, self.num_states:]

        if inplace:
            self.transient_state.from_numpy(states)
            self.asoh.update_parameters(asoh)
            return self.transient_state, self.asoh
        else:
            return self.transient_state.make_copy(states), self.asoh.make_copy(asoh)

    # TODO (wardlt): Consider passing the ASOH parameters through somewhere else
    def get_x0(self) -> np.ndarray:
        """Generate an initial guess

        Returns:
            A 1D vector used as a starting point for class to this class
        """
        if self.asoh.num_updatable == 0:
            return self.transient_state.to_numpy()[0, :]
        return np.concatenate([self.transient_state.to_numpy(),
                               self.asoh.get_parameters()], axis=1)[0, :]  # Get a 1D vector

    def __call__(self, x: np.ndarray, observations: BatteryDataset) -> np.ndarray:
        """
        Compute the fitness of a set of parameters to the observed data.

        Args:
            x: Batch of parameters to be evaluated (2D array)
            observations: Observed data from the battery to be used to compute loss

        Returns:
            A single measure of parameter fitness for each member of batch (1D array)
        """
        raise NotImplementedError()

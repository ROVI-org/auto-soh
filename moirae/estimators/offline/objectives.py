"""Interfaces that evaluate the fitness of a set of battery state parameters
provided as a NumPy array."""
import numpy as np

from batdata.data import BatteryDataset
from moirae.models.base import CellModel, HealthVariable, GeneralContainer


# TODO (wardlt): Add degradation model after we decide how to handle its parameters
class Objective:
    """
    Base class for objective functions which evaluate the ability of a set
    of battery health parameters to explain the observed performance data
    """

    cell_model: CellModel
    """Cell model used to compute """
    asoh: HealthVariable
    """Initial guess for battery health"""
    state: GeneralContainer
    """Initial guess for battery state"""
    observations: BatteryDataset
    """Observed data for the battery performance"""

    def __init__(self,
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 state: GeneralContainer,
                 observations: BatteryDataset):
        self.cell_model = cell_model
        self.asoh = asoh
        self.state = state
        self.observations = observations

    def get_x0(self) -> np.ndarray:
        """
        Prepare an initial guess for an optimizer

        Returns:
            The state and ASOH parameters being fit
        """

        if self.asoh.num_updatable == 0:
            return self.state.to_numpy()[0, :]
        return np.concatenate([self.state.to_numpy(), self.asoh.get_parameters()], axis=1)[0, :]  # Get a 1D vector

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the fitness of a set of parameters to the observed data.

        Args:
            x: Batch of parameters to be evaluated (2D array)

        Returns:
            A single measure of parameter fitness for each member of batch (1D array)
        """
        raise NotImplementedError()


# TODO (wardlt): Generalize to other outputs when we have them
class MeanSquaredLoss(Objective):
    """
    Score the fitness of a set of health parameters by the mean squared error
    between observed and predicted terminal voltage.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Translate input parameters to state and ASOH parameters
        n_states = len(self.state)
        state_x = self.state.make_copy(x[:, :n_states])
        asoh_x = self.asoh.make_copy(x[:, n_states:])

"""Base class for an offline estimator"""
from typing import Optional, Any

import numpy as np

from moirae.estimators.offline.objectives import Objective
from moirae.models.base import GeneralContainer, HealthVariable


class OfflineEstimator:
    """
    Base class for tools which estimate battery health parameters given
    many measurements of the battery performance over time.

    Create the class by passing a fully-configured :meth:`~moirae.estimators.offline.objectives.Objective`
    then perform the estimation using the :meth:`estimate` function.
    """

    objective: Objective
    """Function being optimized"""

    def _get_x0(self,
                state_0: Optional[GeneralContainer],
                asoh_0: Optional[HealthVariable]) -> np.ndarray:
        asoh = self.objective.asoh if asoh_0 is None else asoh_0
        state = self.objective.state if state_0 is None else state_0
        if asoh.num_updatable == 0:
            return state.to_numpy()[0, :]
        return np.concatenate([state.to_numpy(), asoh.get_parameters()], axis=1)[0, :]  # Get a 1D vector

    def estimate(self,
                 state_0: Optional[GeneralContainer] = None,
                 asoh_0: Optional[HealthVariable] = None) -> tuple[GeneralContainer, HealthVariable, Any]:
        """
        Compute an estimate for the initial state and ASOH.

        Args:
            state_0: Initial guess for the transient state. Uses the state provided in :attr:`objective` as default
            asoh_0: Initial guess for the ASOH. Uses the state provided in :attr:`objective` as default

        Returns:
            - Estimate for the initial state
            - Estimate for the ASOH parameters
            - Diagnostic unique to the type of estimator
        """
        raise NotImplementedError()

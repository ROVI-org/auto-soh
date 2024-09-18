"""Base class for an offline estimator"""
from typing import Any

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

    def estimate(self) -> tuple[GeneralContainer, HealthVariable, Any]:
        """
        Compute an estimate for the initial state and ASOH.

        Returns:
            - Estimate for the initial state
            - Estimate for the ASOH parameters
            - Diagnostic unique to the type of estimator
        """
        raise NotImplementedError()

"""Tools which estimate parameters of battery health and performance after-the-fact."""
from typing import Any

from moirae.estimators.offline.loss import BaseLoss
from moirae.models.base import GeneralContainer, HealthVariable


class OfflineEstimator:
    """
    Base class for tools which estimate battery health parameters given
    many measurements of the battery performance over time.

    Create the class by passing a fully-configured :meth:`~moirae.estimators.offline.loss.Loss`
    then perform the estimation using the :meth:`estimate` function.
    """

    loss: BaseLoss
    """Function being minimized"""

    def estimate(self) -> tuple[GeneralContainer, HealthVariable, Any]:
        """
        Compute an estimate for the initial state and ASOH.

        Returns:
            - Estimate for the initial state
            - Estimate for the ASOH parameters
            - Diagnostic unique to the type of estimator
        """
        raise NotImplementedError()

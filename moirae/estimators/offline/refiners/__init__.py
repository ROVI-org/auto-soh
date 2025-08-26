"""
Tools which refine existing estimates of battery health and performance
"""
from typing import Any

from moirae.estimators.offline.refiners.loss import BaseLoss
from moirae.models.base import GeneralContainer, HealthVariable


# TODO (wardlt): Make it possible to define bounds for classes
# TODO (wardlt): Consider letting users pass x0 in as an input
class Refiner:
    """
    Base class for tools which refine battery health parameters given
    many measurements of the battery performance over time.

    Create the class by passing a fully-configured :meth:`~moirae.estimators.offline.refiners.loss.Loss`
    then perform the estimation using the :meth:`refine` function.
    """

    loss: BaseLoss
    """Function being minimized, which contains initial guesses"""

    def refine(self) -> tuple[GeneralContainer, HealthVariable, Any]:
        """
        Compute an estimate for the initial state and ASOH.

        Returns:
            - Estimate for the initial state
            - Estimate for the ASOH parameters
            - Diagnostic unique to the type of estimator
        """
        raise NotImplementedError()

"""Estimate the state of health using SciPy optimizers"""
from typing import Optional

from scipy.optimize import minimize, OptimizeResult

from moirae.estimators.offline.base import OfflineEstimator
from moirae.estimators.offline.objectives import Objective
from moirae.models.base import GeneralContainer, HealthVariable


class ScipyMinimizer(OfflineEstimator):
    """Estimate using SciPy's
    `minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ function.

    Args:
        objective: Objective function to be optimized
        kwargs: Passed to the minimize function
    """

    def __init__(self,
                 objective: Objective,
                 **kwargs):
        self.objective = objective
        self.scipy_kwargs = kwargs

    def estimate(self,
                 state_0: Optional[GeneralContainer] = None,
                 asoh_0: Optional[HealthVariable] = None) \
            -> tuple[GeneralContainer, HealthVariable, OptimizeResult]:
        # Get the scale of the initial error, used to normalize the output
        x0 = self._get_x0(state_0, asoh_0)
        y0 = self.objective(x0[None, :]).item()  # Should be a scalar

        # Assemble the function call
        result = minimize(
            fun=lambda x: self.objective(x[None, :]).item() / max(y0, 1e-12),
            x0=x0,
            **self.scipy_kwargs
        )

        # Assemble the output
        num_states = len(self.objective.state)

        states = self.objective.state.make_copy(result.x[None, :num_states])
        asoh = (self.objective.asoh if asoh_0 is None else asoh_0).make_copy(result.x[None, num_states:])
        return states, asoh, result

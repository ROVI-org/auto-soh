"""Estimate the state of health using SciPy optimizers"""
from scipy.optimize import minimize, OptimizeResult

from moirae.estimators.offline import OfflineEstimator
from moirae.estimators.offline.loss import BaseLoss
from moirae.models.base import GeneralContainer, HealthVariable


class ScipyMinimizer(OfflineEstimator):
    """Estimate using SciPy's minimize_ function.

    Args:
        objective: Objective function to be optimized
        kwargs: Passed to the minimize function. Refer to the documentation to minimize_

    .. _minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    def __init__(self,
                 objective: BaseLoss,
                 **kwargs):
        self.objective = objective
        self.scipy_kwargs = kwargs

    def estimate(self) -> tuple[GeneralContainer, HealthVariable, OptimizeResult]:
        # Get the scale of the initial error, used to normalize the output
        x0 = self.objective.get_x0()
        y0 = self.objective(x0[None, :]).item()   # Used to normalize scale and reduce rtol vs atol issues

        # Assemble the function call
        result = minimize(
            fun=lambda x: self.objective(x[None, :]).item() / max(abs(y0), 1e-12),
            x0=x0,
            **self.scipy_kwargs
        )

        # Assemble the output
        num_states = len(self.objective.state)
        states = self.objective.state.make_copy(result.x[None, :num_states])
        asoh = self.objective.asoh.make_copy(result.x[None, num_states:])
        return states, asoh, result

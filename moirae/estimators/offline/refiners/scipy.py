"""Estimate the state of health using SciPy optimizers"""
from scipy.optimize import differential_evolution, minimize, OptimizeResult, Bounds

from battdat.data import BatteryDataset

from moirae.estimators.offline.refiners import Refiner
from moirae.estimators.offline.refiners.loss import BaseLoss
from moirae.models.base import GeneralContainer, HealthVariable


class ScipyDifferentialEvolution(Refiner):
    """Estimate using SciPy's differential_evolution_ function.

    Args:
        objective: Objective function to be optimized
        bounds: Bounds for variables. There are two ways to specify the bounds: instance of scipy
            Bounds class, or (min, max) pairs for each element in x, defining the finite lower and
            upper bounds for the optimizing argument of func.
        kwargs: Passed to the minimize function.
            Refer to the documentation for differential_evolution_

    .. _differential_evolution:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    """

    def __init__(self,
                 objective: BaseLoss,
                 bounds: Bounds | list[tuple[float, float]],
                 **kwargs):
        self.objective = objective
        self.bounds = bounds
        self.scipy_kwargs = kwargs

    def refine(self, observations: BatteryDataset) -> tuple[GeneralContainer, HealthVariable, OptimizeResult]:
        # Get the scale of the initial error, used to normalize the output
        x0 = self.objective.get_x0()
        y0 = self.objective(x0[None, :], observations).item()  # Used to normalize scale and reduce rtol vs atol issues

        # Assemble the function call
        result = differential_evolution(
            func=lambda x: self.objective(x[None, :], observations).item() / max(abs(y0), 1e-12),
            bounds=self.bounds,
            x0=x0,
            **self.scipy_kwargs
        )

        # Assemble the output
        states, asoh = self.objective.x_to_state(result.x[None, :], inplace=False)
        return states, asoh, result


class ScipyMinimizer(Refiner):
    """Estimate using SciPy's minimize_ function.

    Args:
        objective: Objective function to be optimized
        kwargs: Passed to the minimize function. Refer to the documentation of minimize_

    .. _minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    def __init__(self,
                 objective: BaseLoss,
                 **kwargs):
        self.objective = objective
        self.scipy_kwargs = kwargs

    def refine(self, observations: BatteryDataset) -> tuple[GeneralContainer, HealthVariable, OptimizeResult]:
        # Get the scale of the initial error, used to normalize the output
        x0 = self.objective.get_x0()
        y0 = self.objective(x0[None, :], observations).item()  # Used to normalize scale and reduce rtol vs atol issues

        # Assemble the function call
        result = minimize(
            fun=lambda x: self.objective(x[None, :], observations).item() / max(abs(y0), 1e-12),
            x0=x0,
            **self.scipy_kwargs
        )

        # Assemble the output
        states, asoh = self.objective.x_to_state(result.x[None, :], inplace=False)
        return states, asoh, result

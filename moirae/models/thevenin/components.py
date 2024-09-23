"""Classes used to define components of the ASOH"""
from typing import Callable

from numpy.polynomial.polynomial import polyval

from moirae.models.base import HealthVariable, ListParameter, ScalarParameter


class SOCDependentVariable(HealthVariable, Callable[[float, int], float]):
    """A health variable which is dependent on the state of charge"""

    def __call__(self, soc: float, batch_id: int = 0) -> float:
        """
        Evaluate the value of the parameter at a specific state of charge

        Args:
            soc: State of charge for the battery
            batch_id: Which parameter set for the batch to use
        Returns:
            Value of the parameter
        """
        raise NotImplementedError()


class SOCPolynomialVariable(SOCDependentVariable):
    """A parameter whose dependence on SOC is described by a polynomial"""

    coeffs: ListParameter = 1.
    """Coefficients for the polynomial"""

    def __call__(self, soc: float, batch_id: int = 0) -> float:
        coeffs = self.coeffs[batch_id % self.batch_size, :]
        return polyval(soc, coeffs).item()


class SOCTempDependentVariable(HealthVariable, Callable[[float, float, int], float]):
    """A health variable which is dependent on the state of charge and temperature"""

    def __call__(self, soc: float, temp: float, batch_id: int = 0) -> float:
        """
        Evaluate the value of the parameter at a specific state of charge and temperature

        Args:
            soc: State of charge for the battery
            temp: Temperature of the battery (units: K)
            batch_id: Which parameter set for the batch to use

        Returns:
            Value under the specified conditions
        """
        raise NotImplementedError()


class SOCTempPolynomialVariable(SOCTempDependentVariable):
    """A parameter where the dependence on SOC and temperature are described by polynomial

    The temperature-dependence is described by a polynomial centered on a reference temperature,
    :math:`f(T) = c_0 + c_1 (T - T_{ref}) + ...`
    """

    # TODO (wardlt): We define a constant parameter in both polynomial. Maybe set one to be the linear terms.
    t_ref: ScalarParameter = 298.
    """Reference temperature for the temperature dependence. Units: K"""
    soc_coeffs: ListParameter = 1.
    """Reference parameters for the OCV dependence polynomial"""
    t_coeffs: ListParameter = 0.
    """Reference parameters for the temperature dependence polynomial"""

    def __call__(self, soc: float, temp: float, batch_id: int = 0) -> float:
        t_corr = temp - self.t_ref[batch_id % self.t_ref.shape[0], 0]
        return (
                polyval(t_corr, self.t_coeffs[batch_id % self.t_coeffs.shape[0], :]) +
                polyval(soc, self.soc_coeffs[batch_id % self.soc_coeffs.shape[0], :])
        )

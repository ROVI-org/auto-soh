"""Models for parameters which vary as a function of SOC"""
from abc import ABCMeta, abstractmethod

from numpy.polynomial.polynomial import polyval
import numpy as np

from moirae.models.base import HealthVariable, ScalarParameter, ListParameter
from .soc import adjust_soc_shape, FloatOrArray


class SOCTempDependentHealth(HealthVariable, metaclass=ABCMeta):
    """Interface for variables whose values depend on the state of charge"""

    @abstractmethod
    def get_value(self, soc: FloatOrArray, temp: FloatOrArray, batch_id: int | None = None) -> np.ndarray:
        """
        Computes value(s) at given SOC(s).

        Args:
            soc: Values at which to compute the property. Dimensions must be either

                1. a 2D array of shape `(batch_size, soc_dim)`. The `batch_size` must either be 1 or
                   equal to the batch size fo the parameters.
                2. a 1D array of shape `(soc_dim,)`, in which case we will consider the `batch_size` to be equal to 1
                3. a 0D array (a scalar), in which case both `batch_size` and `soc_dim` are equal to 1.
            temp: Temperatures at which to evaluate the property. (Units: °C)
                 Dimensions follow the same format as ``soc``.
            batch_id: Which batch member for the parameters and input SOC values to use. Default is to use whole batch
        Returns:
            Interpolated values as a 2D with dimensions (batch_size, soc_points).
            The ``batch_size`` is 0 when ``batch_id`` is not None.
        """
        raise NotImplementedError()


class SOCTempPolynomialHealth(SOCTempDependentHealth):
    """A parameter where the dependence on SOC and temperature are described by polynomial

    The temperature-dependence is described by a polynomial centered on a reference temperature,
    :math:`f_T(T) = c_0 + c_1 (T - T_{ref}) + ...`

    The SOC-dependence is described by a polynomial as well,
    :math:`f_{SOC}(soc) = c_0 + c_1 * soc + ...`

    The two are added to express dependence in both: :math:`f(T, SOC) = f_T(T) + f_{SOC}(SOC)`
    """

    # TODO (wardlt): We define a constant parameter in both polynomials. Maybe set one have an assumed constant of zero
    t_ref: ScalarParameter = 25.
    """Reference temperature for the temperature dependence. Units: °C"""
    soc_coeffs: ListParameter = 1.
    """Reference parameters for the OCV dependence polynomial"""
    t_coeffs: ListParameter = 0.
    """Reference parameters for the temperature dependence polynomial"""

    def get_value(self, soc: FloatOrArray, temp: FloatOrArray, batch_id: int | None = None) -> np.ndarray:
        soc = adjust_soc_shape(soc)
        temp = adjust_soc_shape(temp)

        # Compute the reference temperature
        if batch_id is None:
            t_corr = temp - self.t_ref[:, 0]
        else:
            t_corr = temp - self.t_ref[batch_id % self.t_ref.shape[0], 0]

        # Get the proper coefficients then interpolate
        t_coeff = self.t_coeffs if batch_id is None else self.t_coeffs[batch_id % self.t_coeffs.shape[0], :]
        s_coeff = self.soc_coeffs if batch_id is None else self.soc_coeffs[batch_id % self.soc_coeffs.shape[0], :]
        return polyval(t_corr.T, t_coeff.T, tensor=False).T + polyval(soc.T, s_coeff.T, tensor=False).T

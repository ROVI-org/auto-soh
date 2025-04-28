"""Models for parameters which vary as a function of SOC"""
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union, Literal, Callable
from numbers import Number

import numpy as np
from pydantic import Field, PrivateAttr
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval
from numpy.polynomial.legendre import Legendre

from moirae.models.base import HealthVariable, ListParameter, NumpyType

FloatOrArray = Union[Number, List[float], np.ndarray]
"""A number of list of numbers"""


def adjust_soc_shape(soc: Union[Number, List, np.ndarray]) -> np.ndarray:
    """Adjust the SOC from a user-provided shape to 2D array

    Args:
        soc: SOC, as provided by user
    Returns:
        2D array where first dimension is the batch and second is the SOC values.
    """
    soc = np.asarray(soc)  # Ensure it's an array, but don't copy it
    if soc.ndim == 0:
        return soc[None, None]
    elif soc.ndim == 1:
        return soc[None, :]
    elif soc.ndim == 2:
        return soc
    else:
        raise ValueError(f'SOC must be passed as at most a 2D array, but has shape {soc.shape}!')


class SOCDependentHealth(HealthVariable, metaclass=ABCMeta):
    """Interface for variables whose values depend on the state of charge"""

    @abstractmethod
    def get_value(self, soc: FloatOrArray, batch_id: int | None = None) -> np.ndarray:
        """
        Computes value(s) at given SOC(s).

        Args:
            soc: Values at which to compute the property. Dimensions must be either

                1. a 2D array of shape `(batch_size, soc_dim)`. The `batch_size` must either be 1 or
                   equal to the batch size fo the parameters.
                2. a 1D array of shape `(soc_dim,)`, in which case we will consider the `batch_size` to be equal to 1
                3. a 0D array (a scalar), in which case both `batch_size` and `soc_dim` are equal to 1.
            batch_id: Which batch member for the parameters and input SOC values to use. Default is to use whole batch
        Returns:
            Interpolated values as a 2D with dimensions (batch_size, soc_points).
            The ``batch_size`` is 0 when ``batch_id`` is not None.
        """
        raise NotImplementedError()


class SOCInterpolatedHealth(SOCDependentHealth):
    """Health variables where SOC dependence is described by a piecewise-polynomial"""
    base_values: ListParameter = \
        Field(default=0, description='Values at specified SOCs')
    soc_pinpoints: Optional[NumpyType] = Field(default=None, description='SOC pinpoints for interpolation.')
    interpolation_style: Literal[
        'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next'
    ] = Field(default='linear', description='Type of interpolation to perform')

    # Internal caches
    _ppoly_cache: dict[int, tuple[np.ndarray, np.ndarray, Callable]] = PrivateAttr(default_factory=dict)
    """Cache for the interpolation function, and the interpolation points/soc points at which it was produced"""

    def _get_function(self, batch_id: int) -> Callable:
        """Retrieve the interpolation function for a specific batch member

        Uses the internal cache, :attr:`_ppoly_cache`, if the :attr:`base_values` has not changed.

        Returns:
            The polynomial of interest
        """

        # Return the cached spline if it exists and the base values have not changed
        if (cached := self._ppoly_cache.get(batch_id)) is not None:
            org_soc, orig_array, ppoly = cached
            if np.array_equal(orig_array, self.base_values) \
                    and np.array_equal(org_soc, self.soc_pinpoints):
                return ppoly

            # Otherwise, clear the cache (all entries) and continue
            self._ppoly_cache.clear()

        # Generate the SOC interpolation points if they don't already exist
        if self.soc_pinpoints is None:
            self.soc_pinpoints = np.linspace(0, 1, self.base_values.shape[-1])

        # Make the spline then cache it
        func = interp1d(self.soc_pinpoints,
                        self.base_values[slice(None) if batch_id is None else batch_id % self.base_values.shape[0], :],
                        kind=self.interpolation_style,
                        bounds_error=False,
                        fill_value='extrapolate')
        if self.interpolation_style not in {'linear', 'nearest', 'nearest-up'}:  # Don't cache lazy models
            self._ppoly_cache[batch_id] = (self.soc_pinpoints.copy(), self.base_values[batch_id, :].copy(), func)
        return func

    def get_value(self, soc: FloatOrArray, batch_id: int | None = None) -> np.ndarray:
        # Determine the batch sizes for the output
        soc = adjust_soc_shape(soc)
        soc_batch_size, soc_dim = soc.shape
        internal_batch_size = self.base_values.shape[0]

        # Determine the batch size and build the interpolation functions
        if batch_id is None:
            batch_size = max(soc_batch_size, internal_batch_size)
            if min(soc_batch_size, internal_batch_size) != 1 and soc_batch_size != internal_batch_size:
                raise ValueError(f'Batch sizes of provided SOCs ({soc_batch_size}) '
                                 f'and parameters for this class ({internal_batch_size}) are inconsistent.')
        else:
            batch_size = 1

        # Special case: constant value
        if self.base_values.shape[-1] == 1:
            index = slice(None) if batch_id is None else batch_id
            y = self.base_values[index, :]
            return np.tile(y, (batch_size // y.shape[0], soc_dim))  # shape = (batch_size, soc_dim)

        # Make a single call to interpolation
        f = self._get_function(batch_id)
        if batch_id is None:
            results = f(soc)
            output = np.empty((batch_size, soc_dim))
            for b in range(batch_size):
                output[b, :] = results[b % internal_batch_size, b % soc_batch_size, :]
            return output
        else:
            return f(soc[batch_id % soc_batch_size, :])[None, :]


class ScaledSOCInterpolatedHealth(SOCInterpolatedHealth):
    """SOC interpolated health with scaling factors to adjust the shape of the curve

    Scaling factors adjust the shape produced from the interpolation by an additive
    or multiplicative factor which varies as a function of SOC.
    The scaling factor is described using
    `Legendre polynomials <https://en.m.wikipedia.org/wiki/Legendre_polynomials>`_
    defined over the region of [0, 1].
    """

    scaling_coeffs: ListParameter
    """Coefficients of a Legendre polynomial used to adjust the interpolation"""
    additive: bool = True
    """Whether to add or multiply interpolated value with the scaling factor"""

    def get_value(self, soc: FloatOrArray, batch_id: int | None = None) -> np.ndarray:
        # Evaluate the interpolated values
        soc = adjust_soc_shape(soc)
        interpolated = super().get_value(soc, batch_id)

        # Determine batch size of the output, prepare outputs
        scale_batch = self.scaling_coeffs.shape[0]
        inter_batch = self.base_values.shape[0]
        soc_batch = soc.shape[0]
        if batch_id is None:
            batch_size = max(scale_batch, inter_batch)
            if inter_batch != batch_size:
                assert inter_batch == 1, 'Inter batch should be either 1 or equal to the batch size'
                output = np.tile(interpolated, (batch_size, 1))
            else:
                output = interpolated
        else:
            output = interpolated
            batch_size = 1

        # Apply the scaling factors for each batch member
        for i, b in enumerate(range(batch_size) if batch_id is None else [batch_id]):
            # % is a shortcut which gets either the correct batch index for batched data,
            #  and 1 for data which are not batched. It works
            scaler = Legendre(coef=self.scaling_coeffs[b % scale_batch, :])
            scaling_amount = scaler(soc[b % soc_batch, :])
            if self.additive:
                output[i, :] += scaling_amount
            else:
                output[i, :] *= 1 + scaling_amount
        return output


class SOCPolynomialHealth(SOCDependentHealth):
    """A health parameter whose value is described by a power-series polynomial"""
    coeffs: ListParameter = 1.
    """Coefficients for the polynomial"""

    def get_value(self, soc: FloatOrArray, batch_id: Optional[int] = None) -> FloatOrArray:
        soc = adjust_soc_shape(soc)
        coeffs = self.coeffs if batch_id is None else self.coeffs[batch_id % self.batch_size, :]
        return polyval(soc.T, coeffs.T, tensor=False).T

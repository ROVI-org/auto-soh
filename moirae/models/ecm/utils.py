"""Functions used in multiple components"""
from typing import List, Optional, Union, Literal, Callable, Iterator
from numbers import Number

import numpy as np
from pydantic import Field, PrivateAttr
from scipy.interpolate import interp1d

from moirae.models.base import HealthVariable, ListParameter, NumpyType


class SOCInterpolatedHealth(HealthVariable):
    """Health variables which vary as a function of SOC"""
    base_values: ListParameter = \
        Field(default=0, description='Values at specified SOCs')
    soc_pinpoints: Optional[NumpyType] = Field(default=None, description='SOC pinpoints for interpolation.')
    interpolation_style: Literal['linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                                 'quadratic', 'cubic', 'previous', 'next'] = \
        Field(default='linear', description='Type of interpolation to perform')

    # Internal caches
    _ppoly_cache: tuple[np.ndarray, np.ndarray, Callable] | None = PrivateAttr(None)
    """Cache for the interpolation function, and the interpolation points/soc points at which it was produced"""

    def iter_parameters(self, updatable_only: bool = True, recurse: bool = True) -> Iterator[tuple[str, np.ndarray]]:
        for name, param in super().iter_parameters(updatable_only, recurse):
            if name != "soc_pinpoints":
                yield name, param

    def _get_function(self) -> Callable:
        """Retrieve the interpolation function

        Uses the internal cache, :attr:`_ppoly_cache`, if the :attr:`base_values` has not changed.

        Returns:
            The polynomial of interest
        """

        # Return the cached spline if it exists and the base values have not changed
        if self._ppoly_cache is not None:
            org_soc, orig_array, ppoly = self._ppoly_cache
            if np.array_equal(orig_array, self.base_values) \
                    and np.array_equal(org_soc, self.soc_pinpoints):
                return ppoly

            # Otherwise, clear the cache and continue
            self._ppoly_cache = None

        # Generate the SOC interpolation points if they don't already exist
        if self.soc_pinpoints is None:
            self.soc_pinpoints = np.linspace(0, 1, self.base_values.shape[-1])

        # Make the spline then cache it
        func = interp1d(self.soc_pinpoints,
                        self.base_values,
                        kind=self.interpolation_style,
                        bounds_error=False,
                        fill_value='extrapolate')
        if self.interpolation_style not in {'linear', 'nearest', 'nearest-up'}:  # Don't cache lazy models
            self._ppoly_cache = (self.soc_pinpoints.copy(), self.base_values.copy(), func)
        return func

    def get_value(self, soc: Union[Number, List, np.ndarray]) -> np.ndarray:
        """
        Computes value(s) at given SOC(s).

        This function always returns a 3D array, of shape `(internal_batch_size, soc_batch_size, soc_dim)`, where
        `internal_batch_size` is the batch size of the underlying health variable, `soc_batch_size` is the batch size
        of the SOC array, and `soc_dim` is the dimensionality of the SOC. The SOC must be passed as either:
        1. a 2D array of shape `(soc_batch_size, soc_dim)`
        2. a 1D array of shape `(soc_dim,)`, in which case we will consider the `soc_batch_size` to be equal to 1
        3. a 0D array (that is, a number), in which case both `soc_batch_size` and `soc_dim` are equal to 1.

        Args:
            soc: Values at which to compute the property.
        Returns:
            Interpolated values
        """
        # Determine which case we're dealing with
        soc = np.array(soc)
        soc_shape = soc.shape

        soc_batch_size = 1
        soc_dim = 1 if len(soc_shape) == 0 else soc_shape[0]  # taking care of 0 or 1D cases
        if len(soc_shape) > 3:
            raise ValueError(f'SOC must be passed as at most a 2D array, but has shape {soc_shape}!')
        if len(soc_shape) == 2:
            soc_batch_size = soc.shape[0]
            soc_dim = soc.shape[1]
        internal_batch_size = self.batch_size

        # Special case: no interpolation
        if self.base_values.shape[-1] == 1:
            y = self.base_values[:, 0].copy()[:, None]  # shape = (internal_batch_size, 1)
            y = np.tile(y, (soc_batch_size, 1, soc_dim))  # shape = (soc_batch, internal_batch, soc_dim)
            y = np.swapaxes(y, 0, 1)  # shape = (internal_batch, soc_batch, soc_dim)
        # Otherwise, run the interpolator, but the results mean something different
        else:
            y = self._get_function()(soc)  # interpolator adds batch dimension:
            # If the SOC was batched, the shape is (internal_batch, soc_batch, soc_dim)
            # Otherwise, the shape is (internal_batch, soc_dim)
            y = y.reshape((internal_batch_size, soc_batch_size, soc_dim))

        # Now, the y array has shape (internal_batch, soc_batch, soc_dim).
        return y


def realistic_fake_ocv(
        soc_vals: Union[float, np.ndarray]) -> np.ndarray:
    """
    Returns somewhat realistic OCV relationship to SOC
    """
    x_scale = 0.9
    x_off = 0.05
    y_scale = 0.1
    y_off = 3.5
    mod_soc = x_scale * soc_vals
    mod_soc += x_off
    volts = np.log(mod_soc / (1 - mod_soc))
    volts *= y_scale
    volts += y_off
    volts = volts.astype(float)
    return volts


def unrealistic_fake_r0(
        soc_vals: Union[float, np.ndarray]) -> np.ndarray:
    """
    Returns not very realistic R0 relationship to SOC
    """
    ohms = 0.05 * np.ones(np.array(soc_vals).shape)
    return ohms


def hysteresis_solver_const_sign(
        h0: Union[float, np.ndarray],
        M: Union[float, np.ndarray],
        kappa: Union[float, np.ndarray],
        dt: Union[float, np.ndarray],
        i0: Union[float, np.ndarray],
        alpha: Union[float, np.ndarray]
) -> float:
    """
    Helper function to solve for hysteresis at time dt given initial conditions,
    parameters, and current and current slope. Assumes current does not change
    sign during time interval

    Args:
        h0: Initial value of hysteresis, corresponding to h[0]
        M: Asymptotic value of hysteresis at present condition (the value h[t] should approach)
        kappa: Constant representing the product of gamma (SOC-based rate at which hysteresis approaches M),
            Coulombic efficienty, and 1/Qt
        dt: Length of time interval
        i0: Initial current
        alpha: Slope of current profile during time interval

    Returns:
        Hysteresis value at the end of the time interval
    """
    assert i0 * (i0 + (alpha * dt)) >= 0, 'Current flips sign in interval dt!!'
    exp_factor = kappa * dt  # shape (broadcasted_batch_size, 1)
    exp_factor = exp_factor * (i0 + (0.5 * alpha * dt))
    # Now, flip the sign depending if current is positive in the interval
    if i0 > -(alpha * dt):  # this indicates (i0 + alpha * t) > 0
        exp_factor = -exp_factor
    exp_factor = np.exp(exp_factor)
    h_dt = exp_factor * h0
    h_dt = h_dt + ((1 - exp_factor) * M)
    return h_dt

from typing import List, Optional, Union, Literal, Callable, Iterator
from numbers import Number

import numpy as np
from pydantic import Field, field_serializer
from scipy.interpolate import interp1d

from moirae.models.base import HealthVariable, ListParameter


class SOCInterpolatedHealth(HealthVariable):
    """Defines basic functionality for HealthVariables that need interpolation between SOC pinpoints
    """
    base_values: ListParameter = \
        Field(default=0, description='Values at specified SOCs')
    soc_pinpoints: Optional[np.ndarray] = Field(default=None, description='SOC pinpoints for interpolation.')
    interpolation_style: Literal['linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                                 'quadratic', 'cubic', 'previous', 'next'] = \
        Field(default='linear', description='Type of interpolation to perform')

    @field_serializer('soc_pinpoints', when_used='json-unless-none')
    def _serialize_numpy(self, value: np.ndarray):
        return value.tolist()

    def iter_parameters(self, updatable_only: bool = True, recurse: bool = True) -> Iterator[tuple[str, np.ndarray]]:
        for name, param in super().iter_parameters(updatable_only, recurse):
            if name != "soc_pinpoints":
                yield name, param

    @property
    def _interp_func(self) -> Callable:
        """
        Interpolate values. If soc_pinpoints have not been set, assume
        internal_parameters are evenly spread on an SOC interval [0,1].
        """
        if self.soc_pinpoints is None:
            self.soc_pinpoints = np.linspace(0, 1, self.base_values.shape[-1])
        func = interp1d(self.soc_pinpoints,
                        self.base_values,
                        kind=self.interpolation_style,
                        bounds_error=False,
                        fill_value='extrapolate')
        return func

    def get_value(self, soc: Union[Number, List, np.ndarray]) -> np.ndarray:
        """Computes value(s) at given SOC(s)

        Args:
            soc: Values at which to compute the property
        Returns:
            Interpolated values
        """
        # Determine which case we're dealing with
        batch_size = self.batch_size
        soc = np.array(soc)
        input_dims = soc.shape
        soc_batch_size = soc.size

        # Special case: no interpolation
        if self.base_values.shape[-1] == 1:
            y = self.base_values[:, 0].copy()
            if soc_batch_size > 0 and batch_size == 1:
                return np.repeat(y, soc.size, axis=0).reshape(input_dims)
            elif soc_batch_size == 1 and batch_size > 1:
                if len(input_dims) > 0:
                    return y.reshape((batch_size, 1))
                else:
                    return y.reshape((batch_size,))
            return y.reshape(input_dims)

        # Run the interpolator, but the results mean something different
        y = self._interp_func(soc)  # interpolator adds a dimension
        if soc_batch_size > 1 and batch_size > 1:
            y = np.diag(y.squeeze())  # Match the SOC with the model its calling
        elif soc_batch_size == 1 and batch_size > 1:
            return y.reshape((batch_size,) + input_dims)
        return y.reshape(input_dims)


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


def hysteresis_solver_const_sign(
        h0: Union[float, np.ndarray],
        M: Union[float, np.ndarray],
        kappa: Union[float, np.ndarray],
        dt: Union[float, np.ndarray],
        i0: Union[float, np.ndarray],
        alpha: Union[float, np.ndarray]) -> float:
    """
    Helper function to solve for hysteresis at time dt given initial conditions,
    parameters, and current and current slope. Assumes current does not change
    sign during time interval

    Arguments
    ---------
    h0: float
        Initial value of hysteresis, corresponding to h[0]
    M: float
        Asymptotic value of hysteresis at present condition (the value h[t]
        should approach)
    kappa: float
        Constant representing the product of gamma (SOC-based rate at which
        hysteresis approaches M), Coulombic efficienty, and 1/Qt
    dt: float
        Length of time interval
    i0: float
        Initial current
    alpha:
        Slope of current profile during time interval

    Outputs
    -------
    h_dt: float
        Hysteresis value at the end of the time interval
    """
    assert i0 * (i0 + (alpha * dt)) >= 0, 'Current flips sign in interval dt!!'
    exp_factor = kappa * dt
    exp_factor = exp_factor * (i0 + (0.5 * alpha * dt))
    # Now, flip the sign depending if current is positive in the interval
    if i0 > -(alpha * dt):  # this indicates (i0 + alpha * t) > 0
        exp_factor = -exp_factor
    exp_factor = np.exp(exp_factor)
    h_dt = exp_factor * h0
    h_dt = h_dt + ((1 - exp_factor) * M)
    return h_dt

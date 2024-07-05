from typing import List, Optional, Union, Literal, Callable

import numpy as np
from pydantic import Field, computed_field, validate_call, ConfigDict
from scipy.interpolate import interp1d

from asoh.models.base import HealthVariable


class SOCInterpolatedHealth(HealthVariable):
    """Defines basic functionality for HealthVariables that need interpolation between SOC pinpoints
    """
    base_values: Union[float, np.ndarray] = \
        Field(default=0,
              description='Values at specified SOCs')
    soc_pinpoints: Optional[np.ndarray] = Field(default=None, description='SOC pinpoints for interpolation.')
    interpolation_style: \
        Literal['linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                'quadratic', 'cubic', 'previous', 'next'] = \
        Field(default='linear', description='Type of interpolation to perform')
    updatable: set[str] = Field(default_factory=lambda: {'base_values'})

    # Let us cache the interpolation function so we don't have to re-do it every
    # time we want to get a value
    @computed_field
    @property
    def _interp_func(self) -> Callable:
        """
        Interpolate values. If soc_pinpoints have not been set, assume
        internal_parameters are evenly spread on an SOC interval [0,1].
        """
        if self.soc_pinpoints is None:
            self.soc_pinpoints = np.linspace(0, 1, len(self.base_values))
        func = interp1d(self.soc_pinpoints,
                        self.base_values,
                        kind=self.interpolation_style,
                        bounds_error=False,
                        fill_value='extrapolate')
        return func

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_value(self,
                  soc: Union[float, List, np.ndarray]) -> Union[float, np.ndarray]:
        """Computes value(s) at given SOC(s)

        Args:
            soc: Values at which to compute the property
        Returns:
            Interpolated values
        """
        if isinstance(self.base_values, float):
            return self.base_values
        return self._interp_func(soc)


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
        h0: float,
        M: float,
        kappa: float,
        dt: float,
        i0: float,
        alpha: float) -> float:
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
    exp_factor *= (i0 + (0.5 * alpha * dt))
    # Now, flip the sign depending if current is positive in the interval
    if i0 > -(alpha * dt):  # this indicates (i0 + alpha * t) > 0
        exp_factor = -exp_factor
    exp_factor = np.exp(exp_factor)
    h_dt = exp_factor * h0
    h_dt += (1 - exp_factor) * M
    return h_dt

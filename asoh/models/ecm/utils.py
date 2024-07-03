
# General imports
from typing import List, Optional, Union, Literal, Callable
import numpy as np
from pydantic import Field, computed_field, validate_call, ConfigDict
from scipy.interpolate import interp1d

# ASOH imports
from asoh.models.base import HealthVariable


class SOCInterpolatedHealth(HealthVariable):
    """
    Defines basic functionality for HealthVariables that need interpolation
    between SOC pinpoints
    """
    base_values: Union[float, List] = \
        Field(default=0,
              description='Values at specified SOCs')
    soc_pinpoints: Optional[List] = \
        Field(default=[], description='SOC pinpoints for interpolation.')
    interpolation_style: \
        Literal['linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                'quadratic', 'cubic', 'previous', 'next'] = \
        Field(default='linear', description='Type of interpolation to perform')

    # Let us cache the interpolation function so we don't have to re-do it every
    # time we want to get a value
    @computed_field
    @property
    def _interp_func(self) -> Callable:
        """
        Interpolate values. If soc_pinpoints have not been set, assume
        internal_parameters are evenly spread on an SOC interval [0,1].
        """
        if not len(self.soc_pinpoints):
            self.soc_pinpoints = np.linspace(0, 1, len(self.base_values))
        func = interp1d(self.soc_pinpoints,
                        self.base_values,
                        kind=self.interpolation_style,
                        bounds_error=False,
                        fill_value='extrapolate')
        return func

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def value(self,
              soc: Union[float, List, np.ndarray],
              *args, **kwargs) -> Union[float, np.ndarray]:
        """
        Computes value(s) at given SOC(s)
        """
        if isinstance(self.base_values, float):
            return self.base_values
        return self._interp_func(soc)

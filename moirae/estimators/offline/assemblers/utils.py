"""
Utility functions for assemblers
"""
from typing import Callable, Dict, List, Literal

import numpy as np
from scipy.interpolate import make_interp_spline, make_smoothing_spline, make_lsq_spline
from sklearn.isotonic import IsotonicRegression


class SOCRegressor():
    """
    Helper to deal with cases where variable is SOC dependent and we wish to fit a functional form to the extracted data

    Args:
        style: establishes the style of interpolation; options include:
            - 'interpolate': linearly interpolates between the points provided, which are fully captured in the final
                function
            - 'smooth': fits a smoothing spline through the data points, where regularization penalizes the
                integral of the squared second derivative function.
            - 'lsq': fits a Least-Squares spline to minimize the least squares errors; requires knots to be specified
            - 'isotonic': fits an isotonic curve, that is, a non-linear monotonic trend
        parameters: Additional parameters needed to create fits; should not be required at fitting time
    """
    def __init__(self,
                 style: Literal['interpolate', 'smooth', 'lsq', 'isotonic'] = 'interpolate',
                 parameters: Dict = {}):
        self.style = style
        self.parameters = parameters

    @property
    def accepted_styles(self) -> List[str]:
        """
        Defines acceptable interpolation styles
        """
        return ['interpolate', 'smooth', 'lsq', 'isotonic']

    def acceptable_keywords(self) -> List[str]:
        """
        Specifies what keywords are acceptable given the present style chosen
        """
        if self._style == 'interpolate':
            return ['k', 't', 'bc_type', 'axis', 'check_finite']
        elif self._style == 'smooth':
            return ['lam', 'axis']
        elif self._style == 'lsq':
            return ['k', 'w', 'axis', 'check_finite', 'method']
        elif self._style == 'isotonic':
            return ['y_min', 'y_max', 'increasing', 'out_of_bounds']

    @property
    def style(self) -> Literal['interpolate', 'smooth', 'lsq', 'isotonic']:
        return self._style

    @style.setter
    def style(self, value: Literal['interpolate', 'smooth', 'lsq']):
        if value not in self.accepted_styles:
            message = 'Accepted fitting styles are '
            message += ', '.join(self.accepted_styles)
            message += f', not {value}!'
            raise ValueError(message)
        self._style = value

    @property
    def parameters(self) -> Dict:
        return self._params.copy()

    @parameters.setter
    def parameters(self, params: Dict):
        if not isinstance(params, Dict):
            raise TypeError('Parameters must be provided as a dictionary!')
        for key in params.keys():
            if key not in self.acceptable_keywords():
                message = f'Acceptable parameters for {self.style} are '
                message += ', '.join(self.acceptable_keywords()) + f', not {key}!'
                raise ValueError(message)
        self._params = params.copy()

    def fit(self, soc: np.ndarray, targets: np.ndarray, **kwargs) -> Callable:
        """
        Fits a function that tries to match the SOC values provided to the target values. Additional keywords are used
        to update the construction of the fitting.

        Args:
            soc: values of SOC to be used
            targets: values the function is trying to match
            **kwargs: additional arguments that will be used in the construction of the regressor

        Returns:
            function that takes SOC values and outputs predictions
        """
        # Get the parameters
        params = self.parameters

        if self._style == 'isotonic':
            # If sample weights are present, they need to be treated separately
            sample_weight = None
            for key, val in kwargs.items():
                if key == 'sample_weight':
                    sample_weight = val
                else:
                    params.update({key: val})
            # Build regression
            regressor = IsotonicRegression(**params)
            # Fit
            regressor.fit(X=soc, y=targets, sample_weight=sample_weight)
            return regressor.predict

        # Now, we can update the parameters
        params.update(kwargs)

        if self._style == 'interpolate':
            return make_interp_spline(x=soc, y=targets, **params)
        elif self._style == 'smooth':
            return make_smoothing_spline(x=soc, y=targets, **params)
        elif self._style == 'lsq':
            if 't' not in params.keys():
                raise ValueError('Please provide knots for LSQ Spline!')
            return make_lsq_spline(x=soc, y=targets, **params)

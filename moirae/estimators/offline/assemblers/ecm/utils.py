"""
Utility functions for assembling ECM-related Health Variables from Extracted Parameters
"""
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from moirae.models.ecm.components import SOCInterpolatedHealth
from moirae.estimators.offline.extractors.base import ExtractedParameter
from moirae.estimators.offline.assemblers.base import BaseAssembler
from moirae.estimators.offline.assemblers.utils import SOCRegressor


def post_process_extracted(extracted_parameter: ExtractedParameter,
                           weights: Optional[np.ndarray] = None) -> ExtractedParameter:
    """
    Post-processes extracted parameters, ensuring they are sorted and contain no repeated SOC entries.

    Duplicated SOC values are averaged

    Args:
        extracted_parameter: properly formatted dictionary with the Extracted parameter
        weights: weights to be used in averaging

    Returns:
        a cleaned-up version of the extracted parameters
    """
    # To make things easy, we will cast everything as a pandas dataframe
    df = pd.DataFrame.from_dict(extracted_parameter)

    # We don't need to keep the units there
    df.drop('units', axis=1, inplace=True)

    # Prepare weights
    if weights is None:
        w = np.ones(len(df))
    else:
        w = weights.copy()
    df.loc[:, 'weights'] = w

    # Now, we will sort by SOC
    df.sort_values(by='soc_level', inplace=True)

    # Average the repeated values
    cols_to_avg = [col for col in df.columns if (col != 'soc_level')]
    # Function to compute average of the group
    def weighted_avg(g, cols):
        return pd.Series(np.average(g[cols], weights=g['weights'], axis=0), index=cols)
    df = df.groupby('soc_level').apply(weighted_avg, cols=cols_to_avg).reset_index()
    if weights is None:
        df.drop('weights', axis=1, inplace=True)

    # Return to dictionary format
    info = df.to_dict(orient='list')  # 'list' ensures we skip the index of the DataFrame
    info['units'] = extracted_parameter['units']
    return info


class SOCDependentAssembler(BaseAssembler):
    """
    Assembles an Assember object from extracted parameters.

    Args:
        regressor: object to perform SOC regression
        soc_points: points in the SOC domain to be used when assembling the HealthVariable object
    """
    def __init__(self,
                 regressor: SOCRegressor = SOCRegressor(),
                 soc_points: Union[np.ndarray, int] = 11):
        self.regressor = regressor
        self.soc_points = soc_points

    @property
    def soc_points(self) -> np.ndarray:
        return self._soc_pts

    @soc_points.setter
    def soc_points(self, array: Union[np.ndarray, int]):
        if isinstance(array, int):
            soc_pts = np.linspace(0., 1., array)
        else:
            soc_pts = array.copy()
        min_val = min(soc_pts)
        max_val = max(soc_pts)
        if (min_val > 0.) or (max_val < 1.):
            raise ValueError(f'SOC domain not fully covered, only {100*min_val:.1f} -- {100*max_val:.1f}%!')
        self._soc_pts = np.array(soc_pts)

    def _compute_weights(self, extracted_parameter: ExtractedParameter) -> Union[None, np.ndarray]:
        """
        Auxiliary function to compute relevant weights when needed. Defaults to retuning None, which assigns equal
        weights to all observations.

        Args:
            extracted_parameter: dictionary containing information about the extracted parameter

        Returns:
            corresponding weights for each value in the extracted parameter dictionary
        """
        return None

    def _prepare_for_regression(self, extracted_parameter: ExtractedParameter) -> Dict:
        """
        Auxiliary function to prepare dictionary to be given directly to the regressor at the time of fitting

        Args:
            extracted_parameter: dictionary containing information about the extracted parameter

        Returns:
            dictionary of keywords to be used
        """
        # Clean-up the parameters
        weights = self._compute_weights(extracted_parameter=extracted_parameter)
        clean_param = post_process_extracted(extracted_parameter=extracted_parameter, weights=weights)

        regression_dict = {'soc': clean_param['soc_level'],
                           'targets': clean_param['value']}

        for key in clean_param.keys():
            if (key != 'units') and (key != 'soc_level') and (key != 'value'):
                regression_dict[key] = clean_param[key]

        # Make sure to add knots in the case of LSQ Univariate regression!
        if self.regressor.style == 'lsq':
            regression_dict['t'] = self._soc_pts
        print(regression_dict.keys())

        return regression_dict

    def assemble(self, extracted_parameter: ExtractedParameter) -> SOCInterpolatedHealth:
        # Get dictionary of parameters to fit
        regression_dict = self._prepare_for_regression(extracted_parameter=extracted_parameter)

        # Fit
        fit = self.regressor.fit(**regression_dict)

        # Compute values at desired points
        vals = fit(self._soc_pts)

        return SOCInterpolatedHealth(base_values=vals, soc_pinpoints=self._soc_pts)

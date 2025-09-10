"""
Base Definitions for Offline Estimators
"""
from typing import Any, Tuple, Union

import pandas as pd

from battdat.data import BatteryDataset

from moirae.models.base import GeneralContainer, HealthVariable


class BaseOfflineEstimator():
    """
    Base class for tools that perform full offline estimation from data, which can be provided as a raw pandas
    DataFrame, or as a BatteryDataset
    """
    def estimate(self,
                 data: Union[pd.DataFrame, BatteryDataset],
                 *args, **kwargs) -> Tuple[GeneralContainer, HealthVariable, Any]:
        """
        Estimates the initial transient state of the battery, as well as its aSOH in the period provided.

        Note: most Offline Estimators will assume the aSOH is constant throughout the dataset provided

        Args:
            data: data to be used in estimation
            *args: additional arguments needed for estimation
            **kwargs: keyword arguments to further specify estimation

        Returns:
            - Estimate for the initial state
            - Estimate for the ASOH parameters
            - Diagnostic unique to the type of estimator
        """
        raise NotImplementedError('Please implement in child classes!')

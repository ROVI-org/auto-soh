"""Interface for extractors"""
from typing import Union

import pandas as pd

from battdat.data import BatteryDataset

from moirae.models.base import HealthVariable


class DataCheckError(ValueError):
    """
    Custom exception to be used when we encounter issues when checking the provided data for features to extract
    """
    def __init__(self, message: str = "Data check failed!"):
        super().__init__(message)


class BaseDataChecker():
    """
    Base class for tools to check if data being used is appropriate for extractors
    """

    def check(self, data: Union[pd.DataFrame, BatteryDataset]) -> None:
        """
        Verify whether data contains features needed for algorithm

        Args:
            data: Data to be evaluated
        Raises:
            (DataCheckError) If the dataset is missing critical information
        """
        pass  # Default: all data is valid

class BaseExtractor:
    """Base class for tools which determine parameters from special cycles"""

    def extract(self, data: Union[pd.DataFrame, BatteryDataset]) -> HealthVariable:
        """Determine parameters of a physics model from battery dataset or from raw data, which must follow the format
        defined in `RawData format`_ of battery-data-toolkit RawData, given at
        https://rovi-org.github.io/battery-data-toolkit/user-guide/schemas/column-schema.html#rawdata

        Args:
            data: Data to use for parameter assessment

        Returns:
            Part of the parameter set for a model.
        """
        raise NotImplementedError()

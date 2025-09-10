from typing import Union

import pandas as pd

from battdat.data import BatteryDataset

from .utils import ensure_battery_dataset


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

    def check(self, data: Union[pd.DataFrame, BatteryDataset]) -> BatteryDataset:
        """
        Verify whether data contains features needed for algorithm

        Args:
            data: Data to be evaluated

        Raises:
            (DataCheckError) If the dataset is missing critical information

        Returns:
            corresponding `BatteryDataset` object if data is appropriate; can include additional processing
        """
        return ensure_battery_dataset(data=data)  # Default: all data is valid

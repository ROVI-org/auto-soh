"""Interface for extractors"""
from typing import List, TypedDict, Union

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset


class ExtractedParameter(TypedDict):
    """
    Defines how extracted parameters are reported

    Args:
        value: extracted values
        units: unit of measurement for values
        soc_level: SOC level for values, if appropriate
    """
    value: Union[float, np.ndarray, List]
    units: str
    soc_level: Union[List, np.ndarray]


class BaseExtractor:
    """
    Base class for tools which determine parameters from special cycles
    """

    def extract(self, data: Union[pd.DataFrame, BatteryDataset]) -> ExtractedParameter:
        """Determine parameters of a physics model from battery dataset or from raw data, which must follow the format
        defined in `RawData format`_ of battery-data-toolkit RawData, given at
        https://rovi-org.github.io/battery-data-toolkit/user-guide/schemas/column-schema.html#rawdata

        Args:
            data: Data to use for parameter assessment

        Returns:
            Dictionary containing extracted values, units, and, when appropriate, additional information (such as the
            corresponding SOC levels for reported values)
        """
        raise NotImplementedError()

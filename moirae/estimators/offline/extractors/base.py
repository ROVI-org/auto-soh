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
    def compute_parameters(self, data: Union[pd.DataFrame, BatteryDataset], *args, **kwargs) -> ExtractedParameter:
        """
        Function to compute parameters, assuming the data being provided has already been appropriately vetted for
        extraction.

        Args:
            data: Data to use for parameter assessment as either raw dataframe, or BatterDataset object
            *args: necessary ordered arguments for the calculation of parameters
            **kwargs: keyword arguments required for the calculation of parameters

        Returns:
            extracted parameter(s)
        """
        raise NotImplementedError('Please implement in child classes!')

    def extract(self, data: Union[pd.DataFrame, BatteryDataset], *args, **kwargs) -> ExtractedParameter:
        """
        Determine parameters of a physics model from battery dataset or from raw data, which must follow the format
        defined in `RawData format`_ of battery-data-toolkit RawData, given at
        https://rovi-org.github.io/battery-data-toolkit/user-guide/schemas/column-schema.html#rawdata

        The data used will be checked and processed first. If data checks are deemed unnecessary, please use the method
        `compute_parameters` instead.

        Args:
            data: Data to use for parameter assessment
            *args: necessary ordered arguments for the extraction
            **kwargs: keyword arguments required for the extraction

        Returns:
            Dictionary containing extracted values, units, and, when appropriate, additional information (such as the
            corresponding SOC levels for reported values)
        """
        raise NotImplementedError('Please implement in child classes!')

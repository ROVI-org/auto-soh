"""Interface for extractors"""
from typing import Dict, Union

import pandas as pd

from battdat.data import BatteryDataset


class BaseExtractor:
    """Base class for tools which determine parameters from special cycles"""

    def extract(self, data: Union[pd.DataFrame, BatteryDataset]) -> Dict:
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

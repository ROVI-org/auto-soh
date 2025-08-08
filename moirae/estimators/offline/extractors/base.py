"""Interface for extractors"""
from typing import Union

import pandas as pd

from battdat.data import BatteryDataset

from moirae.models.base import HealthVariable


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

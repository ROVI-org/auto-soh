"""
Utility tools for extractors
"""
from typing import Union

import pandas as pd

from battdat.data import BatteryDataset


def ensure_battery_dataset(data: Union[pd.DataFrame, BatteryDataset]) -> BatteryDataset:
    """
    Utility function to make sure we are always dealing with `BatteryDataset` objects

    Args:
        data: data to be checked and converted if necessary

    Returns:
        `BatteryDataset` object corresponding to provided data
    """
    if isinstance(data, BatteryDataset):
        return data
    return BatteryDataset.make_cell_dataset(raw_data=data)

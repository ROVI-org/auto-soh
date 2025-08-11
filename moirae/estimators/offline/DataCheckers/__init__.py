"""
Tools to help check if data being used is appropriate for offline estimation
"""
from typing import Union

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.integral import CapacityPerCycle

from moirae.models.ecm.components import MaxTheoreticalCapacity
from moirae.estimators.offline.DataCheckers.base import BaseDataChecker, DataCheckError
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset


class SingleCycleChecker(BaseDataChecker):
    """
    Ensures the data provided contains only a single cycle.
    """
    def __init__(self):
        pass

    def check(self, data: Union[pd.DataFrame, BatteryDataset]) -> None:
        # Ensure we have a BatteryDataset
        data = ensure_battery_dataset(data)

        # Get only raw data
        raw_data = data.tables.get('raw_data')
        if raw_data is None:
            raise DataCheckError("Raw data table not found in dataset!")

        # Ensure a single cycle
        if len(raw_data['cycle_number'].unique()) > 1:
            raise DataCheckError("Multiple cycles found in data! Please provide a single cycle.")


class DeltaSOCRangeChecker(SingleCycleChecker):
    """
    Ensures the cycle provided has covered a sufficiently large range of SOC values.

    Args:
        capacity: Assumed cell capacity in Amp-hours
        min_delta_soc: Minimum required SOC change; defaults to 10%
        
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 min_delta_soc: float = 0.1):
        self.min_delta_soc = min_delta_soc
        self.capacity = capacity
    
    @property
    def min_delta_soc(self) -> float:
        """Minimum required SOC change"""
        return self._min_delta_soc
    @min_delta_soc.setter
    def min_delta_soc(self, value: float):
        if value < 0 or value > 1:
            raise ValueError("Minimum SOC change must be in the range [0, 1].")
        self._min_delta_soc = value
    
    @property
    def capacity(self) -> float:
        """Assumed cell capacity in Amp-hours"""
        return self._capacity
    @capacity.setter
    def capacity(self, value: Union[float, MaxTheoreticalCapacity]):
        if isinstance(value, MaxTheoreticalCapacity):
            self._capacity = value.amp_hour.item()
        elif isinstance(value, (int, float)):
            if value <= 0:
                raise ValueError("Capacity must be a positive number!")
            self._capacity = float(value)
        else:
            raise TypeError("Capacity must be a float or MaxTheoreticalCapacity object!")


    def check(self, data: Union[pd.DataFrame, BatteryDataset]) -> None:
        # Ensure we have a BatteryDataset
        data = ensure_battery_dataset(data)

        # Check for single cycle
        super().check(data)       
        
        if data.tables.get('cycle_stats') is None or 'max_cycled_capacity' not in data.tables['cycle_stats'].columns:
            CapacityPerCycle().add_summaries(data)
        
        # Compute range of SOC values sampled
        sampled_soc_range = data.tables['cycle_stats']['max_cycled_capacity'].max() / self.capacity
        if sampled_soc_range < self.min_delta_soc:
            raise DataCheckError(f"Dataset must sample at least {self.min_delta_soc * 100:.1f}% of SOC. "
                                 f"Only sampled {sampled_soc_range * 100:.1f}%.")

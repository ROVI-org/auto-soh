"""
Tools to help check if data being used is appropriate for offline estimation
"""
from typing import Union

import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.integral import StateOfCharge

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

        return data


class DeltaSOCRangeChecker(SingleCycleChecker):
    """
    Ensures the cycle provided has covered a sufficiently large range of SOC values.

    Args:
        capacity: Assumed cell capacity in Amp-hours
        coulombic_efficiency: Assumed Coulombic efficiency of the cell; defaults to 1.0 (100%)
        min_delta_soc: Minimum required SOC change; defaults to 10%
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 coulombic_efficiency: float = 1.0,
                 min_delta_soc: float = 0.1):
        self.min_delta_soc = min_delta_soc
        self.capacity = capacity
        self.coulombic_efficiency = coulombic_efficiency

    @property
    def min_delta_soc(self) -> float:
        """Minimum required SOC change"""
        return self._min_delta_soc

    @min_delta_soc.setter
    def min_delta_soc(self, value: float):
        if value < 0 or value > 1:
            raise ValueError("Minimum SOC change must be positive and <= 1.")
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

    @property
    def coulombic_efficiency(self) -> float:
        """Coulombic efficiency of the cell"""
        return self._ce

    @coulombic_efficiency.setter
    def coulombic_efficiency(self, value: float):
        if value < 0 or value > 1:
            raise ValueError("Coulombic efficiency must be between 0 and 1.")
        self._ce = value

    def check(self, data: Union[pd.DataFrame, BatteryDataset]) -> BatteryDataset:
        # Check for single cycle
        data = super().check(data)

        # Compute SOC range if needed
        raw_data = data.tables.get('raw_data')
        if 'CE_adjusted_charge' not in raw_data.columns:
            StateOfCharge(coulombic_efficiency=self._ce).enhance(data=raw_data)
        sampled_soc_range = (raw_data['CE_adjusted_charge'].max() - raw_data['CE_adjusted_charge'].min())
        sampled_soc_range /= self.capacity

        if sampled_soc_range < self.min_delta_soc:
            raise DataCheckError(f"Dataset must sample at least {self.min_delta_soc * 100:.1f}% of SOC. "
                                 f"Only sampled {sampled_soc_range * 100:.1f}%.")

        # Include these modifications back in the data
        data.tables['raw_data'] = raw_data

        return data

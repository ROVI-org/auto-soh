"""
Data checkers specifically for Rereference Performance Tests (RPT) data.
"""
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.tagging import AddState, AddSteps
from battdat.schemas.column import ChargingState

from moirae.estimators.offline.DataCheckers.base import BaseDataChecker, DataCheckError
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset


class CapacityDataChecker(BaseDataChecker):
    """
    Ensures the cycle provided is representative of a capacity check diagnostic cycle, that is, it:
    1. Contains a full charge and discharge, meaning it reaches both upper and lower voltage limits
    2. Is performed at a low enough current

    Args:
        voltage_limits: Tuple of (min_voltage, max_voltage) to check against; if not provided, does not check voltage
        max_C_rate: Maximum approximate C-rate for the cycle to be considered a capacity check; deafults to C/10
        voltage_tolerance: Tolerance for voltage limits, defaults to 1 mV
    """
    def __init__(self,
                 voltage_limits: Optional[Tuple[float, float]] = None,
                 max_C_rate: float = 0.1,
                 voltage_tolerance: float = 0.001):
        self.voltage_limits = voltage_limits
        self.max_C_rate = max_C_rate
        self.voltage_tolerance = voltage_tolerance

    def check(self, data: Union[pd.DataFrame, BatteryDataset]) -> None:
        # Ensure we have a BatteryDataset
        data = ensure_battery_dataset(data)

        # Get only raw data
        raw_data = data.tables.get('raw_data')
        if raw_data is None:
            raise DataCheckError("Raw data table not found in dataset!")

        # Check if we reach the voltage limits
        if self.voltage_limits is not None:
            min_voltage, max_voltage = sorted(self.voltage_limits)
            if not np.allclose(raw_data['voltage'].min(), min_voltage, atol=self.voltage_tolerance):
                raise DataCheckError(f"Cycle does not reach lower voltage limit of {min_voltage:1.2f} V!")
            if not np.allclose(raw_data['voltage'].max(), max_voltage, atol=self.voltage_tolerance):
                raise DataCheckError(f"Cycle does not reach upper voltage limit of {max_voltage:1.2f} V!")
        
        # Now, let's make sure we can find at least one charge and one discharge segments which are slow enough
        if 'state' not in raw_data.columns:
            AddState().enhance(raw_data)
        if 'step_index' not in raw_data.columns:
            AddSteps().enhance(raw_data)

        # Get relevant segments
        charge_segments = raw_data[raw_data['state'] == ChargingState.charging]
        discharge_segments = raw_data[raw_data['state'] == ChargingState.discharging]
        # Compute the min duration of a full charge or discharge from the C-rate
        duration_seconds = 3600. / self.max_C_rate  # seconds
        # Create auxiliary lists to store whether we found valid segments
        segments_data = [charge_segments, discharge_segments]
        found_valid = [False, False]
        # Go through segments and see if we can find a valid one
        for i, segments in enumerate(segments_data):
            for (_, step_data) in segments.groupby('step_index'):
                step_duration = step_data['test_time'].max() - step_data['test_time'].min()
                if step_duration >= duration_seconds:
                    candidate = True
                    # Check voltage limits if needed
                    if self.voltage_limits is not None:
                        min_voltage, max_voltage = sorted(self.voltage_limits)
                        if not np.allclose(step_data['voltage'].min(), min_voltage, atol=self.voltage_tolerance) or \
                           not np.allclose(step_data['voltage'].max(), max_voltage, atol=self.voltage_tolerance):
                            candidate = False
                    found_valid[i] = candidate
                    if found_valid[i]:
                        break
        
        if not found_valid[0]:
            raise DataCheckError(f"Cycle does not contain a valid charge segment at C/{1./self.max_C_rate:.1f} or lower!")
        if not found_valid[1]:
            raise DataCheckError(f"Cycle does not contain a valid discharge segment at C/{1./self.max_C_rate:.1f} or lower!")
        return

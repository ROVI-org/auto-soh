from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.tagging import AddState, AddSteps
from battdat.schemas.column import ChargingState

from moirae.estimators.offline.DataCheckers.base import BaseDataChecker, DataCheckError
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset


class HPPCDataChecker(BaseDataChecker):
    """
    Ensures the cycle provided is representative of a Hybrid Pulse Power Characterization (HPPC) diagnostic cycle

    Args:
        voltage_limits: Tuple of (min_voltage, max_voltage) to check against; if not provided, does not check voltage
        voltage_tolerance: Tolerance for voltage limits, defaults to 1 mV
        min_pulses: Minimum number of pulses (both charge and discharge) required for the cycle to be considered a HPPC
    """
    def __init__(self,
                 voltage_limits: Optional[Tuple[float, float]] = None,
                 voltage_tolerance: float = 0.001,
                 min_pulses: int = 1):
        self.voltage_limits = voltage_limits
        self.voltage_tolerance = voltage_tolerance
        self.min_pulses = min_pulses

    def check(self, data: Union[pd.DataFrame, BatteryDataset]) -> None:
        # Ensure we have a BatteryDataset
        data = ensure_battery_dataset(data)

        # Get only raw data
        raw_data = data.tables.get('raw_data')
        if raw_data is None:
            raise DataCheckError("Raw data table not found in dataset!")
        
        # Ensure a single cycle
        if len(raw_data['cycel_numer'].unique()) > 1:
            raise DataCheckError("Multiple cycles found in data! Please provide a single cycle.")

        # Check if we reach the voltage limits
        if self.voltage_limits is not None:
            min_voltage, max_voltage = sorted(self.voltage_limits)
            if not np.allclose(raw_data['voltage'].min(), min_voltage, atol=self.voltage_tolerance):
                raise DataCheckError(f"Cycle does not reach lower voltage limit of {min_voltage:1.2f} V!")
            if not np.allclose(raw_data['voltage'].max(), max_voltage, atol=self.voltage_tolerance):
                raise DataCheckError(f"Cycle does not reach upper voltage limit of {max_voltage:1.2f} V!")
        
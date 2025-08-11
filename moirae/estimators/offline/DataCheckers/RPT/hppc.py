from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.tagging import AddState, AddSteps, AddMethod, AddSubSteps
from battdat.schemas.column import ChargingState

from moirae.models.ecm.components import MaxTheoreticalCapacity
from moirae.estimators.offline.DataCheckers import DeltaSOCRangeChecker, DataCheckError
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset


class HPPCDataChecker(DeltaSOCRangeChecker):
    """
    Ensures the cycle provided is representative of a Hybrid Pulse Power Characterization (HPPC) diagnostic cycle

    Args:
        capacity: Assumed cell capacity in Amp-hours
        min_delta_soc: Minimum required SOC change; defaults to 10%
        min_pulses: Minimum number of pulses (both charge and discharge) required for the cycle to be considered a HPPC
        ensure_bidirectional: If True, ensures that both charge and discharge pulses are present
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 min_delta_soc: float = 0.1,
                 min_pulses: int = 1,
                 ensure_bidirectional: bool = True):
        super().__init__(capacity=capacity, min_delta_soc=min_delta_soc)
        self.min_pulses = min_pulses
        self.ensure_bidirectional = ensure_bidirectional

    def check(self, data: Union[pd.DataFrame, BatteryDataset]) -> None:
        # Ensure we have a BatteryDataset
        data = ensure_battery_dataset(data)

        # Make sure we have a single cycle and the SOC range is sufficient
        super().check(data=data)

        # Get only raw data
        raw_data = data.tables.get('raw_data')

        # Ensure we have the necessary state, step, and method columns
        if 'state' not in raw_data.columns:
            AddState().enhance(raw_data)
        if 'step_index' not in raw_data.columns:
            AddSteps().enhance(raw_data)
        if 'method' not in raw_data.columns:
            AddMethod().enhance(raw_data)
        if 'sub_step_index' not in raw_data.columns:
            AddSubSteps().enhance(raw_data)
        
        # Now, make sure we can find at least `min_pulses`
        pulses = raw_data[raw_data['method'] == 'pulse']
        num_observed_pulses = len(pulses['substep_index'].unique())
        if num_observed_pulses < self.min_pulses:
            raise DataCheckError(f"Cycle contains only {num_observed_pulses} pulses; "
                                 f"expected at least {self.min_pulses}!")
        
        # Check if we have bidirectional pulses if required
        if self.ensure_bidirectional:
            charge_pulses = pulses[pulses['state'] == ChargingState.charging]
            discharge_pulses = pulses[pulses['state'] == ChargingState.discharging]
            num_charge_pulses = len(charge_pulses['substep_index'].unique())
            num_discharge_pulses = len(discharge_pulses['substep_index'].unique())
            if num_charge_pulses == 0:
                raise DataCheckError("No charge pulses found in the cycle!")
            if num_discharge_pulses == 0:
                raise DataCheckError("No discharge pulses found in the cycle!")
            # Make sure they don't differ by more than 1, which only happens if we are close to a voltage limit
            if abs(num_charge_pulses - num_discharge_pulses) > 1:
                raise DataCheckError(f"Found {num_charge_pulses} charge and {num_discharge_pulses} discharge pulses; "
                                     f"HPPC is not bi-directional!")
        
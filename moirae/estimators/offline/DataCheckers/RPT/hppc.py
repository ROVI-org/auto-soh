from typing import List, Union

import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.tagging import AddState, AddSteps, AddMethod, AddSubSteps
from battdat.schemas.column import ChargingState

from moirae.models.ecm.components import MaxTheoreticalCapacity
from moirae.estimators.offline.DataCheckers import DeltaSOCRangeChecker, DataCheckError
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset


class PulseDataChecker(DeltaSOCRangeChecker):
    """
    Ensures the cycle provided is representative of a Hybrid Pulse Power Characterization (HPPC) diagnostic cycle, that
    is, it contains a sufficient number of pulses and covers a sufficient SOC range.

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

    def check(self,
              data: Union[pd.DataFrame, BatteryDataset],
              extract: bool = False) -> Union[None, List[pd.DataFrame]]:
        """
        Verify whether data contains pulses

        Args:
            data: Data to be evaluated
            extract: flag to indicate if pulses should be returned as a list of DataFrames for further processing;
                defaults to False

        Raises:
            (DataCheckError) If the dataset is missing critical information

        Returns:
            If `extract` is True, returns a list of DataFrames containing the pulses; otherwise, returns None
        """
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

        # Return pulses if requested
        if extract:
            return [group[1] for group in pulses.groupby('substep_index')]


class RestDataChecker(DeltaSOCRangeChecker):
    """
    Ensures the cycle provided contains rest period.

    Args:
        capacity: Assumed cell capacity in Amp-hours
        min_delta_soc: Minimum required SOC change; defaults to 10%
        min_rest_duration: Minimum duration of rest period in seconds; defaults to 600s (10 min)
        min_number_of_rests: Minimum number of rest periods required; defaults to 1
        rest_current_threshold: Maximum absolute current to consider as rest; defaults to 1 mA
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 min_delta_soc: float = 0.1,
                 min_number_of_rests: int = 1,
                 min_rest_duration: float = 600.,
                 rest_current_threshold: float = 1.0e-04):
        super().__init__(capacity=capacity, min_delta_soc=min_delta_soc)
        self.min_rests = min_number_of_rests
        self.min_rest_duration = min_rest_duration
        self.rest_current_threshold = rest_current_threshold

    @property
    def min_rests(self) -> int:
        """Minimum number of rest periods required"""
        return self._min_rests

    @min_rests.setter
    def min_rests(self, value: int):
        if value < 1:
            raise ValueError("Minimum number of rests must be at least 1!")
        self._min_rests = value

    @property
    def min_rest_duration(self) -> float:
        """Minimum duration of rest period in seconds"""
        return self._min_rest_dur

    @min_rest_duration.setter
    def min_rest_duration(self, value: float):
        if value <= 0:
            raise ValueError("Minimum rest duration must be positive!")
        self._min_rest_dur = value

    @property
    def rest_current_threshold(self) -> float:
        """Maximum absolute current to consider as rest"""
        return self._curr_thresh

    @rest_current_threshold.setter
    def rest_current_threshold(self, value: float):
        self._curr_thresh = abs(value)

    def check(self,
              data: Union[pd.DataFrame, BatteryDataset],
              extract: bool = False) -> Union[None, List[pd.DataFrame]]:
        """
        Verify whether data contains rests of sufficient duration

        Args:
            data: Data to be evaluated
            extract: flag to indicate if pulses should be returned as a list of DataFrames for further processing;
                defaults to False

        Raises:
            (DataCheckError) If the dataset is missing critical information

        Returns:
            If `extract` is True, returns a list of DataFrames containing the pulses; otherwise, returns None
        """
        # Ensure we have a BatteryDataset
        data = ensure_battery_dataset(data)

        # Make sure we have a single cycle and the SOC range is sufficient
        super().check(data=data)

        # Get only raw data
        raw_data = data.tables.get('raw_data')

        # Ensure we have the necessary state, step, and method columns
        if 'state' not in raw_data.columns:
            AddState(rest_curr_threshold=self.rest_current_threshold).enhance(raw_data)
        if 'step_index' not in raw_data.columns:
            AddSteps().enhance(raw_data)

        # Now, let's find the rest periods
        rest_data = raw_data[raw_data['state'] == ChargingState.rest]
        rest_periods = []
        for _, group in rest_data.groupby('step_index'):
            duration = group['test_time'].iloc[-1] - group['test_time'].iloc[0]
            if duration >= self.min_rest_duration:
                rest_periods.append(group)

        if len(rest_periods) < self.min_rests:
            raise DataCheckError(f"Cycle contains only {len(rest_periods)} rest periods of at least "
                                 f"{self.min_rest_duration:.1f} seconds; expected at least {self.min_rests:d}!")

        # Return rest periods if requested
        if extract:
            return rest_periods
